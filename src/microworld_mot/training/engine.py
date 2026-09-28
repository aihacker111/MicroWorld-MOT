from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from ..association.sinkhorn import sinkhorn_loss
from ..config import ExperimentConfig, save_config
from ..data import build_datasets
from ..losses import box_iou_aligned, gaussian_mixture_nll, masked_mean
from ..models import build_system
from ..models.registry import resolved_config
from ..models.system import MicroWorldMOT


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _to_device(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _binary_loss(probability: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    # PyTorch deliberately rejects probability-space BCE under CUDA autocast:
    # its FP16 backward can overflow close to 0/1. These model heads already
    # return probabilities, so keep their public contract and evaluate BCE in
    # FP32 outside autocast instead of applying a sigmoid twice.
    with torch.autocast(device_type=probability.device.type, enabled=False):
        values = F.binary_cross_entropy(
            probability.float().clamp(1e-5, 1 - 1e-5),
            target.float(),
            reduction="none",
        )
    return masked_mean(values, mask)


def _identity_contrastive_loss(
    track_features: Tensor,
    target_features: Tensor,
    valid: Tensor,
    temperature: float = 0.10,
) -> Tensor:
    track_features = F.normalize(track_features, dim=-1)
    target_features = F.normalize(target_features, dim=-1)
    logits = torch.einsum("bnd,bmd->bnm", track_features, target_features) / temperature
    logits = logits.masked_fill(~valid.unsqueeze(1), -1e4)
    targets = torch.arange(logits.shape[1], device=logits.device).unsqueeze(0)
    targets = targets.expand(logits.shape[0], -1)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
    ).reshape_as(valid)
    return masked_mean(loss, valid)


def _object_jepa_loss(
    latent_prediction: Tensor | None,
    target_latent: Tensor,
    valid: Tensor,
) -> Tensor:
    """V-JEPA-style stop-gradient L1 prediction in sparse object space."""
    if latent_prediction is None:
        return target_latent.new_zeros(())
    return masked_mean((latent_prediction - target_latent).abs().mean(dim=-1), valid)


def _aligned_observations(
    batch: dict[str, Tensor], time_index: int
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    assignment = batch["assignment"][:, time_index]
    safe_assignment = assignment.clamp_min(0)
    boxes = torch.gather(
        batch["detection_boxes"][:, time_index],
        1,
        safe_assignment.unsqueeze(-1).expand(-1, -1, 4),
    )
    feature_dim = batch["detection_features"].shape[-1]
    features = torch.gather(
        batch["detection_features"][:, time_index],
        1,
        safe_assignment.unsqueeze(-1).expand(-1, -1, feature_dim),
    )
    scores = torch.gather(batch["detection_scores"][:, time_index], 1, safe_assignment)
    detection_is_valid = torch.gather(batch["detection_valid"][:, time_index], 1, safe_assignment)
    observed = (assignment >= 0) & detection_is_valid & batch["slot_valid"]
    return boxes, features, scores, observed


def _delta_time(batch: dict[str, Tensor], time_index: int) -> Tensor:
    if "delta_time" not in batch:
        return batch["boxes"].new_ones(batch["boxes"].shape[0])
    return batch["delta_time"][:, time_index]


def _counterfactual_loss(
    system: MicroWorldMOT,
    predicted_state,
    batch: dict[str, Tensor],
    time_index: int,
    margin: float,
) -> Tensor:
    observation_boxes, observation_features, observation_scores, observed = _aligned_observations(
        batch, time_index
    )
    negative_observed = observed & observed.roll(1, dims=1)
    if not bool(negative_observed.any()):
        return predicted_state.boxes.new_zeros(())
    positive = system.correct(
        predicted_state,
        observation_boxes,
        observation_features,
        observation_scores,
        observed,
    )
    negative = system.correct(
        predicted_state,
        observation_boxes.roll(1, dims=1),
        observation_features.roll(1, dims=1),
        observation_scores.roll(1, dims=1),
        negative_observed,
    )
    next_delta_time = _delta_time(batch, time_index + 1)
    next_camera = batch["camera_motion"][:, time_index + 1]
    positive_future = system.predict(
        positive, camera_motion=next_camera, delta_time=next_delta_time
    ).state.boxes
    negative_future = system.predict(
        negative, camera_motion=next_camera, delta_time=next_delta_time
    ).state.boxes
    target = batch["boxes"][:, time_index + 1]
    target_mask = batch["existence"][:, time_index + 1] & batch["slot_valid"]
    positive_error = masked_mean((positive_future - target).abs(), target_mask)
    negative_error = masked_mean((negative_future - target).abs(), target_mask)
    return F.relu(positive_error - negative_error + margin)


def sequence_objective(
    system: MicroWorldMOT,
    raw_batch: dict[str, Tensor],
    config: ExperimentConfig,
    device: torch.device,
    epoch: int,
) -> tuple[Tensor, dict[str, Tensor]]:
    batch = _to_device(raw_batch, device)
    boxes = batch["boxes"]
    _, sequence_length, num_tracks, _ = boxes.shape
    initial_boxes, initial_features, _, initial_observed = _aligned_observations(batch, 0)
    # GT-first clip construction deliberately retains identities missed by the
    # detector at t=0. Use the GT state as a teacher-forced motion initializer,
    # but never leak a future appearance embedding into an unobserved slot.
    initial_boxes = torch.where(initial_observed.unsqueeze(-1), initial_boxes, boxes[:, 0])
    initial_features = torch.where(
        initial_observed.unsqueeze(-1), initial_features, torch.zeros_like(initial_features)
    )
    state = system.initialize_state(initial_boxes, initial_features, batch["slot_valid"])
    state.existence = batch["existence"][:, 0].float()
    target_appearance = system.project_appearance(batch["appearance"])
    metrics: dict[str, Tensor] = defaultdict(lambda: boxes.new_zeros(()))
    steps = 0
    cf_steps = 0
    jepa_steps = 0
    jepa_rollout_steps = 0

    for time_index in range(1, sequence_length):
        prediction = system.predict(
            state,
            camera_motion=batch["camera_motion"][:, time_index],
            delta_time=_delta_time(batch, time_index),
        )
        target_exists = batch["existence"][:, time_index]
        target_visible = batch["visibility"][:, time_index]
        target_mask = target_exists & batch["slot_valid"]
        target_occluded = target_exists & ~target_visible

        metrics["nll"] += gaussian_mixture_nll(prediction, boxes[:, time_index], target_mask)
        metrics["state"] += masked_mean(
            (prediction.expected_boxes - boxes[:, time_index]).abs(), target_mask
        )
        metrics["existence"] += _binary_loss(
            prediction.state.existence, target_exists, batch["slot_valid"]
        )
        metrics["occlusion"] += _binary_loss(
            prediction.state.occlusion, target_occluded, batch["slot_valid"]
        )

        observation_boxes, observation_features, observation_scores, observed = (
            _aligned_observations(batch, time_index)
        )
        target_latent = system.project_target_appearance(observation_features)
        if prediction.latent_prediction is not None:
            metrics["object_jepa"] += _object_jepa_loss(
                prediction.latent_prediction,
                target_latent,
                observed,
            )
            jepa_steps += 1
        detection_valid = batch["detection_valid"][:, time_index]
        association_logits = system.association_logits(
            prediction.state,
            batch["detection_boxes"][:, time_index],
            batch["detection_features"][:, time_index],
            batch["detection_scores"][:, time_index],
            detection_valid,
        )
        num_detections = association_logits.shape[-1]
        association_target = boxes.new_zeros(boxes.shape[0], num_tracks, num_detections)
        safe_assignment = batch["assignment"][:, time_index].clamp_min(0)
        association_target.scatter_(2, safe_assignment.unsqueeze(-1), 1.0)
        association_target *= observed.unsqueeze(-1)
        valid_pairs = batch["slot_valid"].unsqueeze(-1) & detection_valid.unsqueeze(1)
        metrics["association"] += sinkhorn_loss(association_logits, association_target, valid_pairs)

        correction_observed = observed
        if system.training and config.train.object_mask_ratio > 0:
            masked_objects = observed & (
                torch.rand_like(observation_scores) < config.train.object_mask_ratio
            )
            correction_observed = observed & ~masked_objects
            metrics["object_mask_rate"] += (
                masked_objects.float().sum() / observed.float().sum().clamp_min(1)
            )

        corrected = system.correct(
            prediction.state,
            observation_boxes,
            observation_features,
            observation_scores,
            correction_observed,
        )
        metrics["identity"] += _identity_contrastive_loss(
            corrected.appearance,
            target_appearance,
            observed,
        )

        squared_error = (prediction.expected_boxes - boxes[:, time_index]).square().mean(dim=-1)
        predicted_variance = prediction.uncertainty / 4.0
        metrics["uncertainty"] += masked_mean(
            (
                predicted_variance.clamp_min(1e-8).log()
                - squared_error.detach().clamp_min(1e-8).log()
            ).abs(),
            target_mask,
        )
        iou = box_iou_aligned(prediction.expected_boxes, boxes[:, time_index])
        refresh_target = ((iou < 0.5) | target_occluded).float()
        refresh_loss = F.binary_cross_entropy_with_logits(
            prediction.refresh_logits, refresh_target, reduction="none"
        )
        metrics["refresh"] += masked_mean(refresh_loss, batch["slot_valid"])

        if (
            time_index + 1 < sequence_length
            and time_index % max(1, config.model.rollout_horizon) == 0
        ):
            metrics["counterfactual"] += _counterfactual_loss(
                system,
                prediction.state,
                batch,
                time_index,
                config.loss.counterfactual_margin,
            )
            cf_steps += 1

        # Open-loop one-step consistency after correction. It is cheap and forces
        # the state update to remain useful for the next world transition.
        if time_index + 1 < sequence_length:
            future_prediction = system.predict(
                corrected,
                camera_motion=batch["camera_motion"][:, time_index + 1],
                delta_time=_delta_time(batch, time_index + 1),
            )
            future = future_prediction.expected_boxes
            future_mask = batch["existence"][:, time_index + 1] & batch["slot_valid"]
            metrics["rollout"] += masked_mean(
                (future - boxes[:, time_index + 1]).abs(), future_mask
            )
            if future_prediction.latent_prediction is not None and time_index + 2 < sequence_length:
                second_future = system.predict(
                    future_prediction.state,
                    camera_motion=batch["camera_motion"][:, time_index + 2],
                    delta_time=_delta_time(batch, time_index + 2),
                )
                _, rollout_features, _, rollout_observed = _aligned_observations(
                    batch, time_index + 2
                )
                rollout_target = system.project_target_appearance(rollout_features)
                metrics["object_jepa_rollout"] += _object_jepa_loss(
                    second_future.latent_prediction,
                    rollout_target,
                    rollout_observed,
                )
                jepa_rollout_steps += 1
        state = corrected
        steps += 1

    for key in list(metrics):
        if key == "counterfactual":
            divisor = cf_steps
        elif key == "object_jepa":
            divisor = jepa_steps
        elif key == "object_jepa_rollout":
            divisor = jepa_rollout_steps
        else:
            divisor = steps
        metrics[key] = metrics[key] / max(1, divisor)

    cf_ramp = min(1.0, (epoch + 1) / max(1, config.train.counterfactual_warmup_epochs))
    weights = config.loss
    total = (
        metrics["nll"]
        + weights.state * metrics["state"]
        + weights.rollout * metrics["rollout"]
        + weights.association * metrics["association"]
        + weights.counterfactual * cf_ramp * metrics["counterfactual"]
        + weights.identity * metrics["identity"]
        + weights.existence * metrics["existence"]
        + weights.occlusion * metrics["occlusion"]
        + weights.uncertainty * metrics["uncertainty"]
        + weights.refresh * metrics["refresh"]
        + weights.object_jepa * metrics["object_jepa"]
        + weights.object_jepa_rollout * metrics["object_jepa_rollout"]
    )
    metrics["loss"] = total
    metrics["mean_iou"] = masked_mean(iou, target_mask)
    return total, dict(metrics)


@torch.no_grad()
def evaluate_epoch(
    system: MicroWorldMOT,
    loader: DataLoader,
    config: ExperimentConfig,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    system.eval()
    totals: dict[str, float] = defaultdict(float)
    count = 0
    for batch in loader:
        _, metrics = sequence_objective(system, batch, config, device, epoch)
        for key, value in metrics.items():
            totals[key] += float(value.detach())
        count += 1
    return {key: value / max(1, count) for key, value in totals.items()}


def train(config: ExperimentConfig) -> Path:
    seed_everything(config.train.seed)
    if not 0.0 <= config.train.object_mask_ratio <= 1.0:
        raise ValueError("train.object_mask_ratio must be in [0, 1]")
    if not 0.0 <= config.train.target_ema_decay <= 1.0:
        raise ValueError("train.target_ema_decay must be in [0, 1]")
    config.model = resolved_config(config.model)
    device = resolve_device(config.train.device)
    train_dataset, val_dataset = build_datasets(config.data, config.model, config.train.seed)
    pin_memory = device.type == "cuda"
    sampler = None
    if config.data.balance_datasets and isinstance(train_dataset, ConcatDataset):
        sample_weights: list[float] = []
        for dataset in train_dataset.datasets:
            sample_weights.extend([1.0 / len(dataset)] * len(dataset))
        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(sample_weights), replacement=True
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=config.data.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=pin_memory,
    )
    system = build_system(config.model).to(device)
    optimizer = torch.optim.AdamW(
        system.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    use_amp = config.train.amp and device.type == "cuda"
    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    else:  # PyTorch 2.0-2.2 compatibility
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    start_epoch = 0
    best_loss = float("inf")
    output_dir = Path(config.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config.json")

    if config.train.resume:
        checkpoint = torch.load(config.train.resume, map_location=device, weights_only=False)
        system.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_loss = float(checkpoint.get("best_loss", best_loss))

    history_path = output_dir / "history.jsonl"
    for epoch in range(start_epoch, config.train.epochs):
        system.train()
        epoch_totals: dict[str, float] = defaultdict(float)
        batches = 0
        started = time.perf_counter()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss, metrics = sequence_objective(system, batch, config, device, epoch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(system.parameters(), config.train.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            system.update_target_encoder(config.train.target_ema_decay)
            for key, value in metrics.items():
                epoch_totals[key] += float(value.detach())
            batches += 1

        train_metrics = {key: value / max(1, batches) for key, value in epoch_totals.items()}
        val_metrics = evaluate_epoch(system, val_loader, config, device, epoch)
        record = {
            "epoch": epoch,
            "seconds": time.perf_counter() - started,
            "train": train_metrics,
            "val": val_metrics,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(
            f"epoch={epoch + 1:03d}/{config.train.epochs} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_iou={val_metrics['mean_iou']:.4f} "
            f"time={record['seconds']:.1f}s"
        )
        checkpoint = {
            "epoch": epoch,
            "model": system.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_loss": min(best_loss, val_metrics["loss"]),
            "config": config.as_dict(),
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            torch.save(checkpoint, output_dir / "best.pt")
    return output_dir / "best.pt"
