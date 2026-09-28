from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from ..association.sinkhorn import log_sinkhorn
from ..config import ExperimentConfig, save_config
from ..data import build_raw_datasets
from ..losses import box_iou_aligned, gaussian_mixture_nll, masked_mean
from ..models import build_system
from ..models.registry import resolved_config
from ..models.system import MicroWorldMOT
from ..perception import JointYOLO11
from ..types import WorldState
from .engine import (
    _binary_loss,
    _identity_contrastive_loss,
    _object_jepa_loss,
    _to_device,
    resolve_device,
    seed_everything,
)


def _detector_targets(batch: dict[str, Tensor], time_index: int) -> dict[str, Tensor]:
    valid = batch["detector_target_valid"][:, time_index]
    batch_indices = torch.arange(valid.shape[0], device=valid.device)
    batch_indices = batch_indices[:, None].expand_as(valid)[valid]
    return {
        "batch_idx": batch_indices,
        "cls": batch["detector_labels"][:, time_index][valid].reshape(-1, 1).float(),
        "bboxes": batch["detection_xywh"][:, time_index][valid],
    }


def _aligned_tensor(values: Tensor, assignment: Tensor) -> Tensor:
    safe = assignment.clamp_min(0)
    if values.ndim == 3:
        return torch.gather(values, 1, safe.unsqueeze(-1).expand(-1, -1, values.shape[-1]))
    return torch.gather(values, 1, safe)


def _soft_association_loss(
    logits: Tensor,
    target: Tensor,
    valid_pairs: Tensor,
    *,
    iterations: int,
    temperature: float,
) -> tuple[Tensor, Tensor]:
    masked_logits = logits.masked_fill(~valid_pairs, -1e4)
    log_probability = log_sinkhorn(
        masked_logits,
        iterations=iterations,
        temperature=temperature,
    )
    masked_target = target * valid_pairs.to(target.dtype)
    loss = -(masked_target * log_probability).sum() / masked_target.sum().clamp_min(1.0)
    probability = log_probability.exp() * valid_pairs.to(log_probability.dtype)
    probability = probability / probability.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return loss, probability


def _initial_state(
    system: MicroWorldMOT,
    perception: JointYOLO11,
    batch: dict[str, Tensor],
) -> WorldState:
    assignment = batch["assignment"][:, 0]
    features = perception.encode_rois(
        batch["detection_xywh"][:, 0], batch["detection_valid"][:, 0]
    )
    aligned_features = _aligned_tensor(features, assignment)
    aligned_boxes = _aligned_tensor(batch["detection_boxes"][:, 0], assignment)
    aligned_valid = _aligned_tensor(batch["detection_valid"][:, 0], assignment)
    observed = (assignment >= 0) & aligned_valid & batch["slot_valid"]
    initial_boxes = torch.where(
        observed.unsqueeze(-1), aligned_boxes, batch["boxes"][:, 0]
    )
    initial_features = torch.where(
        observed.unsqueeze(-1), aligned_features, torch.zeros_like(aligned_features)
    )
    state = system.initialize_state(initial_boxes, initial_features, batch["slot_valid"])
    state.existence = batch["existence"][:, 0].float()
    return state


def joint_sequence_objective(
    system: MicroWorldMOT,
    perception: JointYOLO11,
    raw_batch: dict[str, Tensor],
    config: ExperimentConfig,
    device: torch.device,
    epoch: int,
    *,
    include_detector_loss: bool = True,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Unroll detector, soft association, correction and world dynamics jointly."""
    del epoch  # All modules are active from the first update; there is no stage schedule.
    batch = _to_device(raw_batch, device)
    boxes = batch["boxes"]
    _, sequence_length, num_tracks, _ = boxes.shape
    metrics: dict[str, Tensor] = defaultdict(lambda: boxes.new_zeros(()))

    perception.clear_world_prior()
    if include_detector_loss:
        metrics["detector"] += perception.detector_loss(
            batch["images"][:, 0], _detector_targets(batch, 0)
        )
    else:
        perception.extract_features(batch["images"][:, 0])
    state = _initial_state(system, perception, batch)
    detector_steps = 1
    world_steps = 0
    jepa_steps = 0
    rollout_steps = 0

    for time_index in range(1, sequence_length):
        prediction = system.predict(
            state,
            camera_motion=batch["camera_motion"][:, time_index],
            delta_time=batch["delta_time"][:, time_index],
        )
        perception.set_world_prior(prediction)
        if include_detector_loss:
            metrics["detector"] += perception.detector_loss(
                batch["images"][:, time_index], _detector_targets(batch, time_index)
            )
        else:
            perception.extract_features(batch["images"][:, time_index])
        detector_steps += 1

        detection_valid = batch["detection_valid"][:, time_index]
        detection_features = perception.encode_rois(
            batch["detection_xywh"][:, time_index], detection_valid
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

        logits = system.association_logits(
            prediction.state,
            batch["detection_boxes"][:, time_index],
            detection_features,
            batch["detection_scores"][:, time_index],
            detection_valid,
        )
        assignment = batch["assignment"][:, time_index]
        aligned_valid = _aligned_tensor(detection_valid, assignment)
        observed = (assignment >= 0) & aligned_valid & batch["slot_valid"]
        association_target = boxes.new_zeros(
            boxes.shape[0], num_tracks, logits.shape[-1]
        )
        association_target.scatter_(2, assignment.clamp_min(0).unsqueeze(-1), 1.0)
        association_target *= observed.unsqueeze(-1)
        valid_pairs = batch["slot_valid"].unsqueeze(-1) & detection_valid.unsqueeze(1)
        association_loss, association_probability = _soft_association_loss(
            logits,
            association_target,
            valid_pairs,
            iterations=config.joint.sinkhorn_iterations,
            temperature=config.joint.association_temperature,
        )
        metrics["association"] += association_loss

        soft_boxes = torch.einsum(
            "bnm,bmd->bnd",
            association_probability,
            batch["detection_boxes"][:, time_index],
        )
        soft_features = torch.einsum(
            "bnm,bmd->bnd", association_probability, detection_features
        )
        soft_scores = torch.einsum(
            "bnm,bm->bn",
            association_probability,
            batch["detection_scores"][:, time_index],
        )
        row_mass = (
            association_probability * detection_valid.unsqueeze(1)
        ).sum(dim=-1)
        correction_mask = (row_mass > 1e-4) & batch["slot_valid"]
        corrected = system.correct(
            prediction.state,
            soft_boxes,
            soft_features,
            soft_scores,
            correction_mask,
        )

        aligned_features = _aligned_tensor(detection_features, assignment)
        target_latent = system.project_target_appearance(aligned_features)
        if prediction.latent_prediction is not None:
            metrics["object_jepa"] += _object_jepa_loss(
                prediction.latent_prediction, target_latent, observed
            )
            jepa_steps += 1
        metrics["identity"] += _identity_contrastive_loss(
            corrected.appearance, target_latent, observed
        )

        squared_error = (prediction.expected_boxes - boxes[:, time_index]).square().mean(-1)
        predicted_variance = prediction.uncertainty / 4.0
        metrics["uncertainty"] += masked_mean(
            (
                predicted_variance.clamp_min(1e-8).log()
                - squared_error.detach().clamp_min(1e-8).log()
            ).abs(),
            target_mask,
        )
        iou = box_iou_aligned(prediction.expected_boxes, boxes[:, time_index])
        metrics["mean_iou"] += masked_mean(iou, target_mask)
        refresh_target = ((iou < 0.5) | target_occluded).float()
        refresh_loss = F.binary_cross_entropy_with_logits(
            prediction.refresh_logits, refresh_target, reduction="none"
        )
        metrics["refresh"] += masked_mean(refresh_loss, batch["slot_valid"])

        if time_index + 1 < sequence_length:
            future_prediction = system.predict(
                corrected,
                camera_motion=batch["camera_motion"][:, time_index + 1],
                delta_time=batch["delta_time"][:, time_index + 1],
            )
            future_mask = batch["existence"][:, time_index + 1] & batch["slot_valid"]
            metrics["rollout"] += masked_mean(
                (future_prediction.expected_boxes - boxes[:, time_index + 1]).abs(),
                future_mask,
            )
            rollout_steps += 1

        state = corrected
        if config.train.detach_interval > 0 and time_index % config.train.detach_interval == 0:
            state = state.detach()
        world_steps += 1

    perception.clear_world_prior()
    metrics["detector"] /= max(1, detector_steps)
    for key in (
        "nll",
        "state",
        "association",
        "identity",
        "existence",
        "occlusion",
        "uncertainty",
        "refresh",
        "mean_iou",
    ):
        metrics[key] /= max(1, world_steps)
    metrics["object_jepa"] /= max(1, jepa_steps)
    metrics["rollout"] /= max(1, rollout_steps)

    weights = config.loss
    world_loss = (
        metrics["nll"]
        + weights.state * metrics["state"]
        + weights.rollout * metrics["rollout"]
        + weights.association * metrics["association"]
        + weights.identity * metrics["identity"]
        + weights.existence * metrics["existence"]
        + weights.occlusion * metrics["occlusion"]
        + weights.uncertainty * metrics["uncertainty"]
        + weights.refresh * metrics["refresh"]
        + weights.object_jepa * metrics["object_jepa"]
    )
    total = config.joint.world_loss_weight * world_loss
    if include_detector_loss:
        total = total + config.joint.detector_loss_weight * metrics["detector"]
    metrics["world_loss"] = world_loss
    metrics["loss"] = total
    return total, dict(metrics)


@torch.no_grad()
def evaluate_joint_epoch(
    system: MicroWorldMOT,
    perception: JointYOLO11,
    loader: DataLoader,
    config: ExperimentConfig,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    system.eval()
    perception.eval()
    totals: dict[str, float] = defaultdict(float)
    count = 0
    for batch in loader:
        _, metrics = joint_sequence_objective(
            system,
            perception,
            batch,
            config,
            device,
            epoch,
            include_detector_loss=False,
        )
        for key, value in metrics.items():
            totals[key] += float(value.detach())
        count += 1
    return {key: value / max(1, count) for key, value in totals.items()}


def _balanced_sampler(dataset) -> WeightedRandomSampler | None:
    if not isinstance(dataset, ConcatDataset):
        return None
    weights: list[float] = []
    for child in dataset.datasets:
        weights.extend([1.0 / len(child)] * len(child))
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def train_joint(config: ExperimentConfig) -> Path:
    if not config.joint.enabled or config.data.kind != "joint_raw":
        raise ValueError("Joint training requires joint.enabled=true and data.kind='joint_raw'")
    if config.train.gradient_accumulation < 1:
        raise ValueError("train.gradient_accumulation must be >= 1")
    seed_everything(config.train.seed)
    config.model = resolved_config(config.model)
    device = resolve_device(config.train.device)
    train_dataset, val_dataset = build_raw_datasets(config.data, config.train.seed)
    sampler = _balanced_sampler(train_dataset) if config.data.balance_datasets else None
    loader_options = {
        "batch_size": config.train.batch_size,
        "num_workers": config.data.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=sampler is None,
        sampler=sampler,
        **loader_options,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)

    system = build_system(config.model).to(device)
    perception = JointYOLO11(
        config.joint.detector_weights,
        config.model.observation_dim,
        config.joint.roi_size,
        config.joint.world_prior_strength,
    ).to(device)
    backbone, head, adapters = perception.parameter_groups()
    world_parameters = [parameter for parameter in system.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone, "lr": config.joint.detector_backbone_lr, "name": "yolo_backbone"},
            {"params": head, "lr": config.joint.detector_head_lr, "name": "yolo_head"},
            {"params": adapters, "lr": config.joint.perception_lr, "name": "roi_and_prior"},
            {"params": world_parameters, "lr": config.joint.world_lr, "name": "world"},
        ],
        weight_decay=config.train.weight_decay,
    )
    use_amp = config.train.amp and device.type == "cuda"
    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    output_dir = Path(config.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config.json")
    start_epoch = 0
    best_loss = float("inf")
    if config.train.resume:
        checkpoint = torch.load(config.train.resume, map_location=device, weights_only=False)
        system.load_state_dict(checkpoint["model"])
        perception.load_state_dict(checkpoint["joint_perception"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_loss = float(checkpoint.get("best_loss", best_loss))

    history_path = output_dir / "history.jsonl"
    accumulation = config.train.gradient_accumulation
    for epoch in range(start_epoch, config.train.epochs):
        system.train()
        perception.train()
        optimizer.zero_grad(set_to_none=True)
        totals: dict[str, float] = defaultdict(float)
        batches = 0
        started = time.perf_counter()
        for batch_index, batch in enumerate(train_loader):
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss, metrics = joint_sequence_objective(
                    system, perception, batch, config, device, epoch
                )
                scaled_loss = loss / accumulation
            scaler.scale(scaled_loss).backward()
            should_step = (batch_index + 1) % accumulation == 0 or (
                batch_index + 1 == len(train_loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [*system.parameters(), *perception.parameters()],
                    config.train.gradient_clip,
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                system.update_target_encoder(config.train.target_ema_decay)
            for key, value in metrics.items():
                totals[key] += float(value.detach())
            batches += 1

        train_metrics = {key: value / max(1, batches) for key, value in totals.items()}
        val_metrics = evaluate_joint_epoch(
            system, perception, val_loader, config, device, epoch
        )
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
            f"det_loss={train_metrics['detector']:.4f} "
            f"val_world={val_metrics['world_loss']:.4f} "
            f"val_iou={val_metrics['mean_iou']:.4f} "
            f"time={record['seconds']:.1f}s"
        )
        checkpoint = {
            "epoch": epoch,
            "model": system.state_dict(),
            "joint_perception": perception.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_loss": min(best_loss, val_metrics["world_loss"]),
            "config": config.as_dict(),
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if val_metrics["world_loss"] < best_loss:
            best_loss = val_metrics["world_loss"]
            torch.save(checkpoint, output_dir / "best.pt")
    return output_dir / "best.pt"
