from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from ..association.counterfactual import candidate_assignments
from ..association.hungarian import assignment_cost, linear_assignment
from ..config import TrackerConfig
from ..models.system import MicroWorldMOT
from ..types import Detections, TrackOutput, WorldState
from .scheduler import UncertaintyScheduler


def xyxy_to_cxcylogwh(boxes: Tensor, image_size: tuple[int, int] | None = None) -> Tensor:
    boxes = boxes.float()
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    width, height = (x2 - x1).clamp_min(1e-5), (y2 - y1).clamp_min(1e-5)
    if image_size is not None:
        image_height, image_width = image_size
        cx, width = cx / image_width, width / image_width
        cy, height = cy / image_height, height / image_height
    return torch.stack((cx, cy, width.log(), height.log()), dim=-1)


def cxcylogwh_to_xyxy(boxes: Tensor, image_size: tuple[int, int] | None = None) -> Tensor:
    center, size = boxes[..., :2], boxes[..., 2:].exp()
    minimum, maximum = center - size / 2, center + size / 2
    output = torch.cat((minimum, maximum), dim=-1)
    if image_size is not None:
        image_height, image_width = image_size
        scale = output.new_tensor([image_width, image_height, image_width, image_height])
        output = output * scale
    return output


def _select_state(state: WorldState, indices: Sequence[int]) -> WorldState:
    index = torch.as_tensor(indices, dtype=torch.long, device=state.boxes.device)
    return WorldState(
        boxes=state.boxes[:, index],
        velocity=state.velocity[:, index],
        appearance=state.appearance[:, index],
        memory=state.memory[:, index],
        valid=state.valid[:, index],
        existence=state.existence[:, index],
        occlusion=state.occlusion[:, index],
        log_variance=state.log_variance[:, index],
    )


def _concatenate_states(first: WorldState | None, second: WorldState) -> WorldState:
    if first is None or first.num_objects == 0:
        return second
    return WorldState(
        boxes=torch.cat((first.boxes, second.boxes), dim=1),
        velocity=torch.cat((first.velocity, second.velocity), dim=1),
        appearance=torch.cat((first.appearance, second.appearance), dim=1),
        memory=torch.cat((first.memory, second.memory), dim=1),
        valid=torch.cat((first.valid, second.valid), dim=1),
        existence=torch.cat((first.existence, second.existence), dim=1),
        occlusion=torch.cat((first.occlusion, second.occlusion), dim=1),
        log_variance=torch.cat((first.log_variance, second.log_variance), dim=1),
    )


@dataclass
class _TrackMeta:
    track_id: int
    hits: int = 1
    misses: int = 0


class MicroWorldTracker:
    """Online tracker using the learned world model as its motion prior."""

    def __init__(
        self,
        system: MicroWorldMOT,
        config: TrackerConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.system = system.eval().to(device)
        self.config = config or TrackerConfig()
        self.device = torch.device(device)
        self.scheduler = UncertaintyScheduler(
            self.config.refresh_threshold, self.config.max_detector_gap
        )
        self.state: WorldState | None = None
        self.metadata: list[_TrackMeta] = []
        self.next_track_id = 1

    def reset(self) -> None:
        self.state = None
        self.metadata.clear()
        self.next_track_id = 1
        self.scheduler.reset()

    def _aligned_observations(
        self,
        predicted: WorldState,
        detections: Detections,
        matches: Sequence[tuple[int, int]],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        boxes = torch.zeros_like(predicted.boxes)
        appearance = predicted.boxes.new_zeros(
            predicted.batch_size, predicted.num_objects, self.system.config.observation_dim
        )
        scores = torch.zeros_like(predicted.existence)
        mask = torch.zeros_like(predicted.valid)
        for track_index, detection_index in matches:
            boxes[0, track_index] = detections.boxes[detection_index]
            appearance[0, track_index] = detections.appearance[detection_index]
            scores[0, track_index] = detections.scores[detection_index]
            mask[0, track_index] = True
        return boxes, appearance, scores, mask

    def _choose_hypothesis(
        self,
        predicted: WorldState,
        detections: Detections,
        cost: np.ndarray,
    ) -> list[tuple[int, int]]:
        base, _, _ = linear_assignment(cost, self.config.association_threshold)
        if cost.shape[1] < 2:
            return base
        sorted_cost = np.sort(cost, axis=1)
        ambiguous = np.isfinite(sorted_cost[:, 1]) & (
            sorted_cost[:, 1] - sorted_cost[:, 0] < self.config.ambiguity_margin
        )
        if not bool(ambiguous.any()):
            return base
        hypotheses = candidate_assignments(
            cost,
            max_hypotheses=self.system.config.max_hypotheses,
            threshold=self.config.association_threshold,
        )
        if not self.config.use_counterfactual or len(hypotheses) == 1:
            return hypotheses[0]
        best_score = float("inf")
        best = hypotheses[0]
        for hypothesis in hypotheses:
            boxes, appearance, scores, mask = self._aligned_observations(
                predicted, detections, hypothesis
            )
            corrected = self.system.correct(predicted, boxes, appearance, scores, mask)
            rollout = self.system.world_model.rollout(
                corrected, self.system.config.rollout_horizon
            )
            future_energy = float(self.system.future_energy(rollout).mean())
            local_cost = assignment_cost(cost, hypothesis)
            score = local_cost + 0.25 * future_energy
            if score < best_score:
                best_score = score
                best = hypothesis
        return best

    def _add_births(self, detections: Detections, indices: Sequence[int]) -> None:
        accepted = [index for index in indices if float(detections.scores[index]) >= self.config.birth_threshold]
        if not accepted:
            return
        index = torch.as_tensor(accepted, device=self.device, dtype=torch.long)
        new_state = self.system.initialize_state(
            detections.boxes[index].unsqueeze(0),
            detections.appearance[index].unsqueeze(0),
        )
        self.state = _concatenate_states(self.state, new_state)
        for _ in accepted:
            self.metadata.append(_TrackMeta(self.next_track_id))
            self.next_track_id += 1

    def _prune(self) -> None:
        if self.state is None:
            return
        keep = [
            index
            for index, meta in enumerate(self.metadata)
            if meta.misses <= self.config.max_age
            and (
                float(self.state.existence[0, index]) >= self.config.existence_threshold
                or meta.misses <= 1
            )
        ]
        if len(keep) != len(self.metadata):
            self.state = _select_state(self.state, keep) if keep else None
            self.metadata = [self.metadata[index] for index in keep]

    @torch.inference_mode()
    def should_run_detector(self) -> bool:
        if self.state is None:
            return True
        prediction = self.system.predict(self.state)
        return self.scheduler.should_refresh(prediction)

    @torch.inference_mode()
    def step(
        self,
        detections: Detections | None,
        camera_motion: Tensor | None = None,
    ) -> list[TrackOutput]:
        if detections is not None:
            detections = detections.to(self.device)
        if self.state is None:
            if detections is not None and len(detections.boxes) > 0:
                self._add_births(detections, range(len(detections.boxes)))
            self.scheduler.update(detections is not None)
            return self.outputs()

        prediction = self.system.predict(self.state, camera_motion=camera_motion)
        self.state = prediction.state
        for meta in self.metadata:
            meta.misses += 1

        if detections is None or len(detections.boxes) == 0:
            self._prune()
            self.scheduler.update(detections is not None)
            return self.outputs()

        detection_boxes = detections.boxes.unsqueeze(0)
        detection_appearance = detections.appearance.unsqueeze(0)
        detection_scores = detections.scores.unsqueeze(0)
        logits = self.system.association_logits(
            self.state, detection_boxes, detection_appearance, detection_scores
        )
        cost = (-logits[0]).detach().cpu().numpy()
        if self.config.use_counterfactual:
            matches = self._choose_hypothesis(self.state, detections, cost)
            matched_tracks = {row for row, _ in matches}
            matched_detections = {column for _, column in matches}
            unmatched_tracks = [i for i in range(len(self.metadata)) if i not in matched_tracks]
            unmatched_detections = [i for i in range(len(detections.boxes)) if i not in matched_detections]
        else:
            matches, unmatched_tracks, unmatched_detections = linear_assignment(
                cost, self.config.association_threshold
            )

        boxes, appearance, scores, mask = self._aligned_observations(
            self.state, detections, matches
        )
        self.state = self.system.correct(self.state, boxes, appearance, scores, mask)
        for track_index, _ in matches:
            self.metadata[track_index].hits += 1
            self.metadata[track_index].misses = 0
        for track_index in unmatched_tracks:
            self.state.existence[0, track_index] *= 0.95
        self._prune()

        # Pruning changes indices but unmatched detections are independent of tracks.
        self._add_births(detections, unmatched_detections)
        self.scheduler.update(True)
        return self.outputs()

    def outputs(self) -> list[TrackOutput]:
        if self.state is None:
            return []
        outputs: list[TrackOutput] = []
        for index, meta in enumerate(self.metadata):
            if meta.hits < self.config.confirmation_hits:
                continue
            if float(self.state.existence[0, index]) < self.config.existence_threshold:
                continue
            outputs.append(
                TrackOutput(
                    track_id=meta.track_id,
                    box=self.state.boxes[0, index].detach().cpu(),
                    score=float(self.state.existence[0, index]),
                    existence=float(self.state.existence[0, index]),
                    occlusion=float(self.state.occlusion[0, index]),
                )
            )
        return outputs
