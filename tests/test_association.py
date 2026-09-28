import numpy as np
import torch

from microworld_mot.association.counterfactual import candidate_assignments
from microworld_mot.association.sinkhorn import log_sinkhorn


def test_sinkhorn_is_finite() -> None:
    logits = torch.randn(2, 4, 4)
    log_prob = log_sinkhorn(logits, iterations=50, temperature=1.0)
    assert torch.isfinite(log_prob).all()
    probabilities = log_prob.exp()
    assert torch.allclose(probabilities.sum(-1), torch.ones(2, 4), atol=2e-3)
    assert torch.allclose(probabilities.sum(-2), torch.ones(2, 4), atol=2e-3)


def test_counterfactual_candidates_include_best() -> None:
    cost = np.asarray([[0.1, 1.0], [1.0, 0.1]], dtype=np.float32)
    candidates = candidate_assignments(cost, max_hypotheses=3, threshold=2.0)
    assert candidates[0] == [(0, 0), (1, 1)]
    assert len(candidates) >= 2
