# Architecture specification

## State

For every object slot, the system keeps

```text
box           [cx, cy, log(width), log(height)]
velocity      derivative of the box state
appearance    low-dimensional detector ROI embedding
memory        recurrent latent state
existence     probability that the object still exists
occlusion     probability that it is currently hidden
log_variance  diagonal box uncertainty
```

All learned modules receive padded batches `[B,N,*]` and a boolean valid mask.

Detector observations use independent padded slots `[B,M,*]`. Raw frozen
YOLO11n penultimate-layer crop embeddings have dimension 256 and pass through a
trainable normalized projection to the 32-dimensional appearance state. `M`
does not need to equal `N`; false positives remain in the detection set and
missed objects receive an assignment target of `-1`.

## World transition

The sparse graph first produces interaction memory `m_i`. The transition cell
then predicts acceleration `a_i`, damping `rho_i` and new memory:

```text
v(t+1) = rho(t) * v(t) + dt * a(t)
b_phys(t+1) = b(t) + dt * v(t+1) + 0.5 * dt^2 * a(t)
```

A mixture head predicts small residuals around `b_phys`:

```text
p(b(t+1)) = sum_k pi_k Normal(b_phys + delta_k, diag(sigma_k^2))
```

This separates an interpretable kinematic path from learned corrections.

## Object-JEPA latent prediction

The online observation projection maps frozen YOLO11n crop features to object
latents:

```text
q(t) = normalize(E_online(yolo_crop(t)))
```

An identically shaped target projection is updated only by exponential moving
average and receives no gradients:

```text
theta_target <- decay * theta_target + (1 - decay) * theta_online
q_target(t) = stop_gradient(normalize(E_target(yolo_crop(t))))
```

The object world transition predicts a future latent from its next recurrent
memory. Unlike dense V-JEPA video patches, prediction occurs only for sparse
tracked objects:

```text
q_hat_i(t+1) = normalize(P_latent(memory_i(t+1)))
L_object_jepa = mean_i |q_hat_i(t+1) - q_target_i(t+1)|
```

During training, a configurable fraction of matched observations is hidden
from the correction cell. This turns detector misses and simulated occlusions
into a masked-prediction task. A second loss compares a two-step open-loop
latent rollout with the future EMA target to reduce recurrent error growth.
The YOLO11 encoder remains frozen and no pixel decoder is used.

## Association and correction

The association head combines box innovation, Mahalanobis error, appearance
cosine similarity, confidence, existence and occlusion. Training uses Sinkhorn;
inference uses Hungarian.

The correction cell learns separate box and velocity gains. Missing observations
leave the predicted state unchanged.

## Counterfactual decision

The best Hungarian solution is hypothesis zero. Other hypotheses are formed by
forbidding one selected edge at a time, capped by `max_hypotheses`. Each
hypothesis is corrected and rolled forward by `rollout_horizon`; the tracker
chooses the lowest current cost plus future energy.

The implementation is bounded by `B * H` world-model calls and is only intended
for ambiguous associations. A production version should first partition the
cost graph into ambiguity components.

## Single-run objective

The training engine jointly optimizes:

```text
mixture NLL
box state error
open-loop rollout consistency
EMA-target Object-JEPA latent prediction
two-step Object-JEPA latent rollout
Sinkhorn association
counterfactual ranking
appearance identity consistency
existence and occlusion BCE
uncertainty calibration
detector-refresh BCE
```

Counterfactual weight is smoothly warmed up within the same optimizer run. No
checkpoint is reloaded and no separate pretraining stage is required.
