---
hide:
  - toc
---

# TargetGym

<p align="center">
  <b>Reach a setpoint. Hold it forever. Against disturbances.</b><br/>
  Eighteen JAX environments for <i>target MDPs</i> — the control problems industry actually has.
</p>

<p align="center">
  <img src="videos/mosaic_plants.webp" width="100%"/><br/>
  <sub>Twelve process, industrial and energy plants, each held on setpoint by its shipped PID baseline.</sub>
</p>

<p align="center">
  <img src="videos/mosaic_aircraft.webp" width="100%"/><br/>
  <sub>Eight aircraft tasks: altitude hold, step schedules, sinusoid tracking, and three 3D paths.</sub>
</p>

---

```bash
pip install target-gym
```

```python
import jax
import numpy as np
from target_gym import Plane, PlaneParams
from target_gym.registry import REGISTRY

env, params = Plane(), PlaneParams()
obs, state = env.reset_env(jax.random.PRNGKey(0), params)

# Every environment ships a tuned expert, so a learned policy
# has something real to beat.
pid = REGISTRY["plane"].make_pid()
pid.reset()

for t in range(200):
    action = np.atleast_1d(pid(obs))
    obs, state, reward, terminated, truncated = env.step_env(
        jax.random.PRNGKey(t), state, action, params
    )
```

[Browse the eighteen environments →](environments.md){ .md-button .md-button--primary }
[Getting started →](getting-started.md){ .md-button }

---

## Why these environments

Reaching a goal and stopping is not what industrial control is. Holding a
setpoint against disturbances, forever, is — and that changes which failure
modes matter:

| | |
|---|---|
| **Irrecoverable states** | A boiler drum that carries water into the turbine, a reactor past runaway, a kiln that has gone cold |
| **Partial observability** | The furnace hides 6 of 9 states, the reactor 7 of 11, the kiln 64 behind 8 measurements |
| **Non-minimum phase** | Drum level rises as mass *leaves*; the four-tank's obvious loop pairing is unstable |
| **Transport delay** | Half the kiln's response to a fuel change takes a full 25-minute residence time |
| **Multi-timescale** | Millisecond neutronics against hour-long xenon; sub-second flame gas against 30 h glass residence |
| **Finite budgets** | A battery whose tracking *now* costs the ability to track later |

Every environment ships a tuned PID, and sixteen of eighteen also ship an MPC,
so a learned policy has something real to beat — and **where a baseline is weak,
the docs say how weak**.

## Documentation

<div class="grid cards" markdown>

- **Use it**

    [Getting started](getting-started.md) ·
    [Environments](environments.md) ·
    [API reference](api.md)

- **Beat the baselines**

    [Baselines](baselines.md) ·
    [RL protocol](rl-protocol.md) ·
    [RL results](rl-baselines.md)

- **Trust the numbers**

    [Physics methodology](PHYSICS_METHODOLOGY.md) ·
    [Model review checklist](model-review-checklist.md) ·
    [Reward shaping](reward-shaping.md) ·
    [Testing](testing.md)

- **Contribute**

    [Contributing](../CONTRIBUTING.md) ·
    [Roadmap and known gaps](roadmap.md) ·
    [Functional structure](functional-structure.md)

</div>
