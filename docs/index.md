---
hide:
  - toc
---

# TargetGym

<p align="center">
  <b>Reach a setpoint. Hold it forever. Against disturbances.</b><br/>
  Twenty-two JAX environments for <i>target MDPs</i>, the control problems industry actually has.
</p>

<p align="center">
  <img src="videos/mosaic_flagship.webp" width="100%"/><br/>
  <sub>One from each family, held on setpoint by its shipped PID baseline.</sub>
</p>

**21 environments**: 9 aircraft, 5 process control, 5 industrial / energy, 2 renewable energy. Every one of them is in the
[gallery](environments.md), with its own page, clip and baseline numbers.

---

```bash
pip install target-gym
```

Or try it in the browser: [Colab quickstart](https://colab.research.google.com/github/YannBerthelot/TargetGym/blob/main/notebooks/quickstart.ipynb).

```python
import jax
import numpy as np
from target_gym import Plane, PlaneParams
from target_gym.registry import REGISTRY

env, params = Plane(), PlaneParams()
obs, state = env.reset(jax.random.PRNGKey(0), params)

# Every environment ships a tuned expert, so a learned policy
# has something real to beat.
pid = REGISTRY["plane"].make_pid()
pid.reset()

for t in range(200):
    action = np.atleast_1d(pid(obs))
    obs, state, reward, terminated, truncated, info = env.step(
        jax.random.PRNGKey(t), state, action, params
    )
    if terminated or truncated:
        break
```

[Browse the twenty-one environments →](environments.md){ .md-button .md-button--primary }
[Getting started →](getting-started.md){ .md-button }

---

## Why these environments

Holding a setpoint forever breaks differently than reaching a goal once, and
these are the failure modes that come with it:

| | |
|---|---|
| **Irrecoverable states** | A boiler drum that carries water into the turbine, a reactor past runaway, a kiln that has gone cold |
| **Partial observability** | The furnace hides 6 of 9 states, the reactor 7 of 11, the kiln 64 behind 8 measurements |
| **Non-minimum phase** | Drum level rises as mass *leaves*; the four-tank's obvious loop pairing is unstable |
| **Transport delay** | Half the kiln's response to a fuel change takes a full 25-minute residence time |
| **Multi-timescale** | Millisecond neutronics against hour-long xenon; sub-second flame gas against 30 h glass residence |
| **Finite budgets** | A battery whose tracking *now* costs the ability to track later |

Every environment ships a tuned PID, and nineteen of twenty-one also ship an MPC,
so a learned policy has something real to beat. And **where a baseline is weak,
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

    [Contributing](https://github.com/YannBerthelot/TargetGym/blob/main/CONTRIBUTING.md) ·
    [Roadmap and known gaps](roadmap.md) ·
    [Functional structure](functional-structure.md)

</div>
