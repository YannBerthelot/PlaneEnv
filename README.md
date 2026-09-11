<h1 align="center">TargetGym</h1>

<p align="center">
  <b>Reach the target. Then hold it.</b><br/>
  21 JAX environments for setpoint tracking, each with a tuned PID and an MPC ceiling.
</p>

<p align="center">
  <a href="https://pypi.org/project/target-gym/"><img alt="PyPI" src="https://img.shields.io/pypi/v/target-gym?color=blue"></a>
  <a href="https://pypi.org/project/target-gym/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/target-gym"></a>
  <a href="https://github.com/YannBerthelot/TargetGym/actions"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/YannBerthelot/TargetGym/python-app.yml?branch=main&label=tests"></a>
  <a href="#license"><img alt="License" src="https://img.shields.io/pypi/l/target-gym"></a>
  <img alt="JAX" src="https://img.shields.io/badge/JAX-jit%20%7C%20vmap%20%7C%20scan-orange">
  <a href="https://colab.research.google.com/github/YannBerthelot/TargetGym/blob/main/notebooks/quickstart.ipynb"><img alt="Open in Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>
</p>

<p align="center">
  <img src="videos/mosaic_flagship.webp" width="100%"/><br/>
  <sub>One example task from each family, under PID control.</sub>
</p>

Most RL benchmarks ask you to reach a goal once and stop. Industrial control
does not work that way: you reach a setpoint and then **hold it**, indefinitely,
against disturbances. That is a [target MDP](docs/target-mdp.md), and it brings
its own failure modes: irrecoverable states, transport delay, finite budgets,
and a target that keeps moving.

**TargetGym** is 21 such environments, drawn from real plants: an A320-like
aircraft, a glass furnace, a nuclear reactor, a cement kiln, a grid battery, a
wind turbine, a formation of aircraft flying a patrol.

|  |  |
| --- | --- |
| **You can tell straight away whether your agent is any good.** | Every environment ships a tuned PID, and 20 of 21 an MPC that reads the true state and acts as an upper bound. Both are recorded over ten seeds, so a new score lands somewhere you can read it against. |
| **The physics is tested, and its limits are stated.** | Each environment carries a `PHYSICS.md` with a sourced parameter table, published validation targets that tests assert, and a list of the approximations it makes. |
| **Fast enough to stay out of the way.** | 0.6 M to 700 M steps/s on CPU, `jit`/`vmap`/`scan` throughout, end-to-end GPU. |

Learned baselines are not published yet; what they will be measured against,
and how, is in [docs/rl-protocol.md](docs/rl-protocol.md).

---

## Installation

```bash
pip install target-gym
# or
poetry add target-gym
uv add target-gym
```

## Quickstart

Everything below also runs in [Colab](https://colab.research.google.com/github/YannBerthelot/TargetGym/blob/main/notebooks/quickstart.ipynb).

```python
import jax
import numpy as np
from target_gym.provenance import load_recorded_baselines
from target_gym.registry import REGISTRY
from target_gym.runners.runners import baseline_policy

spec = REGISTRY["plane"]
env, params = spec.make_env(), spec.make_test_params()


def evaluate(policy, seed):
    # One episode. Any policy taking (obs, state) works, including yours.
    key = jax.random.PRNGKey(seed)
    obs, state = env.reset_env(key, params)
    total = 0.0
    for _ in range(int(params.max_steps_in_episode)):
        action = policy(np.asarray(obs), state)
        obs, state, reward, terminated, _ = env.step_env(key, state, action, params)
        total += float(reward)
        if bool(terminated):
            break
    return total


print("PID:", evaluate(baseline_policy(spec, "pid", params), seed=0))

# The MPC ceiling is already recorded on these same seeds, so there is nothing
# to run: it costs minutes per seed and would come out the same. Evaluate your
# agent on seeds 0-9 and read off where you land between the two.
print("MPC:", load_recorded_baselines()["plane"]["mpc_returns"][0])
```

Both baselines take `(obs, state)`, so one loop serves either and your agent
drops in beside them. The asymmetry is deliberate. **The PID ignores `state`**
and reads instruments, the way a plant controller does. **The MPC ignores
`obs`** and reads the true state, because it is there to be an upper bound. Its
lead therefore includes the extra information it gets, and the table in
[docs/baselines.md](docs/baselines.md) says so.

<details>
<summary>Or use another RL library, JAX or not (e.g. stable-baselines3)</summary>

Note that only a JAX-based library gives you end-to-end GPU.

```python
# doc: skip (trains for 10 000 steps; tests/plane/test_agent.py covers this path)
from target_gym import GymnasiumPlane
from stable_baselines3 import SAC

env = GymnasiumPlane()
model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000, log_interval=4)

obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

</details>


**[docs/getting-started.md](docs/getting-started.md)** covers the rest: driving
`step_env` directly, vectorising rollouts with `vmap` and `scan`, working from
the registry, the multi-agent patrol interface, and the wind and turbulence
model.

---

## Environments

Four families, all behind one interface.

**[Browse all 21 environments →](docs/environments.md)**

| Family | Count | What they are |
|---|---|---|
| **Aircraft** | 9 | An A320-like 2D aircraft on three reference patterns (hold, sine, altitude-and-airspeed); four 3D path-following tasks; and two formation-patrol variants |
| **Process control** | 5 | CSTR, first-order lag, four-tank, pH neutralisation, binary distillation |
| **Industrial / energy** | 5 | Glass furnace, nuclear reactor, building HVAC, boiler drum, cement kiln |
| **Renewable energy** | 2 | Wind turbine, grid battery |

The CSTR, first-order and four-tank models come from
[PC-gym](https://github.com/MaximilianB2/pc-gym), checked against its source
term by term. Each says so in its `PHYSICS.md` provenance line.

The two **patrol** environments are the multi-agent case: wingmen holding a
slot on a lead that flies its own route, so the target is another aircraft
instead of a fixed setpoint, and you can collide with it.

### Rendering

Every environment draws a control-room dashboard: a plant schematic, gauges with
limit and setpoint markers, and strip charts. Quantities the controller cannot
measure are marked, so you can see what the agent is flying blind on.

**[Rendering guide →](docs/rendering.md)**

### Complexity

Six difficulty tiers, weighing the dynamics (linearity, coupling, stiffness)
alongside the RL side (dimensionality, horizon, partial observability), so you
can use the suite as a curriculum: a first-order lag at tier 1, the cement
kiln's 25-minute transport delay and the multi-agent patrol at tier 6.

**[The full ladder, with observation and action widths →](docs/complexity.md)**

---

## Baselines

All 21 environments ship a tuned PID, and 20 of them an MPC, so you have
something real to beat from the first run. Structure matters more than gains
here, and the baselines are chosen to show it: three-element control on the
boiler drum so the level gauge cannot lie to the controller, a cascade on the
kiln because integral action on a half-hour-old measurement oscillates at the
delay period, and **crossed** loops on the four-tank, whose negative RGA
element makes the obvious pairing unstable.

Every baseline has to beat the best constant action on its environment. That is
a low bar on purpose: it is the one a mis-wired controller trips over. Where a
baseline is weak, its page says how weak.

> **If a PID loses here, that says something about the task, and nothing about
> PID control.** These are problems where anticipation pays. When you can simply
> react to the reference and the disturbances, a PID is optimal or close enough
> that the gap will not show up in a return. This project has learned that twice
> from its own numbers: a battery whose dispatch signal was so noisy that no
> controller could score above 0.43 of the ceiling, and a patrol follower
> chasing a lead at one fixed turn rate, which a single feedforward term
> cancels. Both times the environment was at fault, and both were fixed.

**[docs/baselines.md](docs/baselines.md)** has the details: the three MPC
implementations and why each plant gets the one it does, tuning and caching,
solver convergence reporting, and per-environment coverage.

---

## Why these environments

Holding a setpoint breaks in ways that reaching a goal once does not:

| | |
|---|---|
| **You can break it for good** | A drum that carries water into the turbine, a reactor past runaway, a kiln that has gone cold |
| **You cannot see most of it** | The furnace hides 6 of its 9 states, the reactor 7 of 11, the kiln 64 behind 8 measurements |
| **It moves the wrong way first** | Open the steam valve and the drum level *rises* as water leaves. Pair the four-tank's obvious loops and it goes unstable |
| **It answers late** | Half the kiln's response to a fuel change takes a full 25-minute residence time |
| **Fast and slow at once** | Millisecond neutronics against hour-long xenon. Sub-second flame gas against 30-hour glass residence |
| **Tracking now costs you later** | A battery spends charge to follow dispatch, and then cannot follow it |

The suite also models actuator lag, competing objectives, and **anticipation**:
scheduled setpoints that a purely reactive controller cannot reach, like the
building's night setback, the furnace's crown schedule and the battery's
dispatch blocks. Every aircraft flies in steady wind, altitude shear and
Ornstein-Uhlenbeck turbulence, none of it observable by default.

---

## Physics validation

Every environment carries a `PHYSICS.md` next to its code: what it models and
where that holds, a parameter table where each constant is cited or derived in
view (and flagged `TUNED - not sourced` when it is neither), the published
numbers it has to reproduce, and the deviations it knowingly has.

The tests check consequences: ISA table values, L/D ratios, thermal time
constants, energy balances, equilibria. A test that recomputes the code's own
formula would pass just as happily on a wrong one.

All 21 environments are covered by fifteen contracts, because the aircraft
variants share the plant they are built on. Every environment also runs the same conformance
suite for determinism, PRNG handling, `jit`/`vmap`/`scan` and numerical health
over a full episode, so the one you pick behaves like the rest.

**[Physics methodology](docs/PHYSICS_METHODOLOGY.md)** has the method, and a
per-environment summary of what each model is validated against together with
an example of something that validation caught.

---

## Documentation

| | |
|---|---|
| **[Getting started](docs/getting-started.md)** | Run an episode, vectorise it, plug into Gymnasium |
| **[Target MDPs](docs/target-mdp.md)** | The formal setting the suite is built around |
| **[Environment reference](docs/environments.md)** | All 21: shapes, tracked variables, baselines, contracts |
| **[Public API](docs/api.md)** | What is stable and what is provisional |
| **[Baselines](docs/baselines.md)** | The shipped PID and MPC controllers, and tuning them |
| **[RL protocol](docs/rl-protocol.md)** | How to measure a learned policy so the number means something |
| **[Reward shaping](docs/reward-shaping.md)** | Why the tracking rewards have the shape they do |
| **[Model review checklist](docs/model-review-checklist.md)** | Thirteen checks worth running on any plant model |
| **[Physics methodology](docs/PHYSICS_METHODOLOGY.md)** | How the physics is sourced, validated and bounded |
| **[Rendering](docs/rendering.md)** | The dashboards, the two toolkits, regenerating the media |
| **[Complexity ladder](docs/complexity.md)** | Six tiers, for using the suite as a curriculum |
| **[Throughput](docs/performance.md)** | Steps per second per environment, and how it is measured |
| **[Testing](docs/testing.md)** | How the suite is organised, if you are contributing |

The full index is at **[docs/](docs/index.md)**.

---

## Related projects

* **[gymnax](https://github.com/RobertTLange/gymnax)**: the JAX environment API
  TargetGym implements. Every environment here is a `gymnax` environment.
* **[Gymnasium](https://github.com/Farama-Foundation/Gymnasium)**: the standard
  single-agent API; TargetGym ships a wrapper so any environment can be driven
  by stable-baselines3 and friends.
* **[PC-gym](https://github.com/MaximilianB2/pc-gym)**: process-control
  environments for RL. The CSTR, first-order and four-tank models here are
  adapted from it.
* **[safe-control-gym](https://github.com/utiasDSL/safe-control-gym)**: the
  closest thing in intent: classical control, MPC and RL on the same tasks. It
  covers more controllers, this covers more plants. Worth a look either way.
* **[JaxMARL](https://github.com/FLAIROX/JaxMARL)**: the multi-agent JAX API
  the patrol formation environment follows.

---

## Roadmap

Approaching 1.0. The rewards, the physics, the baselines and the measurement
protocol are settled; what is left is publishing learned-policy results and
putting the docs online. Known gaps are written down rather than hidden.

**[Full roadmap and known gaps →](docs/roadmap.md)**

## Contributing

Contributions are welcome: bug reports, new environments, better baselines, or
corrections to the physics.

```bash
git clone https://github.com/YannBerthelot/TargetGym.git
cd TargetGym

uv sync --group dev   # creates .venv with runtime, test and lint deps
make ci               # what CI runs: ruff, black --check, docs, fast tests
```

Other tasks live in the `Makefile`: `make test`, `make test-all`, `make figures`,
`make videos`, `make tuning`.

**[CONTRIBUTING.md](CONTRIBUTING.md)** has the rest: running the test suite,
the style rules, and what adding an environment takes. Short version: you
register an `EnvSpec` and it inherits every shared check for free, and you write
a `PHYSICS.md` saying where your numbers come from.

---

## Citation

If you use **TargetGym** in your research or project, please cite it as:

```bibtex
@misc{targetgym2025,
  title        = {TargetGym: Reinforcement Learning Environments for Target MDPs},
  author       = {Yann Berthelot},
  year         = {2025},
  url          = {https://github.com/YannBerthelot/TargetGym},
  note         = {Lightweight physics-based RL environments for aircraft, process control, and industrial systems}
}
```

---

## License

MIT License. Free to use in research and projects.
