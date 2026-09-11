<h1 align="center">TargetGym</h1>

<p align="center">
  <b>Reach and maintain.</b><br/>
  21 environments and controllers for <i>target MDPs</i>: Setpoint tracking benchmark.
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
  <sub>One from each family, held on setpoint by its shipped PID baseline.</sub>
</p>

**21 environments**: 9 aircraft, 5 process control, 5 industrial / energy,
2 renewable energy. Every one ships a tuned PID, and twenty an MPC as well,
both recorded over ten episode seeds. Learned baselines are not published yet;
what they will be measured against, and how, is in
[docs/rl-protocol.md](docs/rl-protocol.md).

**TargetGym** is a collection of JAX **reinforcement learning environments**
built around [**target MDPs**](docs/target-mdp.md): tasks where you **reach a
target and then hold it**, instead of reaching a goal once and stopping. Most
industrial control works that way. You keep a setpoint against disturbances,
indefinitely.

They are fast (0.6 to 700 M steps/s on CPU, see
[docs/performance.md](docs/performance.md)) and end-to-end GPU compatible, with
`jit`/`vmap`/`scan` throughout. Their physics is a **documented, tested
contract**: every environment carries a `PHYSICS.md` with a sourced parameter
table, published validation targets asserted by tests, and quantified known
deviations.

---

## Installation

```bash
pip install target-gym
# or
poetry add target-gym
uv add target-gym
```

## Quickstart

Prefer a browser? The [Colab quickstart](https://colab.research.google.com/github/YannBerthelot/TargetGym/blob/main/notebooks/quickstart.ipynb) runs everything below on a free CPU runtime in about a minute.

```python
import jax
import numpy as np
from target_gym.provenance import load_recorded_baselines
from target_gym.registry import REGISTRY
from target_gym.runners.runners import baseline_policy

spec = REGISTRY["plane"]
env, params = spec.make_env(), spec.make_test_params()


def evaluate(policy, seed):
    # One episode. Any policy taking (obs, state) works -- including yours.
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

Both baselines take `(obs, state)` so one loop serves either, and your agent
drops in beside them. The asymmetry is deliberate: **the PID ignores `state`**,
reading instruments as a plant controller does, while **the MPC ignores `obs`**,
reading the true state because it is a full-state *ceiling* rather than a peer.
A benchmark whose upper bound sees more than its contestants should say so.

<details>
<summary>Or use your favourite RL library, JAX or not (stable-baselines3)</summary>

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
| **Aircraft** | 10 | An A320-like 2D aircraft holding altitude, on four target patterns; four 3D path-following tasks; and two multi-agent patrol variants |
| **Process control** | 5 | CSTR, first-order lag, four-tank, pH neutralisation, binary distillation |
| **Industrial / energy** | 5 | Glass furnace, nuclear reactor, building HVAC, boiler drum, cement kiln |
| **Renewable energy** | 2 | NREL 5 MW wind turbine, grid battery |

The CSTR, first-order and four-tank models are adapted from
[PC-gym](https://github.com/MaximilianB2/pc-gym) and were verified against its
source term for term; each says so in its `PHYSICS.md` provenance line.

The two **patrol** environments are the multi-agent case: wingmen holding a slot
on a *manoeuvring lead*, so the target is another agent rather than a fixed
setpoint, and now you can crash into it.

### Rendering

Every environment draws a control-room dashboard rather than a plot: a live
plant schematic, an instrument stack with limit and setpoint markers, and strip
charts. The schematics are drawn from state the controller usually *cannot*
see, like riser voidage or the kiln's axial profile, and those quantities are
marked, so a frame shows both what the agent measures and what it is actually
up against.

**[Rendering guide →](docs/rendering.md)**

### Complexity

The suite spans six difficulty tiers, weighing both the dynamics (linearity,
coupling, stiffness) and the RL side (dimensionality, horizon, partial
observability), so it works as a curriculum and not only as a benchmark:
a first-order lag at tier 1, the cement kiln's 25-minute transport delay and
the multi-agent patrol at tier 6.

**[The full ladder, with observation and action widths →](docs/complexity.md)**

---

## Baselines

All 21 environments ship a tuned PID, and 20 of them an MPC as well, so you
have something real to beat from the first run.

**A PID losing here is a claim about these tasks, not about PID control.** The
suite collects problems where anticipation pays. Where the reference and the
disturbances are things you can react to rather than foresee, a PID is optimal
or close enough that the gap is unmeasurable, and twice this project has had to
learn that from its own measurements: the battery's dispatch signal was so noisy
that *no* controller could score above 0.43 of the ceiling, and the patrol
follower chased a lead at one constant turn rate, which a single feedforward
term cancels outright. Both times the environment was at fault and was fixed.
A PID is also cheap, transparent, certifiable and runs on a microcontroller,
and none of that shows up in a return.

The right *structure* usually matters more than the gains, and the baselines are
chosen to show it: three-element control on the boiler drum, where feedwater
tracks measured steam flow so the level gauge cannot lie to it; a cascade on the
kiln, because integral action on a half-hour-old measurement oscillates at the
delay period; and **crossed** loops on the four-tank, whose negative RGA element
makes the obvious diagonal pairing unstable.

Every baseline has to beat the best constant action on its environment, which
is a low bar on purpose, because it is the one a mis-wired controller trips
over. And where a baseline is weak, its page says how weak, so you know what
you are actually measuring against.

**[docs/baselines.md](docs/baselines.md)** has the details: the three MPC
implementations and why each plant gets the one it does, the cascaded autopilot,
tuning and caching, and per-environment baseline coverage.

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

On top of those, the suite deliberately models actuator lag, competing
objectives, **anticipation** (scheduled setpoints that a purely reactive
controller cannot reach: the building's night setback, the furnace's crown
schedule, the battery's dispatch blocks), moving targets such as the patrol
slot behind a manoeuvring lead, multi-agent coordination in the MARL patrol
task, and a full wind model on every aircraft -- steady wind, altitude shear
and Ornstein-Uhlenbeck turbulence, unobservable by default.

---

## Physics validation

These are meant to be plants you can believe, so every environment carries a
`PHYSICS.md` next to its code: what it models and where that holds, a parameter
table where each constant is cited or shown being derived (and flagged
`TUNED - not sourced` when it is neither), the published numbers it has to
reproduce, and the deviations it knowingly has.

Tests check *consequences* rather than restating the formulas: ISA table
values, L/D ratios, thermal time constants, energy balances, equilibria. A test
that recomputes the code's own expression would pass just as happily on a wrong
one.

A per-environment summary of what each model is validated against, and an
example of something that validation caught, is in
[`docs/PHYSICS_METHODOLOGY.md`](docs/PHYSICS_METHODOLOGY.md#what-each-model-is-checked-against-and-something-it-caught).

**All 21 environments are covered** by fifteen contracts, since the aircraft
variants share the plant they are built on. On top of that, every environment
runs the same conformance suite (determinism, PRNG handling,
`jit`/`vmap`/`scan`, numerical health over a full episode), so the one you pick
behaves like the rest.

Method and reasoning: [`docs/PHYSICS_METHODOLOGY.md`](docs/PHYSICS_METHODOLOGY.md).

---

## Performance

Fast enough that the environment is not the bottleneck: **0.5 M to 1.7 G
steps/s on CPU**, vmapped and scanned inside one `jit`, and higher on GPU. The
spread across the suite is three orders of magnitude, so you can spend a sample
budget where the dynamics are hard and iterate quickly everywhere else.

Per-environment figures, the method, and how to reproduce them:
[`docs/performance.md`](docs/performance.md).

---

## Documentation

| | |
|---|---|
| **[Getting started](docs/getting-started.md)** | Run an episode, vectorise it, plug into Gymnasium |
| **[Target MDPs](docs/target-mdp.md)** | The formal setting the suite is built around |
| **[Environment reference](docs/environments.md)** | All 22: shapes, tracked variables, baselines, contracts |
| **[Public API](docs/api.md)** | What is stable and what is provisional |
| **[Baselines](docs/baselines.md)** | The shipped PID and MPC controllers, and tuning them |
| **[RL protocol](docs/rl-protocol.md)** | How to measure a learned policy so the number means something |
| **[Reward shaping](docs/reward-shaping.md)** | Why the tracking rewards have the shape they do |
| **[Model review checklist](docs/model-review-checklist.md)** | Thirteen checks worth running on any plant model |
| **[Physics methodology](docs/PHYSICS_METHODOLOGY.md)** | How the physics is sourced, validated and bounded |
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
