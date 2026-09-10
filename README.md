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
2 renewable energy. Every one ships with MPC, PID and RL baseline controllers.

**TargetGym** is a collection of JAX **reinforcement learning environments**
built around [**target MDPs**](docs/target-mdp.md): tasks where you **reach a
target and then hold it**, instead of reaching a goal once and stopping. Most
industrial control works that way. You keep a setpoint against disturbances,
indefinitely.

They are fast (0.5-17 M steps/s on CPU) and end-to-end GPU compatible, with
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
from target_gym import Plane, PlaneParams
from target_gym.registry import REGISTRY

env, params = Plane(), PlaneParams()
obs, state = env.reset(jax.random.PRNGKey(0), params)

# Every environment ships a tuned expert, so a learned policy has something
# real to beat. The docs say how good each one is.
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

Or use your favourite RL library, JAX or not (stable-baselines3 here). Note
that you only get end-to-end GPU with a JAX-based one:

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

Every environment comes with visuals. The non-aircraft plants share one control-room
toolkit (`target_gym/render_kit.py`): a live plant schematic, an instrument
stack with limit and setpoint markers, and strip charts. The schematics are
drawn from state the controller usually *cannot* see, like riser voidage,
thermal mass, or the kiln's axial profile. So a frame shows you both what the
agent measures and what it is actually up against, with a purple dot on each
hidden quantity.

### Complexity

There is a wide difficulty range here, so you can use the suite as a curriculum
and not just a benchmark. Tiers weigh both the **dynamics** (linearity,
coupling, stiffness) and the **RL side** (dimensionality, horizon, partial
observability).

| Tier | Environment | Obs | Act | Dynamics | Key RL challenges |
|---|---|---|---|---|---|
| 1 (Trivial) | First Order System | 2 | 1 | Linear SISO | Baseline sanity-check |
| 2 (Medium) | CSTR | 3 | 1 | Nonlinear SISO | Exponential Arrhenius kinetics, stiff dynamics, exothermic runaway risk |
| 3 (Hard) | Building HVAC | 7 | 1 | Linear RC network | **Partial observability** (thermal mass hidden), 43 h time constant, setback anticipation, comfort/energy trade-off |
| 3 (Hard) | Four Tank | 6 | 2 | Nonlinear MIMO | **Non-minimum phase** (gamma1+gamma2 = 0.4): the RGA element is *negative*, so the obvious diagonal pairing is unstable and the loops must be crossed |
| 3 (Hard) | Grid Battery | 5 | 1 | Nonlinear ECM | **Finite budget**: tracking now costs the ability to track later; irrecoverable charge limits, state-dependent efficiency |
| 4 (Very Hard) | Wind Turbine | 6 | 2 | Nonlinear aero-elastic | Turbulent unmeasured inflow, region switching, drive-train torsion, thrust/power trade-off |
| 4 (Very Hard) | pH Neutralisation | 3 | 1 | Implicit algebraic | 45x steady-state gain variation across the range, unmeasured buffering, same pH from different states |
| 4 (Very Hard) | Binary Distillation | 6 | 2 | Stiff nonlinear MIMO | **Ill-conditioned** (condition number ~140): the two purities move together far more easily than apart |
| 4 (Very Hard) | Plane 2D | 10 | 2 | 2D aerodynamics | Coupled nonlinear aerodynamics, very long horizon |
| 4 (Very Hard) | Glass Furnace | 5 | 1 | Nonlinear radiation (T^4) | **Partial observability** (6/9 states hidden), regenerator reversal cycle, multi-hour transients, batch-blanket nonlinearity |
| 4 (Very Hard) | Nuclear Reactor | 4 | 1 | Stiff multi-timescale | **Partial observability** (7/11 states hidden), xenon memory trap, 86k-step horizon |
| 5 (Extreme) | Boiler Drum | 7 | 2 | Nonlinear, two-phase | **Non-minimum phase**: the level's first move is the wrong way. Integrating output, irrecoverable trips both sides, hidden riser voidage |
| 5 (Extreme) | Plane 3D Heading | 15 | 3 | 3D aerodynamics | Multi-objective (altitude + heading), roll/pitch/yaw coupling |
| 5 (Extreme) | Plane 3D Circle | 17 | 3 | 3D + path following | Sustained coordinated banked turns, km-scale circular path |
| 5 (Extreme) | Plane Patrol | 26 | 3 | 3D + moving target | **Non-stationary manoeuvring reference**, relative-frame observation, collision (irrecoverable) |
| 6 (Extreme+) | Cement Kiln | 8 | 2 | Distributed (1D advection + Arrhenius) | **Transport delay**: half the response to a fuel change takes a full 25-min residence time. 64 hidden states behind 8 measurements, one input that moves the delay itself |
| 6 (Extreme+) | Plane 3D Figure Eight | 19 | 3 | 3D + twisted lemniscate | 3D path with altitude crossovers, direction reversal |
| 6 (Extreme+) | Plane Patrol MARL | 18 / 26 | 3 + 3 | 3D two-body | **Multi-agent coordination**, non-stationary co-player, shared collision state |

The target-pattern variants (`plane_sine`, `plane_energy`) and
the 3D holding pattern (`plane3d_racetrack`) share their base aircraft's plant
and therefore its tier; what differs is the reference they must track.

---

## Baselines

All 21 environments ship a tuned PID, and 19 of them an MPC as well, so you
have something real to beat from the first run. The two **patrol** variants have
no MPC: their reference is a manoeuvring lead, and an MPC would need its future
trajectory to plan against.

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

On top of those, the suite deliberately models:

* **Actuator lag**: inputs such as engine power take time to fully apply.
* **Competing objectives**: reach the target quickly while minimising
  overshoot or cost.
* **Anticipation**: scheduled setpoints reward acting *before* the step. The
  building's night setback and the furnace's crown schedule are both unreachable
  by a purely reactive controller.
* **Moving, non-stationary targets**: the patrol slot tracks a manoeuvring
  lead aircraft.
* **Multi-agent coordination**: the MARL patrol task requires the team to
  cooperate on formation *and* trackable flight.
* **Disturbances**: every plane-based environment inherits a full wind model
  as a physics-engine property: steady wind (`wind_x/y/z`), altitude-dependent
  **shear** (`wind_shear_x/y`), and **Ornstein-Uhlenbeck turbulence**
  (`turbulence_sigma`), all applied to the air-relative aerodynamics. Formations
  feel one shared gust field. Wind is *unobservable* by default;
  `Plane(observe_wind=True)` exposes it for a fully-observable baseline.

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

<details>
<summary>What each model is checked against, and something it caught</summary>

| Environment | Validated against | Example finding |
|---|---|---|
| Plane 2D | ISA atmosphere tables, A320 figures of merit | Lift-curve slope was 54 % below what its own aspect ratio implies, putting clean stall speed at 228 kt instead of ~150 kt |
| Glass Furnace | Published float-furnace data (4-6 GJ/tonne, 24-30 h residence) | Regenerators were absent entirely, so the energy balance was out by ~2x |
| Nuclear Reactor | Keepin 1965 delayed-neutron data, the inhour equation, published Xe-135 behaviour | The reactivity budget leaves only 30 pcm of rod margin at full power, which is what gives the xenon pit its teeth |
| Building HVAC | ISO 13790 5R1C; heavyweight-dwelling time constant and design load | Daily temperature cycle was inverted, coldest at 15:00 |
| pH Neutralisation | Gustafsson & Waller / Henson & Seborg reaction-invariant benchmark | Nominal design point reproduces pH 7.03, pinning feeds and flows jointly |
| Binary Distillation | Skogestad "Column A" (41 stages, alpha = 1.5) | Perturbation-derived gain matrix contradicted the mass balance, because the steps had not converged |
| Wind Turbine | NREL 5 MW reference turbine definition | A Region 2 torque cap made things worse: it only binds *below* rated speed |
| Grid Battery | Published Li-ion grid-BESS behaviour (round-trip, voltage window, thermal rise) | Sizing caught three errors before coding: 0.05 ohm gives 79 % round-trip, passive cooling implies a 438 K rise, OCV exceeded the 4.2 V ceiling |
| Boiler Drum | IAPWS steam tables, Astrom & Bell drum geometry, circulation ratio 5-15 | Tracking riser steam as *quality* rather than mass suppressed the swell entirely. Every coefficient was correct, and still no inverse response |
| Cement Kiln | Published 3.0-3.5 MJ/kg heat consumption, Sullivan residence correlation, 0.5-2 % free lime | An energy audit caught the kiln being fed *raw* meal instead of calcined hot meal, overstating its thermal load by ~50 % |
| Four Tank | Johansson (2000); RGA, reachability of the target box | The sampled targets sat entirely **above** what the plant can reach, so every episode was unwinnable, and the loops were paired the unstable way round |
| CSTR | Steady-state multiplicity, branch stability | The 350 K runaway trip sits exactly where the unstable middle steady state does, so termination fires as the reactor ignites |
| 3D Aircraft | Coordinated-turn relation, load factor | Banked flight reproduces psi_dot = g tan(phi)/V to within 0.5 %, though nothing in the model computes a turn rate |

</details>

**All 21 environments are covered** by fifteen contracts, since the aircraft
variants share the plant they are built on. On top of that, every environment
runs the same conformance suite (determinism, PRNG handling,
`jit`/`vmap`/`scan`, numerical health over a full episode), so the one you pick
behaves like the rest.

Method and reasoning: [`docs/PHYSICS_METHODOLOGY.md`](docs/PHYSICS_METHODOLOGY.md).

---

## Performance

Throughput is measured with `python -m target_gym.benchmark_speed`: 256
environments under `vmap`, stepped 800 deep inside one `jit`-compiled `scan`, on
CPU, which is how an RL loop actually drives them. Figures scale with batch
size and are much higher on GPU.

| Environment | Steps/s (CPU, vmap 256) | | Environment | Steps/s (CPU, vmap 256) |
|---|---|---|---|---|
| First Order System | ~1670 M | | Wind Turbine | ~17.8 M |
| CSTR | ~111 M | | Building HVAC | ~17.1 M |
| Four Tank | ~101 M | | Boiler Drum | ~10.0 M |
| Grid Battery | ~7.5 M | | Glass Furnace | ~3.0 M |
| pH Neutralisation | ~2.1 M | | Plane 2D | ~1.4 M |
| Nuclear Reactor | ~1.3 M | | Plane 3D (all) | ~0.9 M |
| Cement Kiln | ~0.7 M | | Binary Distillation | ~0.5 M |
| Plane Patrol | ~0.5 M | | | |

The spread is useful: a first-order lag and a distributed kiln are three orders
of magnitude apart, so you can spend your sample budget where the dynamics are
actually hard and iterate fast everywhere else.

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
