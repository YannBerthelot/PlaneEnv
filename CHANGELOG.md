# Changelog

Notable changes to TargetGym. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[semantic versioning](https://semver.org/).

Tags before this file were named `vX.Y.Z`, except `0.5.0`; future tags use the
`vX.Y.Z` form.

## [Unreleased]

The work between `0.5.0` and 1.0. Grouped by what it changes for a user rather
than by commit.

### Added

- **Eleven environments**: building HVAC, pH neutralisation, Skogestad's
  distillation Column A, the NREL 5 MW reference wind turbine, a grid battery,
  a boiler drum, and a cement kiln; then four aircraft tasks that move the
  setpoint instead of holding it, since a tuned PID finishes an altitude hold
  with 0.0 m of settled error and can no longer tell two controllers apart:
  `plane_steps` (a staircase), `plane_sine` (a sinusoid, whose amplitude ratio
  and phase lag are the closed loop's frequency response), `plane_energy`
  (altitude and airspeed together, which removes the spare actuator), and
  `plane3d_racetrack` (a holding pattern, which is heading hold and a sustained
  coordinated turn in one task). Twenty-two in total.
- **Environment versioning.** `EnvSpec.version` and `spec.versioned_name` give
  every environment a public identity such as `plane-v1`, stamped in
  `data/env_versions.json` by `scripts/stamp_env_versions.py`.
  `tests/test_env_versions.py` fails when an environment's fingerprint moves
  without its version being bumped, so a published number keeps meaning what it
  meant. Everything ships as `v1`.
- **A harness for learned-policy results**: `data/rl_results.json`, written
  through `target_gym.rl_results.record_result` and guarded by an environment
  fingerprint deliberately narrower than the one the shipped baselines use, so
  re-tuning a controller cannot throw away a training run. The experimental
  design is fixed in advance in `docs/rl-protocol.md`. No results are published
  yet, and nothing in the suite presumes RL beats a PID.
- **`docs/target-mdp.md`**, stating the definition the whole suite is built on:
  the target set, the tracking shape, the admissibility criterion on the target
  and the feasibility condition on the plant.
- **A Colab quickstart** (`notebooks/quickstart.ipynb`), linked from the README
  and the documentation index, and executed by the test suite so it cannot rot.
- `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1).
- **A central registry** (`target_gym.registry`) describing every environment,
  its parameters and its baselines, and a **shared conformance suite** that runs
  the same contracts against all of them -- PRNG hygiene, determinism, the
  gymnax six-value step API, and a PID that beats the best constant action.
- **Physics contracts.** Every environment carries a `PHYSICS.md` with a sourced
  parameter table, published validation targets asserted by tests, and numbered
  known deviations. The method is written up in `docs/PHYSICS_METHODOLOGY.md`.
- **Documentation** under `docs/`: getting started, a public API contract, a
  baselines guide, and an environment reference generated from the registry.
  Every runnable example is executed by the test suite.
- `CONTRIBUTING.md`, `LICENSE`, `CITATION.cff`, and a pre-commit configuration.
- `tracked_names`, `obs_value_index` and `obs_target_index` on every
  environment, so generic tooling can find the tracked variable and its
  setpoint without naming the environment.
- Actuator lag parameters for the aircraft (`power_response_rate`,
  `stick_response_rate`, `aileron_response_rate`), previously literals.
- `CHANGELOG.md`, issue templates and a pull-request template.

### Changed

- **Migrated to the gymnax 1.0 six-value step API.** `step_env` now reports
  natural termination alone; the time limit is gymnax's, via `step`.
- **Every renderer rebuilt** on a shared control-room toolkit
  (`target_gym.render_kit`), and the README gallery regenerated and extended to
  every environment.
- **Seven per-environment figure/video runners consolidated** into one
  registry-driven module, which covers all twenty-two environments rather than
  eight.
- The PyPI development status classifier moves from `3 - Alpha` to
  `4 - Beta`. Not a promise of an API freeze, which stays deferred; a statement
  that the surface is settled enough to build against.
- Python support is 3.11 through 3.14, tested on all four in CI.
- The test suite runs in parallel; full-suite wall time went from about
  fifteen minutes to under two.

### Fixed

- **Four-tank**: the target range sat entirely above what the plant can reach,
  so every episode was unwinnable. The range, the loop pairing (the RGA puts
  λ11 at −0.067, so the loops must be crossed) and the tuner objective were all
  corrected.
- **Aircraft lift curve**: `cl_alpha` was 54 % below what the wing's own aspect
  ratio implies, and the stall clamp was applied before the Prandtl--Glauert
  factor, so peak lift *rose* with Mach instead of falling past `M_crit`.
- **Reactor renderer** produced no frames for short episodes and never reset its
  history between episodes: it advances `state.time` by a control period, so the
  `time == 1` episode-start signal never fired.
- **Bearing-only patrol** gained a baseline, via a lead-state estimator feeding
  the same pursuit law the full-observation variant uses.
- The glass furnace setpoint band was narrowed and given a working tuner.
- **Four-tank gradients.** Outflow goes as the square root of the level and a
  tank can sit empty; written as `sqrt(max(h, 0))` the forward value is right
  but the reverse-mode derivative is NaN at zero, which made gradient-based PID
  tuning return NaN gains from a loss that evaluated perfectly well. Forward
  results are unchanged.
- **Gradient MPC could park an actuator at a limit and never move it again.**
  These plants saturate, an engine cannot make less than zero thrust, and
  saturation is written with `clip` or `maximum`, whose derivative at the kink
  is exactly zero. `GradientMPC._optimize` projected its iterates onto the
  closed action interval, so any overshooting step put an action exactly on a
  bound, where its derivative was then zero and gradient descent could never
  move it again. Measured on `plane_steps`, at the plan where the aircraft gave
  up: the true one-sided slope in thrust is +3.0 and autodiff returns 0.0, with
  thrust pinned at -1.000 for 800 steps while the elevator went on being
  optimised normally. Nothing looks wrong from outside, because a planner that
  has stopped searching still emits finite, in-bounds actions. Iterates are now
  held 1e-3 inside the bounds, which is what interior-point solvers do and for
  this reason. Twelve of the twenty environments with an MPC share it. The
  four plants among them were re-recorded and moved by under half a point of
  return, so the defect was latent there and real only on the aircraft.
- **The aircraft MPC flew the plan into the ground.** With altitude scored and
  two actuators available, the best plan over a 90 s window is a zoom climb
  that trades airspeed for altitude faster than the engines can replace it: on
  `plane_steps` it reached the commanded altitude at t=90 with 30 m/s of
  airspeed left, departed at 91 degrees angle of attack and hit the ground at
  t=372. The planner's objective now carries a barrier on airspeed against the
  stall speed at the current mass and altitude, the pattern
  `make_wind_turbine_mpc` already used. Fencing angle of attack instead does
  not work, since it sits at 4-8 degrees throughout the manoeuvre and only
  crosses 15 degrees one step before the departure. Together with the bound fix
  above, `plane_steps` goes from terminating early on all ten seeds and scoring
  656 against the PID's 1593, to flying full episodes and leading the PID on
  every seed tried.
- **`plane3d_racetrack` could not be rolled out at all.** The class declared
  `obs_value_index` and no `obs_target_index`, which `runners.rollout` reads to
  find the setpoint. The gain search scores candidates inside a `try`, so it
  reported every one as `-inf` and finished successfully having changed
  nothing, and no baseline could have been recorded for the environment. The
  conformance suite now checks both indices across the registry; the tests that
  covered this named three classes by hand, which is how a fourth got past
  them.
- **`plane3d_racetrack` guidance gains**, never previously searched. Settled
  cross-track error goes from 3.02 km to 0.31 km against an 8.4 km turn radius,
  and the return from 269.2 to 319.0, so it holds the pattern rather than
  flying its shape. The roll loop is deliberately held out of the search: left
  free it stiffens the loop and strips its damping, buying return while taking
  the achieved bank to 48 degrees against a 30 degree command limit and making
  the tracking worse. The environment's `expert_degraded` note is gone with it.
- **`render_mode="human"` raised on every environment whose renderer was
  rebuilt on the shared toolkit.** Those draw to an image and return no pygame
  screen, while the Gymnasium wrapper pumped the video system unconditionally,
  so `render()` failed with "video system not initialized". It now pumps only
  when a renderer actually opened a window.

- **The glass furnace was missing the two things that make furnace control
  hard.** It had thermal inertia but no dead time, and a first-order plant with
  no transport delay has no bandwidth limit, so a PID could be tuned arbitrarily
  tight against it: one held the crown to 0.078 K mean against a 10 K open-loop
  drift, roughly thirty times better than a real furnace is held, leaving the
  task no headroom. Its only load variation was an AR(1) drift on pull at a
  50 min correlation time, slower than the plant, and a slow smooth load is
  precisely what integral action cancels perfectly. Now: a 120 s crown
  thermocouple lag on the observation (the reward still scores the true crown
  temperature), 60 s of fuel transport delay, discrete batch charging on a 300 s
  charger cycle with dose-to-dose mass jitter, and the 40 s firing interruption
  at each reversal that deviation D2 had recorded as missing. Open-loop crown
  swing went from 10 K to 31 K, and the PID from 99% of the reward ceiling to
  90%.

- **Setpoint schedules re-derived against what the plants can actually be
  asked.** The 2D aircraft's staircase was a square wave between two altitudes
  2.4 km apart; it is now a ladder of eight levels with adjacent changes of 0.2
  to 0.8 of the amplitude. `plane_sine` commanded a peak climb rate of 20.9 m/s
  against an aircraft that can sustain 14.6, so the target was unreachable by
  construction; its amplitude is now 300 m, peaking at 7.9 m/s. The glass
  furnace's five setpoints were independent draws from a 45 K band, spanning
  30 K on average against the 10-20 K trim its own comment described; the
  schedule is now a bounded walk of at most 6 K a step. Every aircraft task and
  the furnace now start near the level they are commanded, rather than drawing
  the start independently of the target: the 3D tasks began a median 1.6 km
  from their assigned altitude, so the episode opened with minutes of open-loop
  climb.

- **Episode lengths cut where the protocol's own criterion says they are too
  long.** `plane_energy` ran 104 time constants against a suite that clusters at
  10 to 15; the path-following tasks ran up to 3.6 laps where one shows whether
  the path can be flown. Five aircraft episodes shortened, taking the aircraft
  recording from 12.9 h to about 4.5 h. See `docs/rl-protocol.md`.

- **One renderer for every aircraft panel.** `render_aircraft.py` now holds the
  palette, the console chrome, the A320 mesh and its projections; the 2D tasks,
  the 3D tasks and both patrol variants draw with it. `patrol` had been reaching
  into `plane3d.rendering` for eleven private names, so a palette change in one
  aircraft environment silently restyled another and nothing said so. Gallery
  clips are regenerated for both the PID and the MPC, all at 10 fps and about
  ten seconds.

### Removed

- **`plane_steps`**, absorbed into `plane_energy`. They were the same
  environment: same plant, same setpoint schedule, same disturbances, same
  episode length, differing only in `speed_weight` being 0.0 rather than 0.5.
  Two registered environments for one reward coefficient is not two tasks, and
  the pair cost 9 h of the 12.9 h it took to record the aircraft. The pure
  altitude staircase is still reachable as `PlaneParams(speed_weight=0.0)`.

- **Running-cost terms from seven rewards**, for this release line: fuel on the
  glass furnace, the boiler drum and the cement kiln, energy on the building,
  reboiler duty on the column, and reagent on the pH loop. All six weights are
  zero and those tasks score setpoint tracking alone.

  The battery was briefly in this list by mistake. Its `cost_weight` is not a
  consumption cost: it gates the degradation and state-of-charge terms, which
  are what keep that control problem well posed, and zeroing it made the
  optimal policy follow dispatch until the pack hit a limit. Restored, and the
  field now says what it is. Cost is real, but its weight against tracking accuracy silently picks a
  point on a Pareto front, and none of the seven had an argument behind its
  number. The glass furnace showed what that costs: a 0.1 fuel weight made a
  3.3 K standing error the optimum of what its MPC was asked to minimise, so the
  controller sat 6 K cold with fuel at minimum 80% of the time and lost to its
  own PID. The fields and the terms are still wired, so a weight can be restored
  once there is a defensible way to set one.

- Dead modules carrying no importers: `experts/degradation.py`,
  `experts/cpg.py` and `experts/pd.py` (the latter two were Brax/MuJoCo
  locomotion experts for environments this project does not have), and
  `scripts/benchmark_integration.py`, which imported an undeclared dependency
  and could not run.

### Known gaps

- Both patrol variants ship a PID, but it holds formation only loosely --
  roughly 139 m of settled slot error against a 60 m tolerance, pinned by six
  `strict` xfail cases.
- ~~Two of the seven gradient PID tuners return NaN gains.~~ **Resolved**, and
  the cause turned out to be one line. `plane.dynamics.aero_coefficients` wrote
  its stall blend as `CL_linear / (1 + exp(u))`; past about 77 degrees of
  incidence the exponent overflows float32, `exp` returns `inf`, and the
  reverse-mode derivative of `x / (1 + inf)` is NaN even though the forward value
  is a perfectly good 0. Any rollout long enough for the aircraft to depart
  reached that incidence, so every gradient through it came back NaN. Written as
  `jax.nn.sigmoid`, the same function evaluated stably, all seven tuners pass.
- No published RL baseline results yet.

[Unreleased]: https://github.com/YannBerthelot/TargetGym/compare/0.5.0...HEAD
