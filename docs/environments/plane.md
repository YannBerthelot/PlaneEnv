# Plane

<p align="center"><img src="../videos/plane/pid_output_short.gif" width="480px"/></p>

| | |
|---|---|
| Action space | `Box((2,))`, all actions in [-1, 1] |
| Observation space | `Box((9,))` |
| Tracked variable(s) | altitude (m) |
| Episode length | 280 steps (280 s at dt = 1 s) |
| Import | `from target_gym import Airplane2D, PlaneParams` |

## Action space

Actions are normalised to `[-1, 1]` and mapped onto the plant's real
actuator range inside the environment.

| # | meaning | min | max |
|---|---|---|---|
| 0 |  | -1 | 1 |
| 1 |  | -1 | 1 |

## Observation space

9 values. Indices (1,) carry the tracked variable(s) that the
reward scores.

## Rewards

Return reward for a given state. Safe for JIT.

Every environment in this suite scores on one contract: the reward is
`(tracking terms, multiplied) x (1 - weighted costs)`, bounded in
`[0, 1]`, and reaches 1 only while the target is held exactly. See
[Reward shaping](../reward-shaping.md).

## Starting state

`reset` samples the initial condition and the target; state has 16 fields.

## Episode end

**Termination.** Return True if the episode should terminate.

**Truncation.** After 280 steps.

## Baselines

Measured over 10 seeds on a 280-step episode (see [Baselines](../baselines.md)):

| controller | return | per step |
|---|---|---|
| PID | 170.7 | 0.609 |
| MPC | 205.4 | 0.734 |

## Arguments

| parameter | default |
|---|---|
| `delta_t` | 1 |
| `max_steps_in_episode` | 280 |
| `gravity` | 9.81 |
| `initial_mass` | 73500 |
| `thrust_output_at_sea_level` | 240000 |
| `air_density_at_sea_level` | 1.225 |
| `frontal_surface` | 12.6 |
| `wings_surface` | 122.6 |
| `C_x0` | 0.095 |
| `C_z0` | 0.9 |
| `initial_fuel_quantity` | 19088 |
| `specific_fuel_consumption` | 0.0175 |
| `cl_alpha` | 0.08786 |
| `cl0` | 0.2 |
| … | 34 more, see the params dataclass |

