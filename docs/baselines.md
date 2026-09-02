# Baselines

Every environment ships a controller so that a learned policy has something
real to beat. A benchmark whose only reference point is a random policy tells
you an agent learned *something*; one with a tuned PID tells you whether it
learned anything worth having.

Reach a baseline through the registry rather than by importing a factory:

```python
from target_gym.registry import REGISTRY

spec = REGISTRY["cstr"]
env, params = spec.make_env(), spec.params_cls()

pid = spec.make_pid()
pid.reset()

mpc = spec.make_mpc(env, params)
mpc.reset()
```

## Coverage

All eighteen environments ship a PID. Sixteen also ship an MPC; the two
`patrol` variants do not, and `EnvSpec.baselines_note` records why -- the
follower's plant is the full 3D aircraft and its reference is a *manoeuvring
lead*, so an MPC needs the lead's future trajectory as a time-varying
parameter, which is not yet wired.

A missing baseline is a documented gap rather than a silent one: the
conformance suite reads `baselines_note` and skips with that reason, so a
baseline cannot quietly disappear.

## PID

The PID baselines are stateful objects with `reset()` and `__call__(obs)`.
They range from a single loop to gain-scheduled and cascaded structures, and
for the multi-loop plants a MIMO form with a deliberate pairing -- the
four-tank's loops are **crossed**, because its relative gain array puts
λ11 at −0.067 and the obvious pairing is unstable.

Gains are tuned by `scripts/tune_pid.py` and cached in `data/pid_gains.json`:

```bash
uv run python scripts/tune_pid.py --envs cstr    # or `make tuning-cstr`
make clear-tuning                                # drop the cache and retune
```

The script skips any environment already present in the cache, so re-running it
without `--envs` or `make clear-tuning` is a no-op for everything already tuned.

Two caveats worth knowing before you re-tune anything:

- **Re-tuning can make an environment worse.** The searches are stochastic and
  the relay experiment is sensitive to the operating point, so a fresh "best"
  is not automatically better than what is shipped. Measure both before keeping
  one: re-running the tuner over the whole registry produced a genuinely better
  `cstr` and a distinctly worse `first_order` in the same pass.
- **A tuned gain cannot rescue an infeasible task.** The circle expert's radial
  gains were searched to convergence against a task that, on a third of its own
  radius range, no gain could fly: holding 230 m/s around an 8.4 km circle needs
  33.8 deg of bank against a 30 deg limit, so the aircraft sat pinned at the
  limit for the whole episode. Giving it the airspeed the radius admits was
  worth 31% of the return, against nothing at all from further tuning. Check
  feasibility before searching gains.

- **Check the winner is not the last point in the grid.** The glass furnace's
  search is a grid over (Kp, Ki, Kd), and it returned `Kp=0.040` -- the largest
  value in `kp_grid`, with the score rising monotonically across the entire Kp
  column right up to it. That is the signature of a boundary solution, not an
  optimum, and it is invisible in the output unless you compare the winner
  against the grid's own bounds. Probing outward found the real turn at
  `Kp~5`, roughly 125x further out, and `Ki` was then pinned at *its* edge as
  well. Widening both grids moved the furnace PID from 144.1 to 149.1 over ten
  seeds. The grids now bracket their optima on both sides, and the chosen
  `Ki=0.15` was verified interior by direct probe rather than assumed.

  Beware the interaction: `Ki` and `Kd` had been chosen while `Kp` was pinned,
  so all three had to be re-examined at the new operating point rather than
  just the one that hit the wall.

- **The aircraft are tuned by coordinate descent, not by the relay.** Both of the
  other methods fail on these plants, and structurally rather than by bad luck:
  the relay reports *"every operating point failed (no zero-crossings)"* because
  the altitude/power loop will not sustain a bang-bang oscillation, and the
  gradient tuner returns NaN gains (a documented xfail in
  `tests/experts/test_pid_tuning.py`, hardened against the obvious causes and
  still not localised). Coordinate descent on episode return needs neither an
  oscillation nor a derivative, only forward rollouts.

  It scores **return**, not tracking error. Scoring one term of a
  multi-objective reward is the mistake that made the MPC baselines look broken
  for a long time; the same trap applies here.

  Two bugs were fixed alongside it. Every tuner in both systems pinned
  `integration_method="rk4_1"`, so they would have tuned against the plant as it
  was before the integration order was corrected. And the 2D aircraft's tuner
  wrote the `plane` key while its shipped autopilot reads `plane_cascaded` --
  a key that had never existed in the file. `make tuning-plane` therefore ran,
  reported success, wrote gains nothing loaded, and left the controller on its
  constructor defaults. The tuner now seeds from those defaults and writes the
  key the controller actually reads.

  Gains improved on seeds the search never saw, which is the only number worth
  quoting (search on seeds 0-2, held out 3-9):

  | | shipped | tuned | |
  | --- | --- | --- | --- |
  | plane | 194.62 | 285.20 | +47% |
  | plane3d_heading | 60.93 | 80.25 | +32% |
  | plane3d_circle | 67.23 | 84.59 | +26% |
  | plane3d_figure8 | 34.15 | 81.58 | +139% |

  Those four returns were measured under the previous reward, which charged a
  flat -200 for a crash; the ratios are what the table is for, and the gains
  themselves were re-checked against the current reward and did not move.

## MPC

Three implementations, chosen per environment by what its dynamics allow:

| Implementation | Used by | When it applies |
|---|---|---|
| `CasadiMPC` subclasses | 7 environments | A direct nonlinear program over an explicit model; the sharpest when the model can be written in CasADi |
| `GradientMPC` | 8 environments | Differentiates the JAX dynamics directly and descends the objective |
| `SamplingMPC` | cement kiln | Cross-entropy sampling, for when gradients are unusable |

The cement kiln uses sampling because its adjoint overflows: half its response
to a fuel change takes a full 25-minute residence time, and differentiating
back through that transport delay does not survive in floating point.

An MPC objective must share the **minimiser** of the environment's reward, not
its shape. A reward with a flat or clipped region is fine to score against but
useless to descend, so the MPC objectives are written to be smooth where the
reward is not.

MPC rollouts are expensive, so episodes are cached under `data/mpc_cache/`:

```bash
make clear-mpc     # drop the MPC trajectory cache
```

### Horizons, and which ones are too short

`scripts/audit_mpc_horizons.py` checks each MPC's horizon against `tau_close`,
the time a *viable* controller needs to bring the tracking error to 1/e and keep
it there. A receding-horizon controller can only optimise what it can see, so
`horizon * mpc_dt` has to cover that transient. Most environments pass with room
to spare; two groups do not:

| Environment | horizon | `tau_close` | ratio | |
|---|---|---|---|---|
| `plane` | 30 | 37 | 0.81 | myopic |
| `plane3d_heading` | 30 | 40 | 0.75 | myopic |
| `plane3d_circle` | 30 | 40 | 0.75 | myopic |
| `four_tank` | 5 | 198 | 0.03 | myopic |

The aircraft cases are **not** fixed by nudging the horizon to meet the
criterion. Measured on `plane3d_heading` over 150 steps, horizon 30 and horizon
40 both leave the altitude error *larger* than it started (3228 m and 3168 m
against an initial 2623 m) for 17% more compute -- the difference is noise. An
earlier measurement at horizon 80 did help substantially (921 m against 1741 m),
so the horizon really is the binding constraint, but the useful size is several
times the audit's minimum and costs roughly 9x. These are `GradientMPC`
instances, which roll out `step_env` itself, so covered time cannot be bought
with a coarser `mpc_dt` the way the CasADi controllers allow.

`four_tank` is the CasADi case where that trick does apply: at ratio 0.03 it is
the worst in the suite, and a coarser prediction step would buy the covered time
at the same optimisation cost.

Both are open items rather than tuning knobs, and neither is affected by the
reward shape -- `GradientMPC` sums the environment's reward directly, so it
picks up reward changes without any objective to re-derive.

## Regenerating the figures and videos

```bash
make figures          # or figures-<env>
make videos           # or videos-<env>
make short-gifs       # lightweight *_short.gif copies, which are what is committed
```

The committed media does **not** currently round-trip through these targets, and
that is worth knowing before you regenerate anything:

- The runner writes `sweep.png`, `pid_response.png` and `comparison.png`, none of
  which are tracked. The five tracked `figures/**/*.png` come from an older
  script and are not reproduced by `make figures`.
- `make videos` writes the 3D aircraft tasks to `videos/plane3d_heading/` while
  the committed gifs live at `videos/plane3d/heading_short.gif`.
- Regenerating `cstr` produces a 5-frame 1400x750 gif where the committed one is
  80 frames at 760x407, so the episode length and figure size used for the
  committed media are not the current defaults.

Until that is reconciled, regenerate media deliberately and compare frame counts
and sizes before committing, rather than taking whatever the target emits.

## What the suite guarantees

One contract, asserted for every registered environment, runs in the `slow`
job: the PID must beat the **best constant action**. A weak bar, but exactly
the one a mis-indexed setpoint fails -- it caught a furnace PID tracking fuel
percentage as its temperature setpoint.

The MPC now has a contract too, in the same job: it must not end the episode in
a terminal state, and must not return materially less than the PID. Until it
existed, `tests/experts/test_mpc_baselines.py` asserted only that a controller
built and emitted finite, in-bounds actions -- which is exactly what a
controller that has given up does, so an MPC returning -0.02 against a PID's 393
passed for as long as it was there.

The bar is deliberately loose (10% of the PID's return, five seeds, episodes
capped at 250 steps). It is a tripwire against gross regression, not
the published comparison: the numbers below were measured over ten seeds, and
the tolerance was set from the real defects rather than chosen. Verified by
reverting each fix: the wind turbine (terminates at step 20 of 400) and the
battery (-27.5%) are caught; the glass furnace is *not*, and that is the
contract's honest limit -- its bug costs -17.6% over ten seeds but only -4.4%
over five, against +3.7% when fixed, and no sane threshold separates those.
Subtle objective errors are below its resolution; this table is what finds
them. Those three percentages were measured under the previous reward and have
not been re-derived -- reverting each fix again costs hours and would restate a
conclusion about the contract's *resolution*, which the reward change does not
alter. Two aircraft are recorded as `EnvSpec.mpc_degraded` and
xfail with their measured reasons, so a known gap is explicit rather than
absent.

## MPC against PID, ten seeds

Return, paired per seed, on each environment's own episode. Every number here
was re-measured after the reward unification -- returns are not comparable
across that change, so the previous table was discarded rather than patched.

Both the mean and the median are given. They disagree on two rows, in opposite
directions, and either one alone would misreport the pair.

| environment | mean | median | seeds won |
| --- | --- | --- | --- |
| plane3d_figure8 | **+450.8** | +449.6 | 10/10 |
| plane3d_heading | **+236.9** | +314.1 | 8/10 |
| plane3d_circle | +145.1 | +200.1 | 6/10 |
| four_tank | +56.9 | +64.6 | 10/10 |
| boiler_drum | +51.4 | +54.9 | 10/10 |
| plane | +33.8 | +30.2 | 8/10 |
| ph_neutralization | +33.4 | +29.8 | 9/10 |
| distillation | +29.2 | +26.6 | 10/10 |
| reactor | +26.6 | +28.1 | 10/10 |
| wind_turbine | +11.8 | +13.9 | 9/10 |
| cement_kiln | +9.4 | +9.3 | 10/10 |
| hvac | +8.4 | +8.4 | 10/10 |
| cstr | +5.3 | +4.5 | 10/10 |
| first_order | +2.5 | +2.3 | 10/10 |
| glass_furnace | -7.0 | -5.2 | 2/10 |
| battery | +14.0 | **-4.1** | 1/10 |

The MPC is the upper bound on **fourteen of the sixteen**, and behind on the
glass furnace and the battery. Both shortfalls are inside the contract's 10%
tolerance (4.7% and 2.6%), so this is a documented gap rather than a failure.

**The battery's mean is the wrong statistic.** It is carried by a single seed
where lookahead pays enormously -- 350 against the PID's 164 -- while the
controller trails on the other nine for a median of -4.1 and one win in ten.
Horizon, iterations and step size were all swept without closing it.

**The glass furnace lost this row to a better opponent, not to a regression.**
Its MPC is unchanged and scores exactly what it scored before (142.09). What
moved was the PID: its gains had been pinned at the edge of the tuner's search
grid, and widening the grid took it from 144.1 to 149.1. That was enough to
turn a split -- MPC ahead on 7 of 10 seeds -- into a clear PID win at 2 of 10.
Strengthening a baseline is supposed to be able to do this, and reporting the
flip is the point of tuning the baseline honestly in the first place.

The aircraft rows now carry win counts. The previous table quoted their margins
as a difference of means with no per-seed count, because the PID column had been
re-tuned while the MPC column had not, and re-running the MPC cost hours. Both
columns are current here, so the counts are real: the MPC leads on all four.

The circle row moved after the table was first measured, and not because of the
MPC, which is unchanged at 361.23. Its PID gained 31% -- 165.5 to 216.1 -- when
the expert was given the ability to trade speed for turn radius, without which a
third of the task's own radius range is unflyable at the cruise it holds (check 9
in the model review checklist). A stronger baseline narrows the MPC's lead from
+195.8 to +145.1 and its win count from 8 of 10 to 6, which is what a better
opponent is supposed to do.

**The aircraft rows are the second measurement.** On the previous dynamics the
2D plane scored -171 and the 3D heading task -34, each winning most seeds and
losing the average to three terminations worth -600 apiece. A great deal of
effort went into those crashes -- a stall-margin barrier, an altitude barrier,
tails out to 240 steps, a crash charge matched to the environment's own penalty,
and making terminations visible to the planner -- and the one that helped fixed
a single seed. None of it was the cause. The integrator was: at one RK4 substep
the plant the MPC plans against and the plant it is stepping through disagree
enough to fly into the ground. At two substeps both controllers win 10 of 10
with no terminations at all.

The machinery was then re-measured rather than left to rot. The 2D aircraft's
terminal cost -- `n_tail=60`, which holds the last action and scores the flight
that follows -- still earns its place: five seeds out of five with it, four
without, and 61 more return. Those figures predate the reward unification, but
the argument for it got *stronger*, not weaker. Removing the environment's flat
crash charge means forgone reward is now the entire cost of a crash, and a
planner can only see forgone reward by looking past its own horizon. Its
stall-margin barrier does not earn its place: it adds 1.3%, inside this
machine's noise, and it existed only to fight the crashes, so it has been
removed.

The wind turbine's overspeed barrier and the battery and furnace surrogates
were re-verified against the new reward rather than assumed. The barrier is
still the difference between controlling and tripping (172.1 with it, 52.4
without, and 6 of 6 episodes ending early on the overspeed trip). The two
surrogates survive for a reason that changed: they were written to route around
a clipped tracking term that was exactly flat outside its band, and no reward in
the library has such a term any more. Log-scaling fixes the *value*, though, not
the gradient -- see "The MPC objective is not the reward" below.

Two seeds would misreport almost everything. Measuring on two produced three
wrong conclusions during this work -- the wind turbine at "98% of the PID", the
2D aircraft at "no crashes", and an original verdict of MPC ahead on 14 of 16 --
each overturned by widening the seed count. Nothing here is quoted below ten.

### Why these failed, which was never tuning

Every MPC that lost to its PID was given an objective it could not descend, or
one that did not share the reward's minimiser:

- **wind_turbine** and **battery** scored tracking as `clip(1 - err/band, 0, 1)**2`.
  One step outside the band leaves that term flat, so the only surviving gradient
  belongs to the *penalty* terms -- and the planner is then correctly guided to
  stop acting. Both were fixed by a smooth surrogate with the same minimiser.
- **glass_furnace** normalised its error by the crown's whole 250 K envelope
  where the reward uses 40 K, six times too flat against an unchanged fuel
  penalty, and its term turned back upward past the band so that beyond twice it
  the objective preferred *more* error.
- **plane** and **plane3d_heading** crash, and a crash penalty behind
  `where(terminated, ...)` is a boolean: it carries a cost but no gradient away
  from the boundary. A differentiable barrier on the approach is what works; it
  fixed the wind turbine's overspeed trip and one of the plane's two failing
  seeds.

The one time the *optimiser* was improved instead -- swapping the aircraft's
gradient descent for Adam, which tripled the predicted return -- closed-loop
performance got dramatically worse (737 m to 4527 m, with new crashes). A weak
optimiser was masking the myopia; pursuing a truncated-horizon objective harder
just exploited it. Fix the objective first.

## Structure over gains

The right structure usually matters more than the numbers, and the shipped
baselines are chosen to show it:

- **Three-element control on the boiler drum.** Feedwater tracks measured steam
  flow as a feedforward, so shrink-and-swell cannot fool the level loop: the
  drum level *rises* when steam demand increases, and a naive level controller
  responds by cutting feedwater at exactly the wrong moment.
- **A cascade on the cement kiln.** Integral action on a measurement half an
  hour old oscillates at the delay period; an inner loop on a faster
  measurement is what makes the outer loop tractable.
- **Crossed loops on the four-tank.** Its relative gain array puts λ11 at
  −0.067, so pairing each pump with the tank beneath it -- the obvious choice --
  is unstable. The shipped PID pairs them the other way.
- **A cascaded autopilot for aircraft altitude** (altitude → vertical speed →
  pitch → elevator) with attitude limiting and angle-of-attack protection. A
  single loop mapping altitude error straight to elevator departs controlled
  flight on large climbs.

## The MPC objective is not the reward

An MPC objective must share the reward's *minimiser*, not its shape. Copying a
clipped tracking reward gives the optimiser no gradient exactly where it is
needed; dropping the clip makes large errors score better than they should.
Both failures happened here before the objectives became plain quadratics.

Unifying the rewards on a log scale removed every clipped plateau, so the
obvious next step was to delete the two surrogates and let the planners descend
the reward itself. Measured, that is clearly wrong: the turbine scores 341.9 on
its surrogate against 172.1 on the reward, with the same barrier in both.

The reason is worth stating, because it is a property of log-scaled rewards in
general and not a defect in these two plants. A log-scaled reward is
scale-free in *value* -- every halving of the error is worth the same increment,
which is exactly what makes it good to learn from. Its gradient is not
scale-free. Differentiating

    r(e) = 1 - log1p(e / floor) / log1p(envelope / floor)

gives `-1 / ((floor + e) * log1p(envelope / floor))`, which decays like `1/e`:
the pull toward the setpoint is *weakest* precisely where the controller is
furthest from it. A quadratic in the normalised error has the same minimiser and
a gradient that instead grows with the error.

So the surrogates are no longer workarounds for a broken reward. They are
planner-side reformulations of a reward that is now correct -- which is an
ordinary thing for an MPC to carry, and the distinction matters for anyone
reading them as evidence that the reward needs fixing.

The cement kiln is the clearest case for choosing the implementation to fit the
plant: its free lime depends on temperature through a 280 kJ/mol Arrhenius term
that is then advected down the kiln, so reverse-mode gradients overflow to NaN
after about eight steps while finite differences on the same objective stay
clean. Hence cross-entropy sampling rather than a gradient method.
