# Roadmap and known gaps

Moved out of the README, which is the shop window rather than the planning
board. This is the honest state of the project: what is done, what is next, and
what is broken and recorded rather than hidden.

## Roadmap

* [x] Mature the glass furnace and reactor environments (physics, reward shaping, episode lengths).
* [x] Document and test every environment's physics against published data.
* [x] Rebuild every renderer on a shared control-room toolkit, and regenerate
      the gallery clips against it.
* [ ] Restore the Plane Patrol baselines with pursuit guidance (see *Baseline coverage*).
* [ ] Add microburst / spatially-varying wind fields (position-dependent, not just altitude-linear).
* [ ] Provide benchmark results for popular RL baselines.
* [ ] Add random orientation variations to circle and heading tasks.

### Before 1.0

* [ ] **Host the documentation.** `docs/` is written and its examples are
      executed by the suite, but it is read as Markdown on GitHub. A GitHub
      Pages site (MkDocs Material) would give it navigation, search and a
      versioned URL, built and deployed from the same workflow that tests it.
* [ ] **Publish RL baseline results.** The environments claim a learned policy
      has something real to beat; no learned policy's numbers are published yet.
      The harness is in place -- `data/rl_results.json`, written through
      `target_gym.rl_results.record_result` and guarded by a fingerprint of the
      environment, so a result recorded before a reward or dynamics change is
      refused rather than quoted. Training runs outside this package (the
      dependency goes RL-library-to-here, never the reverse); see
      [docs/rl-baselines.md](docs/rl-baselines.md).
* [x] **Drop the git dependency on `gymnax`.** Gone, and it turned out not to be
      needed. The pin tracked upstream `main` on the reasoning that released
      gymnax 1.0.0 caps `gymnasium<1.2` and that "conflicts with newer
      gymnasium" -- but nothing in this project requires newer gymnasium. It
      declares `gymnasium>=1.1,<1.4`, and 1.1.1 satisfies that. Resolving from
      PyPI alone gives gymnax 1.0.0 with gymnasium 1.1.1, on which the whole
      suite passes unchanged: 1225 fast, 69 slow, same four and two xfails.
      The tested configuration is now reproducible from PyPI, which was the
      point. `uv.lock` carries no git dependencies at all.
* [x] **A performance phase.** Four defects, all paid by every user and none
      visible to a throughput benchmark, which measures steady state after
      compilation. Every environment returned a *weakly typed* reset state, so
      anything jitted over the state compiled twice. Each new environment
      instance retained a compiled executable, leaking ~2.6 MB per construction.
      The pH solver spent its whole runtime on 44 bisection halvings resolving to
      1e-13. And `runners.rollout` re-jitted the environment on every call, so a
      warmed rollout still spent 0.222 s of 0.355 s compiling. Fast CI 164 s ->
      128 s, `tests/experts` 143 s -> 37 s, warm rollouts ~5x, pH throughput 2x.

      Two restructurings were measured and rejected: vectorising the aircraft's
      three aerodynamic calls into one is 0.76x, and `donate_argnums` on the
      batched rollout does nothing (the carried state is 0.26 MB). The remaining
      slow environments are honestly slow -- distillation needs 16 substeps
      across 41 stages for stability, the cement kiln sweeps 16 zones in
      sequence.

      Two cautions for whoever picks this up. Benchmarks on a laptop vary 41%
      across identical trials, so every number here is a min of many; a
      single-shot measurement produced a confident and wrong conclusion partway
      through this work. And the pass found a *correctness* bug while looking for
      speed -- see the integration order note below -- which is the main reason
      it was worth doing.

      The table's throughput column has since been re-measured with
      `python -m target_gym.benchmark_speed` (batch 256, best of three, after
      warm-up). Every process plant came back within 10% of its published figure,
      which is what makes the aircraft rows conclusive: all five were about 2x
      optimistic, because the post-stall aerodynamics, the three moment
      decompositions, pitch damping and fuel burn were added to those dynamics
      after the numbers were taken. They now read as measured.

      Throughput is also strongly batch-dependent for the aircraft, which the
      single number does not convey: the 3D plane roughly doubles between batch
      256 and batch 16384. Anyone training on these should batch at 4096 or more.

      The aircraft rows fell again when the integration order was corrected from
      one RK4 substep to two (see `plane3d/PHYSICS.md`). That is the honest cost
      of a converged trajectory: at one substep the altitude was 20 m out over
      150 steps, against a reward that resolves to 1 m.
* [x] **Apply the model review checklist to the other environments.** The
      aircraft work produced twelve checks in
      [docs/model-review-checklist.md](docs/model-review-checklist.md), derived
      from real defects rather than from good intentions. All twelve have now
      been run across all eighteen environments.

      The pass so far: check 3 found two write-only state fields a hand review
      had missed; check 9 found the bank-commanded loop gain varies 2.38x on the
      figure-8, and that removing it changes nothing measurable; check 10's own
      advice cleared the integrator in one run. The circle's path-following
      failure turned out not to be a guidance fault at all -- a third of its
      radius range is unflyable at the cruise speed its autopilot holds, and
      trading speed for radius took the worst seed from 1860 m to 73 m, closing
      half of a long-standing strict xfail. Check 12 exists because the first
      version of that fix edited one of *three* copies of the same control law
      and measurably did nothing.

      Checks 5, 7 and 8 each needed a plant-agnostic form to be run at all, and
      each needed its first metric discarded. Check 5 is now done by autodiff:
      comparing the two one-sided Jacobians of a step tells a kink from a steep
      curve, which comparing sample-to-sample steps cannot -- that ranked
      Arrhenius above every real seam. Check 7 became an unforced run, with
      linear growth separated from accelerating growth so that an aircraft is
      not flagged for flying forwards. Both come back clean: no plant produces
      energy from its own equations, and the only seams that survive refinement
      are a mass clamped at zero, a power limit binding, and one in the aircraft
      at 308 m/s that full actuator travel cannot reach.

      Checks 3, 4, 5, 7 and 8 now run in
      [the conformance suite](tests/test_env_conformance.py) against every
      registered environment, each with an allowlist so it reports *new*
      defects rather than restating known-benign ones. A new environment
      inherits them by adding one line to the registry.
* [x] **A reward-shaping phase.** The rewards had been written per environment as
      each was added, and the conventions had drifted -- Gaussian versus
      quadratic tracking terms, differing crash penalties, differing treatment
      of the target band, and four environments whose reward was *identically
      zero* across the first three halvings of their error. All eighteen now
      share one contract: `(tracking terms, multiplied) x (1 - weighted costs)`,
      bounded in `[0, 1]`, log-scaled around a floor taken from each plant's own
      instrumentation. Costs multiply rather than subtract, so nothing is earned
      without tracking and no episode can profit by ending early -- which made
      the flat crash penalties redundant, and they are gone. See
      [docs/reward-shaping.md](docs/reward-shaping.md).
* [ ] **Move off the Alpha classifier** once the others above are settled.

### Known gaps

The test suite records these rather than hiding them -- five `strict` xfail
cases, from two markers, plus the patrol baseline notes above:

* **Plane Patrol expert quality**: both patrol variants now ship a PID, but it
  completes roughly half of evaluation seeds. The failure is a lateral bank
  oscillation that sets in once the follower overshoots *ahead* of the slot
  chasing a steeply descending lead: pursuit guidance then commands a turn the
  bank loop cannot make, and it rings between its limits. No lateral gain
  combination clears it, so the guidance law needs energy management -- the
  follower cannot shed speed in a descent -- rather than further tuning.
* **Four-tank zero is fixed**: the real apparatus is celebrated for letting you
  move the multivariable zero across the imaginary axis by turning two valves.
  Here `gamma1` and `gamma2` are constants, so only the non-minimum-phase
  configuration is available.
* **CSTR target margin**: the bottom of its sampled band needs the coolant
  within a fraction of a kelvin of its stop. Reachable, but with almost no
  authority left for disturbance rejection.

---

