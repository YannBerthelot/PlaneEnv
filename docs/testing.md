# Testing

Two jobs, both required to finish inside ten minutes on a GitHub runner.

| job | when | local wall | what it covers |
| --- | --- | --- | --- |
| fast | every push, Python 3.11-3.14 | 34 s warm / 217 s cold | everything except closed-loop rollouts |
| slow | merges to `main`, one interpreter | 42 s | PID beats the best constant action, states stay physical under extreme actions, regimes join |

Both were far outside that budget until the two changes below, and both changes
came from profiling rather than from trimming assertions -- coverage is
unchanged at 90.7%.

## The expensive measurement is recorded, not reproduced

Checking that each shipped MPC really is an upper bound on its PID meant rolling
sixteen controllers out on five seeds: about forty minutes of CPU, and
`[plane]` alone was **836 s**. Because `pytest-xdist` distributes across tests
and not within one, that single test set roughly 70% of the slow job's
wall-clock floor and came close to its 30-minute timeout on CI's slower cores.

It is now recorded by hand into `data/baseline_returns.json` and asserted from
there, guarded by a fingerprint that refuses a record whose code has moved. See
[Baselines](baselines.md) for the design and the reasoning behind fingerprinting
source rather than behaviour.

```bash
make baselines           # re-measure everything, ~40 min
make baselines-plane     # or just one environment
```

Run it when a test tells you a record is stale, and commit the result with the
change that invalidated it.

## XLA compilations are cached, in the process and between runs

What remained was not stepping the plants but *compiling* them. Reverse-mode
through an RK4 aircraft builds a large graph, and it showed: a tuner smoke test
that takes **two** gradient steps still cost 45 s, and shrinking its problem from
36 target pairs and 2000 steps to 2 and 100 barely moved it. The graph was the
cost, not the rollout.

JAX can persist compiled executables to disk, keyed on the computation itself.
`tests/conftest.py` points it at `.jax_cache`, which does two things at once:
under xdist it stops every worker compiling the same graph, and in CI
`actions/cache` restores it between runs, so compilation becomes a
once-per-code-change cost rather than once-per-push.

Measured on the whole fast suite: **217 s cold, 34.5 s warm**. On one tuner test
alone, 44.9 s against 3.4 s. The cache is about 51 MB.

A stale entry is never wrongly reused -- entries are keyed by the jaxpr, the
backend and the JAX version, so a changed computation simply misses. The CI key
includes a hash of `src/`, with a looser `restore-keys` prefix so a code change
still seeds from the previous cache instead of starting empty.

## Conventions

- `-m "not slow"` is the default; mark anything that rolls a plant out for
  hundreds of steps as `slow`.
- Coverage is gated at 89% and measured on the fast subset only, so moving a
  test to `slow` removes it from the gate. Check the number before doing it.
- `strict=True` on every xfail. A known defect that gets fixed must fail the
  suite until its annotation is updated, so the record cannot quietly rot.
