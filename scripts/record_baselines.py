"""Record how well the shipped controllers actually control, for CI to check.

Why this is a script and not a test
-----------------------------------
Measuring it costs about forty minutes of CPU, nearly all of it in four
environments: a single 2D-aircraft MPC step optimises a 90-step rollout fifty
times over, so one parametrisation took 836 s of a 19:47 slow job. Because
pytest-xdist parallelises across tests and not within one, that single test set
about 70% of the job's wall-clock floor, and on GitHub's slower four-core
runners it approached the job's 30-minute timeout on its own.

The answer only changes when the physics, the controllers or their gains change,
which most merges do not touch. So the measurement is taken here, by hand, and
committed to ``data/baseline_returns.json``; CI reads the recorded numbers and
checks the contract against them.

That is only safe if a stale record cannot pass silently. Each record carries a
fingerprint of the environment's modules, the shared controller and integration
code, the tuned gains and the parameter values, so a record that no longer
describes the tree is refused rather than believed. See
``target_gym.provenance``.

Usage
-----
    uv run python scripts/record_baselines.py              # everything
    uv run python scripts/record_baselines.py --envs plane cstr

Run it whenever the test suite tells you a record is stale, and commit the
result alongside the change that invalidated it.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import contextlib
import datetime as _dt
import json
import multiprocessing as mp
import os
import signal
import statistics
import sys
import threading
import time

import numpy as np

from target_gym.provenance import BASELINES_PATH, baseline_fingerprint
from target_gym.registry import REGISTRY
from target_gym.runners.runners import (
    mpc_policy,
    pid_policy,
    rollout,
    rollout_mpc_batch,
)

# Ten seeds, because two mislead and five is a tripwire rather than a
# measurement. During the work these baselines came from, an MPC scored 277 on
# seed 0 and 65 on seed 1 -- an average of the two looks healthy where the
# ten-seed truth was -61.
#
# Ten rather than five because this file is now the *only* place these numbers
# are measured. docs/baselines.md used to publish a separate ten-seed table
# alongside a five-seed contract, which is two measurements of one quantity and
# a guarantee they eventually disagree. The published table is generated from
# this artifact instead, so the contract and the claim cannot drift apart.
#
# Episodes come from each environment's own EnvSpec.test_params, which since the
# episode-length audit satisfy N >= max(10 * tau_actuator, 3 * T_period) -- long
# enough that holding the target, not reaching it, is what is being scored.
SEEDS = 10


@contextlib.contextmanager
def _running(live, key):
    """Announce this job while it runs, so the monitor can see it in flight.

    A pool hides the difference between queued and slow. Without this the only
    signal a job exists is its result, which is exactly how a glass-furnace
    seed ran for seventy minutes against a three-minute median with nothing on
    screen to say so.
    """
    if live is not None:
        live[key] = time.time()
    try:
        yield
    finally:
        if live is not None:
            live.pop(key, None)


#: How often the monitor prints, and the multiple of an environment's own
#: median seed time past which a seed in flight is called out by name.
PROGRESS_SECONDS = 30
STRAGGLER_FACTOR = 4.0


def _hms(seconds: float) -> str:
    m, sec = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{sec:02d}s"


def _by_cost(names: list[str]) -> list[str]:
    """Most expensive first, from what the previous record says each one cost.

    Longest-processing-time-first is the standard greedy schedule and it is
    worth the two lines: the seeds that set the wall-clock have to start in the
    first wave, not the last. An environment with no recorded time sorts first,
    because an unknown cost is more likely to be a new expensive one than a new
    cheap one, and being wrong about that costs nothing.
    """
    previous = {}
    if BASELINES_PATH.exists():
        try:
            previous = json.loads(BASELINES_PATH.read_text())
        except json.JSONDecodeError:
            previous = {}
    return sorted(
        names,
        key=lambda n: -float(previous.get(n, {}).get("seconds", float("inf"))),
    )


def _monitor(inflight, durations, rows, total, stop) -> None:
    """Print what is running, every ``PROGRESS_SECONDS``, until told to stop.

    A pool reports results and nothing else, so between one environment
    finishing and the next there is silence, and silence looks the same whether
    the run is healthy or wedged. It was not a cosmetic problem: a furnace seed
    ran seventy minutes against a three-and-a-half-minute median and the only
    way to find out was to go looking.

    So the summary line is unconditional, and any seed running past
    ``STRAGGLER_FACTOR`` times the median already measured for its own
    environment gets named on its own line. Its own environment, because the
    seeds differ across the suite by four orders of magnitude and a global
    median would flag every furnace seed and no reactor one.
    """
    t0 = time.time()
    while not stop.wait(PROGRESS_SECONDS):
        now = time.time()
        try:
            running = sorted(((now - t, k) for k, t in inflight.items()), reverse=True)
        except Exception:  # the manager is gone; the run is finishing
            return
        done = sum(len(v) for v in durations.values())
        print(
            f"  [{_hms(now - t0):>7s}] {len(rows):2d}/{total} envs, "
            f"{done:3d} seeds done, {len(running):2d} running"
            + (
                f", longest {running[0][1]} at {_hms(running[0][0])}" if running else ""
            ),
            flush=True,
        )
        for elapsed, key in running:
            seen = durations.get(key.rsplit(" seed ", 1)[0], [])
            if len(seen) < 3:
                continue
            median = statistics.median(seen)
            if elapsed > STRAGGLER_FACTOR * median:
                print(
                    f"      !! {key} has run {_hms(elapsed)}, "
                    f"{elapsed / median:.0f}x the {_hms(median)} median for it",
                    flush=True,
                )


def _pid_one_seed(args):
    """One PID episode, for the pool."""
    name, seed, live = args
    t0 = time.time()
    with _running(live, f"{name} PID seed {seed}"):
        spec = REGISTRY[name]
        params = spec.make_test_params()
        value = float(np.sum(rollout(spec, params, pid_policy(spec), seed)[2]))
    return value, time.time() - t0


def _mpc_one_seed(args):
    """One CasADi MPC episode, for the pool. Rebuilds everything per worker."""
    name, seed, live = args
    t0 = time.time()
    with _running(live, f"{name} MPC seed {seed}"):
        spec = REGISTRY[name]
        params = spec.make_test_params()
        env = spec.make_env()
        policy = mpc_policy(spec, env, params)
        _, _, r = rollout(spec, params, policy, seed)
        out = float(np.sum(r)), int(len(r)), policy.controller.solver_report()
    return (*out, time.time() - t0)


def _mpc_batch_one_env(args):
    """All ten seeds of a ``GradientMPC`` environment, vmapped, for the pool.

    A whole environment rather than a seed, because ``rollout_mpc_batch``
    already batches the seeds into one call. It still belongs in the pool: a
    vmapped rollout on CPU is not multicore -- measured at 126-147% of a
    fourteen-core machine -- so running these seven one after another in the
    parent left twelve cores idle for forty minutes while the pool sat empty.
    """
    name, live = args
    t0 = time.time()
    with _running(live, f"{name} MPC batch"):
        spec = REGISTRY[name]
        _, _, r, ended = rollout_mpc_batch(spec, spec.make_test_params(), SEEDS)
        out = [float(v) for v in r.sum(axis=1)], int(ended.sum())
    return (*out, time.time() - t0)


def _merge_reports(reports: list[dict]) -> dict:
    """Pool per-seed solver health into one summary for the environment."""
    calls = sum(r.get("solver_calls", 0) for r in reports)
    if not calls:
        return {}
    weighted = sum(
        r.get("solver_mean_iters", 0.0) * r.get("solver_calls", 0) for r in reports
    )
    return {
        "solver_calls": calls,
        "solver_failures": sum(r.get("solver_failures", 0) for r in reports),
        "solver_capped": sum(r.get("solver_capped", 0) for r in reports),
        "solver_mean_iters": round(weighted / calls, 1),
    }


def _split_by_planner(names: list[str]) -> tuple[list[str], list[str]]:
    """Separate the environments that vmap from the ones that need a process each.

    ``GradientMPC`` is JAX throughout and its ``_optimize`` is a pure function
    of ``(actions_init, state)``, so its ten seeds batch into one call. The
    CasADi planners call IPOPT, which is outside JAX and cannot be vmapped or
    jitted, so their seeds become separate processes. IPOPT is single-threaded,
    so that parallelises cleanly.
    """
    from target_gym.experts.mpc import GradientMPC

    batched, per_seed = [], []
    for name in names:
        spec = REGISTRY[name]
        if not spec.has_pid or spec.make_mpc is None:
            print(f"  {name:20s} no MPC baseline, skipped", flush=True)
            continue
        probe = spec.make_mpc(spec.make_env(), spec.make_test_params())
        (batched if isinstance(probe, GradientMPC) else per_seed).append(name)
    return batched, per_seed


def _row(name, pid, mpc, terminated_early, reports, seconds) -> dict:
    spec = REGISTRY[name]
    row = {
        "fingerprint": baseline_fingerprint(spec),
        "steps": int(spec.make_test_params().max_steps_in_episode),
        "seeds": SEEDS,
        "pid_returns": [round(v, 6) for v in pid],
        "mpc_returns": [round(v, 6) for v in mpc],
        "mpc_terminated_early": int(terminated_early),
        "seconds": round(seconds, 1),
    }
    row.update(_merge_reports(reports))
    return row


def _report(name: str, row: dict) -> None:
    p, m = np.mean(row["pid_returns"]), np.mean(row["mpc_returns"])
    verdict = "MPC leads" if m >= p else f"PID leads by {p - m:.1f}"
    health = ""
    if row.get("solver_calls"):
        bad = row["solver_failures"]
        health = f"  solver {100 * (1 - bad / row['solver_calls']):5.1f}% ok"
        if row["solver_capped"]:
            health += f" ({row['solver_capped']} capped)"
    print(
        f"  {name:20s} PID {p:9.2f}  MPC {m:9.2f}  {verdict:22s} "
        f"term {row['mpc_terminated_early']}  {row['seconds']:6.0f}s{health}",
        flush=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--envs", nargs="*", default=None, help="default: all with an MPC")
    args = ap.parse_args()

    names = args.envs or [
        n for n, s in REGISTRY.items() if s.has_pid and s.make_mpc is not None
    ]
    batched, per_seed = _split_by_planner(names)
    live = batched + per_seed

    # Only this run's rows are collected here; the file is re-read at write
    # time and these are merged onto whatever it holds *then*. A run takes
    # tens of minutes, and reading the file at startup meant a second run --
    # a forgotten `--envs` job left going in another terminal -- would write
    # its hour-old snapshot over every row recorded in the meantime, silently
    # reverting them. That happened here: a stale two-environment job from an
    # earlier session was still running, 1h49m in, and would have reverted a
    # full twenty-environment re-record on finishing.
    rows: dict[str, dict] = {}
    started = {n: time.time() for n in live}
    pid_out: dict[str, list] = {n: [None] * SEEDS for n in live}
    mpc_out: dict[str, list] = {n: [None] * SEEDS for n in per_seed}
    batch_out: dict[str, tuple] = {}

    def emit(name: str) -> None:
        """Write the row for *name* once every one of its seeds is in.

        Checkpointing after each environment rather than at the end. A run that
        died or was interrupted used to lose everything, because the file was
        written once after the loop. That cost a full re-record twice: once
        when a serial run was killed to parallelise it, and once when a single
        pathological glass-furnace seed ran twenty times its siblings and there
        was no way to stop waiting without discarding twelve finished
        environments. Writing as we go makes an interrupted run resumable with
        ``--envs`` for whatever is left.
        """
        if name in rows or any(v is None for v in pid_out[name]):
            return
        if name in per_seed:
            if any(v is None for v in mpc_out[name]):
                return
            mpc = [v for v, _, _ in mpc_out[name]]
            ended = sum(1 for _, n, _ in mpc_out[name] if n < rows_steps[name])
            reports = [r for _, _, r in mpc_out[name]]
        else:
            if name not in batch_out:
                return
            mpc, ended = batch_out[name]
            reports = []
        rows[name] = _row(
            name, pid_out[name], mpc, ended, reports, time.time() - started[name]
        )
        _write(rows)
        _report(name, rows[name])

    rows_steps = {
        n: int(REGISTRY[n].make_test_params().max_steps_in_episode) for n in live
    }

    # Every core to the pool. The parent used to run the vmapped batches
    # itself and needed one held back; now it only submits, drains and writes.
    workers = max(1, os.cpu_count() or 2)

    # Reap our own pool if we are killed.
    #
    # ProcessPoolExecutor cleans up when the ``with`` block exits normally, and
    # not at all when the parent is signalled: the workers are orphaned and
    # keep running. Killing two interrupted runs left fifty of them alive for
    # two and a half hours, one holding 48 minutes of CPU, and they are
    # invisible to ``pkill -f record_baselines`` because a spawned child's
    # command line says ``spawn_main``. The next run then looked wedged for six
    # minutes with nothing in flight, because its workers could not get a core.
    def _reap(signum, _frame):
        for child in mp.active_children():
            child.terminate()
        sys.exit(128 + signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _reap)

    manager = mp.Manager()
    inflight = manager.dict()
    durations: dict[str, list[float]] = {}
    stop = threading.Event()

    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {}
        # Expensive environments first, and each one's MPC and PID submitted
        # together.
        #
        # Together, because an environment is only checkpointed once both are
        # in. Queueing every MPC ahead of every PID put the long pole first but
        # meant nothing could be written until the *last* MPC seed finished,
        # which is the reporting blackout this whole run is meant to end.
        #
        # Expensive first, ordered by what the previous record says each one
        # cost, so the seeds that set the wall-clock start in the first wave
        # instead of the last. Self-tuning: the file it reads is the one this
        # script writes.
        # Batched environments first: they are the longest single jobs in the
        # suite and each occupies one worker for its whole duration, so they
        # have to start in the first wave or they set the wall-clock alone.
        # Their PID seeds follow immediately, so those environments can be
        # checkpointed as soon as the batch lands.
        for name in _by_cost(batched):
            futures[pool.submit(_mpc_batch_one_env, (name, inflight))] = (
                "batch",
                name,
                -1,
            )
        for name in batched:
            for seed in range(SEEDS):
                futures[pool.submit(_pid_one_seed, (name, seed, inflight))] = (
                    "pid",
                    name,
                    seed,
                )
        for name in _by_cost(per_seed):
            for seed in range(SEEDS):
                futures[pool.submit(_mpc_one_seed, (name, seed, inflight))] = (
                    "mpc",
                    name,
                    seed,
                )
            for seed in range(SEEDS):
                futures[pool.submit(_pid_one_seed, (name, seed, inflight))] = (
                    "pid",
                    name,
                    seed,
                )
        watcher = threading.Thread(
            target=_monitor,
            args=(inflight, durations, rows, len(live), stop),
            daemon=True,
        )
        watcher.start()

        for fut in cf.as_completed(futures):
            kind, name, seed = futures[fut]
            result = fut.result()
            if kind == "pid":
                pid_out[name][seed], seconds = result
            elif kind == "batch":
                *core, seconds = result
                batch_out[name] = tuple(core)
            else:
                *core, seconds = result
                mpc_out[name][seed] = tuple(core)
            durations.setdefault(f"{name} {kind.upper()}", []).append(seconds)
            emit(name)

    stop.set()

    _write(rows)
    print(f"\n  wrote {BASELINES_PATH}")
    return 0


def _write(rows: dict[str, dict]) -> None:
    """Merge *rows* onto the file as it stands and write it back.

    Re-read at write time rather than held from startup, so a second job
    running concurrently cannot revert rows recorded in the meantime.
    """
    out = {}
    if BASELINES_PATH.exists():
        out = json.loads(BASELINES_PATH.read_text())
    out.update(rows)

    # Drop rows for environments the registry no longer has. Without this a
    # removed environment leaves a fossil for ever: the merge above preserves
    # whatever the file already held, which is what makes a partial re-record
    # safe, and is exactly why nothing ever cleans up. ``plane_steps`` sat in
    # here after being absorbed into ``plane_energy``, describing a task that
    # cannot be constructed.
    dead = [k for k in out if k != "_meta" and k not in REGISTRY]
    for k in dead:
        del out[k]
        print(f"  dropped {k}: no longer in the registry", flush=True)

    out["_meta"] = {
        "generated": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "seeds": SEEDS,
        "note": (
            "Regenerate with scripts/record_baselines.py when a test reports a "
            "stale fingerprint. Commit the result with the change that "
            "invalidated it."
        ),
    }
    BASELINES_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINES_PATH.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    sys.exit(main())
