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
import datetime as _dt
import json
import sys
import time

import numpy as np

from target_gym.provenance import BASELINES_PATH, baseline_fingerprint
from target_gym.registry import REGISTRY
from target_gym.runners.runners import mpc_policy, pid_policy, rollout

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


def record(name: str) -> dict | None:
    spec = REGISTRY[name]
    if not spec.has_pid or spec.make_mpc is None:
        return None

    params = spec.make_test_params()
    horizon = int(params.max_steps_in_episode)
    env = spec.make_env()

    pid, mpc, terminated_early = [], [], 0
    t0 = time.time()
    for seed in range(SEEDS):
        _, _, r = rollout(spec, params, pid_policy(spec), seed)
        pid.append(float(np.sum(r)))
        _, _, r = rollout(spec, params, mpc_policy(spec, env, params), seed)
        mpc.append(float(np.sum(r)))
        terminated_early += int(len(r) < horizon)

    return {
        "fingerprint": baseline_fingerprint(spec),
        "steps": horizon,
        "seeds": SEEDS,
        "pid_returns": [round(v, 6) for v in pid],
        "mpc_returns": [round(v, 6) for v in mpc],
        "mpc_terminated_early": terminated_early,
        "seconds": round(time.time() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--envs", nargs="*", default=None, help="default: all with an MPC")
    args = ap.parse_args()

    names = args.envs or [
        n for n, s in REGISTRY.items() if s.has_pid and s.make_mpc is not None
    ]
    # Only this run's rows are collected here; the file is re-read at write
    # time and these are merged onto whatever it holds *then*. A run takes
    # tens of minutes, and reading the file at startup meant a second run --
    # a forgotten `--envs` job left going in another terminal -- would write
    # its hour-old snapshot over every row recorded in the meantime, silently
    # reverting them. That happened here: a stale two-environment job from an
    # earlier session was still running, 1h49m in, and would have reverted a
    # full twenty-environment re-record on finishing.
    rows: dict[str, dict] = {}
    for name in names:
        row = record(name)
        if row is None:
            print(f"  {name:20s} no MPC baseline, skipped", flush=True)
            continue
        rows[name] = row
        p, m = np.mean(row["pid_returns"]), np.mean(row["mpc_returns"])
        verdict = "MPC leads" if m >= p else f"PID leads by {p - m:.1f}"
        print(
            f"  {name:20s} PID {p:9.2f}  MPC {m:9.2f}  {verdict:22s} "
            f"term {row['mpc_terminated_early']}  {row['seconds']:6.0f}s",
            flush=True,
        )

    out = {}
    if BASELINES_PATH.exists():
        out = json.loads(BASELINES_PATH.read_text())
    out.update(rows)
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
    print(f"\n  wrote {BASELINES_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
