"""Generate one documentation page per environment, from the registry.

Modelled on the Gymnasium environment pages, which are the convention readers
already know: a picture, then action space, observation space, rewards, starting
state, episode end, arguments, in that order.

Generated rather than written, for the same reason ``docs/environments.md`` is:
eighteen hand-maintained pages drift the moment a parameter changes, and a
documentation page that quietly disagrees with the code is worse than none. Run

    uv run python scripts/generate_env_pages.py            # rewrite
    uv run python scripts/generate_env_pages.py --check    # CI: fail if stale

Everything on a page is read from the environment itself: spaces from
``observation_space``/``action_space``, tracked variables from
``obs_value_index``, rewards and termination from the docstrings of
``compute_reward`` and ``check_is_terminal``, arguments from the params
dataclass, baseline scores from ``data/baseline_returns.json``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")

import jax  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from target_gym.registry import REGISTRY  # noqa: E402

OUT_DIR = ROOT / "docs" / "environments"
BASELINES = ROOT / "data" / "baseline_returns.json"

# Where each environment's gif lives, relative to the repository root. Only
# environments with a rendered clip get a picture; the rest simply omit it.
VIDEO_CANDIDATES = (
    "videos/{name}/pid_output_short.gif",
    "videos/{name}/pid_output.gif",
)
# ``target_gym.runners.runners`` writes videos/<env>/pid_output.gif, so that is
# the path a page should use. This map is only for environments whose clip is
# shared with another -- the two patrol variants render the same formation --
# and it deliberately no longer points at videos/plane3d/*_short.gif, which were
# hand-placed leftovers of an older layout that no generator refreshes. Those
# went stale silently: the gallery showed pre-re-skin aircraft for a week.
SPECIAL_VIDEOS = {
    "patrol_bearing_only": "videos/patrol/pid_output.gif",
}


def _video(name: str) -> str | None:
    for candidate in (
        SPECIAL_VIDEOS.get(name),
        *(c.format(name=name) for c in VIDEO_CANDIDATES),
    ):
        if candidate and (ROOT / candidate).exists():
            return candidate
    return None


def _first_paragraph(doc: str | None) -> str:
    if not doc:
        return ""
    out: list[str] = []
    for line in doc.strip().splitlines():
        if not line.strip() and out:
            break
        out.append(line.strip())
    return " ".join(out)


def _env_module(spec):
    package = type(spec.make_env()).__module__.rsplit(".", 1)[0]
    try:
        return importlib.import_module(package + ".env")
    except ModuleNotFoundError:
        return None


def _action_labels(env, n: int) -> list[str]:
    """Action meanings, where the environment's own docstring states them.

    Several class docstrings carry a line like
    ``Action  (2): [fuel, feedwater], raw in [-1, 1]``. That is the only place
    the meanings are written down, so it is the only honest source for them --
    inventing labels here would put words in the environment's mouth.
    """
    doc = type(env).__doc__ or ""
    for line in doc.splitlines():
        if "ction" not in line or "[" not in line:
            continue
        start = line.index("[")
        stop = line.find("]", start)
        if stop < 0:
            continue
        inside = line[start + 1 : stop]
        parts = [x.strip() for x in inside.split(",") if x.strip()]
        if len(parts) == n:
            return parts
    return []


def _space_table(space, labels: list[str]) -> str:
    low = np.broadcast_to(np.asarray(space.low, float), space.shape or (1,))
    high = np.broadcast_to(np.asarray(space.high, float), space.shape or (1,))
    rows = ["| # | meaning | min | max |", "|---|---|---|---|"]
    for i, (lo, hi) in enumerate(zip(low, high)):
        rows.append(
            f"| {i} | {labels[i] if i < len(labels) else ''} | {lo:g} | {hi:g} |"
        )
    return "\n".join(rows)


def _params_table(params, limit: int = 14) -> str:
    fields = [
        (k, v)
        for k, v in vars(params).items()
        if isinstance(v, (int, float, bool)) and not isinstance(v, bool)
    ]
    rows = ["| parameter | default |", "|---|---|"]
    for k, v in fields[:limit]:
        rows.append(
            f"| `{k}` | {v:g} |" if isinstance(v, float) else f"| `{k}` | {v} |"
        )
    if len(fields) > limit:
        rows.append(f"| … | {len(fields) - limit} more, see the params dataclass |")
    return "\n".join(rows)


def _baseline_section(name: str, spec) -> str:
    recorded = {}
    if BASELINES.exists():
        recorded = json.loads(BASELINES.read_text()).get(name, {})
    lines = []
    if not spec.has_pid:
        return f"No baseline ships for this environment. {spec.baselines_note or ''}".strip()
    if recorded.get("pid_returns"):
        pid = np.mean(recorded["pid_returns"])
        steps = recorded["steps"]
        lines.append(
            f"Measured over {recorded['seeds']} seeds on a {steps}-step episode "
            f"(see [Baselines](../baselines.md)):\n"
        )
        lines.append("| controller | return | per step |")
        lines.append("|---|---|---|")
        lines.append(f"| PID | {pid:.1f} | {pid / steps:.3f} |")
        if recorded.get("mpc_returns"):
            mpc = np.mean(recorded["mpc_returns"])
            lines.append(f"| MPC | {mpc:.1f} | {mpc / steps:.3f} |")
    else:
        lines.append("A tuned PID ships with this environment.")
    if spec.mpc_degraded:
        lines.append(
            f'\n!!! warning "The MPC is not an upper bound here"\n'
            f"    {spec.mpc_degraded.split('.')[0]}."
        )
    return "\n".join(lines)


def page(name: str, spec) -> str:
    env = spec.make_env()
    params = spec.make_test_params()
    key = jax.random.PRNGKey(0)
    obs, state = env.reset_env(key, params)
    module = _env_module(spec)

    obs_space = env.observation_space(params)
    act_space = env.action_space(params)
    n_obs = int(np.prod(obs_space.shape or (1,)))
    tracked = env.obs_value_index
    tracked = tracked if isinstance(tracked, tuple) else (tracked,)

    from target_gym.runners.runners import _tracked_labels

    try:
        labels = _tracked_labels(env, len(tracked))
    except Exception:
        labels = []

    title = name.replace("_", " ").title()
    video = _video(name)
    dt = float(getattr(params, "delta_t", 1.0))
    episode = int(params.max_steps_in_episode)

    out = [f"# {title}", ""]
    if video:
        out += [
            f'<p align="center"><img src="../{video}" width="480px"/></p>',
            "",
        ]
    if module and module.__doc__:
        out += [_first_paragraph(module.__doc__), ""]

    out += [
        "| | |",
        "|---|---|",
        f"| Action space | `Box({act_space.shape or (1,)})`, all actions in [-1, 1] |",
        f"| Observation space | `Box({obs_space.shape or (1,)})` |",
        f"| Tracked variable(s) | {', '.join(labels) if labels else 'see below'} |",
        f"| Episode length | {episode} steps ({episode * dt:g} s at dt = {dt:g} s) |",
        f"| Import | `from target_gym import {type(env).__name__}, "
        f"{type(params).__name__}` |",
        "",
        "## Action space",
        "",
        "Actions are normalised to `[-1, 1]` and mapped onto the plant's real",
        "actuator range inside the environment.",
        "",
        _space_table(
            act_space, _action_labels(env, act_space.shape[0] if act_space.shape else 1)
        ),
        "",
        "## Observation space",
        "",
        f"{n_obs} values. Indices {tracked} carry the tracked variable(s) that the",
        "reward scores.",
        "",
        "## Rewards",
        "",
    ]
    reward_doc = _first_paragraph(
        getattr(module, "compute_reward", None).__doc__
        if module and hasattr(module, "compute_reward")
        else None
    )
    out += [
        reward_doc or "See the environment's `compute_reward`.",
        "",
        "Every environment in this suite scores on one contract: the reward is",
        "`(tracking terms, multiplied) x (1 - weighted costs)`, bounded in",
        "`[0, 1]`, and reaches 1 only while the target is held exactly. See",
        "[Reward shaping](../reward-shaping.md).",
        "",
        "## Starting state",
        "",
        f"`reset` samples the initial condition and the target; state has "
        f"{len(state.__dataclass_fields__)} fields.",
        "",
        "## Episode end",
        "",
    ]
    term_doc = _first_paragraph(
        getattr(module, "check_is_terminal", None).__doc__
        if module and hasattr(module, "check_is_terminal")
        else None
    )
    out += [
        f"**Termination.** {term_doc or 'See `check_is_terminal`.'}",
        "",
        f"**Truncation.** After {episode} steps.",
        "",
        "## Baselines",
        "",
        _baseline_section(name, spec),
        "",
        "## Arguments",
        "",
        _params_table(params),
        "",
    ]
    return "\n".join(out) + "\n"


def build() -> dict[str, str]:
    return {name: page(name, spec) for name, spec in REGISTRY.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="fail if any page is stale")
    args = ap.parse_args()

    pages = build()
    if args.check:
        stale = []
        for name, content in pages.items():
            path = OUT_DIR / f"{name}.md"
            if not path.exists() or path.read_text() != content:
                stale.append(name)
        if stale:
            print(f"stale environment pages: {stale}; run {__file__}")
            return 1
        print(f"{len(pages)} environment pages are up to date")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, content in pages.items():
        (OUT_DIR / f"{name}.md").write_text(content)
    print(f"wrote {len(pages)} pages to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
