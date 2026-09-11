"""Actuator-to-output time constants, and the table docs/rl-protocol.md carries.

Run with ``uv run python scripts/measure_time_constants.py``. It measures every
environment and rewrites the generated table in ``docs/rl-protocol.md`` in
place. Re-run it after changing an actuator, a plant's dynamics or an episode
length.

The measurement itself lives in :mod:`target_gym.episode_length`, so this
script and ``tests/test_episode_length.py`` cannot disagree about what a time
constant is. It used to live here alone, and the table drifted: the glass
furnace gained a regenerator, a reversal cycle, a thermocouple lag and a fuel
dead time, its crown response went from 132 steps to 822, and the table still
said 132.
"""

import warnings

warnings.filterwarnings("ignore")

import pathlib  # noqa: E402

from target_gym.episode_length import TAU_MULTIPLE, measure_response  # noqa: E402
from target_gym.registry import REGISTRY  # noqa: E402

BEGIN = "<!-- BEGIN GENERATED TIME CONSTANTS -->"
END = "<!-- END GENERATED TIME CONSTANTS -->"
DOC = pathlib.Path(__file__).resolve().parent.parent / "docs" / "rl-protocol.md"


def rows():
    for name, spec in sorted(REGISTRY.items()):
        r = measure_response(spec)
        n = r["episode"]
        gamma = 1.0 - 1.0 / n
        tau = r["tau_steps"]
        if tau != tau:
            yield f"| `{name}` | {n} | {r['shape']} | n/a | {gamma:.5f} |"
            continue
        need = TAU_MULTIPLE * tau
        flag = "" if n >= need else f" **{n / tau:.1f} tau**"
        yield f"| `{name}` | {n} | {tau:.0f} | {need:.0f}{flag} | {gamma:.5f} |"


def main() -> int:
    table = [
        f"| environment | N | tau (steps) | {TAU_MULTIPLE}*tau required | gamma |",
        "| --- | --- | --- | --- | --- |",
        *rows(),
    ]
    body = "\n".join(table)
    text = DOC.read_text()
    i, j = text.index(BEGIN), text.index(END)
    DOC.write_text(text[: i + len(BEGIN)] + "\n\n" + body + "\n\n" + text[j:])
    print(f"wrote {len(table) - 2} rows into {DOC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
