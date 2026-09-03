"""Fingerprints that say whether a recorded measurement still describes the code.

Measuring how well the shipped controllers actually *control* costs about forty
minutes of CPU, because a single aircraft MPC step optimises a 90-step rollout
fifty times over. Paying that on every merge is wasteful: the answer only moves
when the physics, the controllers or their gains move, and most merges touch
none of the three.

So the measurement is recorded to ``data/baseline_returns.json`` by
``scripts/record_baselines.py`` and CI compares the recorded numbers instead of
reproducing them. That trade is only safe if a stale record cannot pass
silently, which is what this module is for: each record carries a fingerprint of
everything that determines it, and the test refuses to read a record whose
fingerprint no longer matches the tree.

Why the fingerprint is taken over *source* rather than over behaviour
--------------------------------------------------------------------
The obvious alternative -- roll the plant forward a few steps and hash the
trajectory -- is more direct, and it does not survive the matrix. Those numbers
are float32 results of ``exp``, ``sin`` and ``tanh``, whose last bits are not
guaranteed identical between macOS arm64 (where a maintainer regenerates) and
Linux x86 (where CI checks), nor across libm versions. A fingerprint that
disagrees with itself across platforms would fail every CI run and teach people
to regenerate on red rather than on change, which is worse than no check.

Hashing the source has the opposite failure mode, and it is the safe one: it can
report staleness that is not real (a rename invalidates a record the behaviour
would have kept), never freshness that is not real. The cost of a false stale is
one regeneration command; the cost of a false fresh is CI certifying a claim
about code that no longer exists.

Comments and docstrings are stripped before hashing, because this repository
edits prose constantly and none of it changes a number.
"""

from __future__ import annotations

import ast
import hashlib
import json
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parent
_REPO = _ROOT.parent.parent
GAINS_PATH = _REPO / "data" / "pid_gains.json"
BASELINES_PATH = _REPO / "data" / "baseline_returns.json"

# Shared controller code. A change here can move any environment's numbers, so
# it belongs in every environment's fingerprint.
_SHARED_SOURCES = (
    _ROOT / "experts" / "pid.py",
    _ROOT / "experts" / "mpc.py",
    _ROOT / "utils.py",
    _ROOT / "integration.py",
)


def _stable_ast_digest(source: str) -> str:
    """Hash a module's structure, ignoring comments, docstrings and formatting.

    ``ast.dump`` is not used: its output gained fields between Python versions
    (``type_params`` in 3.12, for one), so the same file would fingerprint
    differently across the CI matrix. This walks the tree and emits only node
    type names and the handful of value-bearing fields, which have been stable
    across every version this package supports.
    """
    tree = ast.parse(source)
    out: list[str] = []

    def emit(node: ast.AST) -> None:
        out.append(type(node).__name__)
        if isinstance(node, ast.Constant):
            out.append(repr(node.value))
        elif isinstance(node, ast.Name):
            out.append(node.id)
        elif isinstance(node, ast.Attribute):
            out.append(node.attr)
        elif isinstance(node, ast.arg):
            out.append(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.append(node.name)

        body = getattr(node, "body", None)
        for field, value in ast.iter_fields(node):
            items = value if isinstance(value, list) else [value]
            for i, child in enumerate(items):
                if not isinstance(child, ast.AST):
                    continue
                # Drop docstrings: the leading bare string of a module, class or
                # function body.
                if (
                    field == "body"
                    and i == 0
                    and body is not None
                    and isinstance(child, ast.Expr)
                    and isinstance(getattr(child, "value", None), ast.Constant)
                    and isinstance(child.value.value, str)
                    and isinstance(
                        node,
                        (
                            ast.Module,
                            ast.ClassDef,
                            ast.FunctionDef,
                            ast.AsyncFunctionDef,
                        ),
                    )
                ):
                    continue
                emit(child)

    emit(tree)
    return hashlib.sha256("\x00".join(out).encode()).hexdigest()


def _digest_paths(paths) -> str:
    h = hashlib.sha256()
    for path in sorted(paths, key=str):
        if not path.exists():
            continue
        h.update(path.name.encode())
        h.update(_stable_ast_digest(path.read_text()).encode())
    return h.hexdigest()


def _env_sources(spec) -> list[pathlib.Path]:
    """Every Python module in the package the environment is defined in."""
    module = type(spec.make_env()).__module__
    package = module.rsplit(".", 1)[0]
    directory = _ROOT.parent / pathlib.Path(*package.split("."))
    if not directory.is_dir():
        return []
    return [p for p in sorted(directory.glob("*.py")) if p.name != "rendering.py"]


def baseline_fingerprint(spec) -> str:
    """Everything that determines this environment's recorded baseline scores.

    Four inputs: the environment's own modules, the shared controller and
    integration code, the tuned gains, and the parameter values the measurement
    is taken at. Rendering is excluded -- it cannot change a return.
    """
    params = spec.make_test_params()
    values = {
        k: repr(v)
        for k, v in sorted(vars(params).items())
        if isinstance(v, (int, float, bool, str, tuple))
    }

    gains = {}
    if GAINS_PATH.exists():
        allgains = json.loads(GAINS_PATH.read_text())
        # A controller may read a differently-named key (the 2D aircraft's
        # autopilot reads "plane_cascaded"), so take every key that starts with
        # the environment's name rather than only the exact match.
        gains = {k: v for k, v in allgains.items() if k.startswith(spec.name)}

    h = hashlib.sha256()
    h.update(_digest_paths(_env_sources(spec)).encode())
    h.update(_digest_paths(_SHARED_SOURCES).encode())
    h.update(json.dumps(values, sort_keys=True).encode())
    h.update(json.dumps(gains, sort_keys=True).encode())
    return h.hexdigest()[:16]


def load_recorded_baselines() -> dict:
    """The recorded measurements, or an empty mapping if none exist yet."""
    if not BASELINES_PATH.exists():
        return {}
    return json.loads(BASELINES_PATH.read_text())
