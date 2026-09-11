"""The documentation is checked, not merely written.

Two failure modes this guards against, both of which happen quietly:

* ``docs/environments.md`` states shapes and baselines for eighteen
  environments. It is generated from the registry, and drifts the moment
  someone adds an environment or a baseline without regenerating it.
* Code in the docs rots. Every runnable example here is executed, so a renamed
  argument or a changed return arity fails the suite rather than greeting a
  new user.

A fenced block is executed unless its first line is ``# doc: skip`` -- used for
snippets that write files, install packages, or need a GPU.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
GENERATOR = ROOT / "scripts" / "generate_env_reference.py"

BLOCK = re.compile(r"^```python\n(.*?)^```", re.MULTILINE | re.DOTALL)


def _runnable_blocks(path: pathlib.Path) -> list[str]:
    return [
        code
        for code in BLOCK.findall(path.read_text())
        if not code.lstrip().startswith("# doc: skip")
    ]


def _doc_cases() -> list[tuple[str, str]]:
    cases = []
    for md in [ROOT / "README.md", *sorted(DOCS.glob("*.md"))]:
        for i, code in enumerate(_runnable_blocks(md)):
            cases.append(pytest.param(code, id=f"{md.name}:{i}"))
    return cases


def test_environment_reference_is_in_sync_with_the_registry():
    """docs/environments.md must match what the generator produces."""
    result = subprocess.run(
        [sys.executable, str(GENERATOR), "--check"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, (
        f"{result.stdout}{result.stderr}\n"
        "Run: python scripts/generate_env_reference.py"
    )


@pytest.mark.parametrize("code", _doc_cases())
def test_documented_example_runs(code, tmp_path, monkeypatch):
    """Every runnable example in the README and docs/ executes without raising.

    The README is included because it is the highest-traffic surface in the
    project and its quickstart is the first code anyone runs. The first version
    of that quickstart imported a name the package does not export, and nothing
    would have caught it -- this test globbed ``docs/`` only.
    """
    monkeypatch.chdir(tmp_path)
    exec(compile(code, "<doc>", "exec"), {"__name__": "__doc_example__"})


def test_every_environment_has_a_physics_contract():
    """The README claims every environment carries a PHYSICS.md. Keep it true.

    The claim is made in four places and is the project's main quality
    argument, so it is asserted rather than trusted -- a new environment
    without a contract fails here instead of quietly weakening the claim.
    """
    from target_gym.registry import REGISTRY

    missing = []
    for name, spec in REGISTRY.items():
        module = type(spec.make_env()).__module__
        for suffix in (".env_jax", ".marl"):
            module = module.removesuffix(suffix)
        if not (ROOT / "src" / module.replace(".", "/") / "PHYSICS.md").exists():
            missing.append(name)
    assert not missing, f"environments without a PHYSICS.md: {missing}"


# Written-out numbers, because that is how the README says them.
_NUMBER_WORDS = {
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}


def test_readme_states_the_right_number_of_physics_contracts():
    """The README counts the contracts in prose; keep the count true.

    It said "fourteen" for a while after a fifteenth was added. A number in
    prose has no other way of being checked, and this one is load-bearing --
    it is how a reader sizes the physics claim.
    """
    import re

    readme = (ROOT / "README.md").read_text()
    match = re.search(r"covered\*{0,2} by (\w+) contracts", readme)
    assert match, "could not find the 'covered by N contracts' claim in README.md"

    word = match.group(1).lower()
    assert word in _NUMBER_WORDS, f"unrecognised number word {word!r} in the claim"

    actual = len(list((ROOT / "src").rglob("PHYSICS.md")))
    assert _NUMBER_WORDS[word] == actual, (
        f"README says {word} ({_NUMBER_WORDS[word]}) contracts, "
        f"but there are {actual} PHYSICS.md files"
    )


def test_the_colab_notebook_still_runs():
    """Execute the quickstart notebook's code cells.

    The notebook is the first thing a new user runs, and it is the example most
    likely to rot silently: it lives outside `docs/`, so the fenced-block runner
    above never sees it, and nobody notices a broken cell until a reader hits
    it. This session alone changed the plane's observation width and moved the
    quickstart from `step_env` to `step`, either of which would have broken it.

    The install cell is skipped, since the package under test is the one already
    importable here.
    """
    import json

    nb = json.loads((ROOT / "notebooks" / "quickstart.ipynb").read_text())
    sources = [
        "".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code"
    ]
    body = [src for src in sources if not src.lstrip().startswith("!pip")]
    assert body, "no runnable cells found in the notebook"

    import matplotlib

    matplotlib.use("Agg")
    namespace: dict = {"__name__": "__notebook__"}
    for i, src in enumerate(body):
        try:
            exec(compile(src, f"<notebook cell {i}>", "exec"), namespace)
        except Exception as exc:  # pragma: no cover - the failure is the point
            raise AssertionError(
                f"notebooks/quickstart.ipynb cell {i} failed: {exc}\n\n{src}"
            ) from exc


def test_documentation_still_describes_the_code():
    """The contracts are read *instead of* the code, so drift misinforms.

    Reviewing all twenty-one environments found this to be the repository's
    most common defect, and three of that review's own findings were wrong
    because of it: a deviation claiming post-stall lift decays to zero when the
    fix had long been implemented, a docstring saying sub-step rewards are
    summed when the code takes their mean, and a deviation crediting a reward
    change to a band the reward does not read.

    ``scripts/check_doc_drift.py`` catches the four mechanical cases. The
    judgement calls it cannot catch are exactly the ones that rot, so this is a
    floor rather than a guarantee.
    """
    import subprocess
    import sys

    for script in ("scripts/generate_physics_facts.py", "scripts/check_doc_drift.py"):
        result = subprocess.run(
            [sys.executable, script, "--check"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, script + "\n" + result.stdout + result.stderr
