"""What the package promises at its top level.

Written after the README's own quickstart failed to import. It reached for
``Airplane2D`` -- which is what ``type(env).__name__`` reports -- while the
package exported that class only as ``Plane``. Nothing caught it, because
nothing asserted anything about the public surface at all.

These are cheap, and each one pins a mistake that had actually been made.
"""

from __future__ import annotations

import types

import pytest

import target_gym
from target_gym.registry import REGISTRY


def test_everything_in_all_resolves():
    missing = [n for n in target_gym.__all__ if not hasattr(target_gym, n)]
    assert not missing, f"__all__ names that do not exist: {missing}"


def test_no_public_name_is_missing_from_all():
    """A name without a leading underscore is a promise, listed or not."""
    listed = set(target_gym.__all__)
    leaked = [
        n
        for n in dir(target_gym)
        if not n.startswith("_")
        and n not in listed
        and not isinstance(getattr(target_gym, n), types.ModuleType)
    ]
    assert not leaked, (
        f"public attributes absent from __all__: {leaked}. Either add them, or "
        "import them under a private alias -- `from importlib.metadata import "
        "version` once put `target_gym.version` in the public namespace, where "
        "it shadowed the meaning of the name it borrowed."
    )


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_environment_class_is_importable_under_its_own_name(name):
    """``type(env).__name__`` must be importable from the package root.

    The first thing anyone does with an unfamiliar object is print its type,
    and the second is import it. If those disagree the user gets an
    ImportError while reading the name the library told them. Friendly aliases
    are welcome; they just cannot be the *only* way in.
    """
    cls = type(REGISTRY[name].make_env()).__name__
    assert hasattr(target_gym, cls), (
        f"{name}: environments report themselves as {cls!r} but that name is "
        f"not importable from target_gym"
    )


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_params_class_is_importable_under_its_own_name(name):
    cls = REGISTRY[name].params_cls().__class__.__name__
    assert hasattr(
        target_gym, cls
    ), f"{name}: parameters class {cls!r} is not importable from target_gym"


def test_version_is_a_string():
    """``__version__``, not ``version`` -- and a string, not the function that
    produced it."""
    assert isinstance(target_gym.__version__, str)
    assert target_gym.__version__
