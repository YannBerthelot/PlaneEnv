"""The 2D aircraft under ``jax.vmap``, batched over seeds.

These were smoke tests: they called ``vmap`` and asserted nothing, so they
passed as long as no exception escaped. That is worth very little -- a batched
reset returning the *same* state for every seed, or a step returning NaN, or a
leading axis silently collapsed, would all have gone through. What is asserted
now is what the batching is actually for: one row per seed, rows that differ,
and finite values.

``tests/test_env_conformance.py`` covers vmap generically for every registered
environment. This file stays for what the generic version cannot reach: the
2D aircraft accepts its action as a *tuple* of ``(power, stick)`` rather than a
stacked array, which is a distinct code path through the batching rules.
"""

import jax
import jax.numpy as jnp
import numpy as np

from target_gym.plane.env_jax import Airplane2D

N = 3


def test_reset_batches_over_seeds():
    env = Airplane2D()
    keys = jax.random.split(jax.random.PRNGKey(42), num=N)

    obs, state = jax.vmap(env.reset, in_axes=0)(keys)

    assert obs.shape[0] == N, f"expected one row per seed, got {obs.shape}"
    assert np.isfinite(np.asarray(obs)).all()
    for leaf in jax.tree_util.tree_leaves(state):
        assert np.asarray(leaf).shape[0] == N

    # Distinct seeds must give distinct initial conditions, or the batch is
    # measuring one episode N times.
    assert not np.allclose(np.asarray(obs)[0], np.asarray(obs)[1])


def test_step_batches_over_seeds_with_a_tuple_action():
    env = Airplane2D()
    keys = jax.random.split(jax.random.PRNGKey(42), num=N)
    obs, state = jax.vmap(env.reset, in_axes=0)(key=keys)

    action = (jnp.ones(N), jnp.zeros(N))
    next_obs, next_state, reward, terminated, truncated, _ = jax.vmap(
        env.step, in_axes=0
    )(key=keys, state=state, action=action)

    assert next_obs.shape == obs.shape
    assert reward.shape == (N,)
    assert terminated.shape == (N,) and truncated.shape == (N,)
    assert np.isfinite(np.asarray(next_obs)).all()
    assert np.isfinite(np.asarray(reward)).all()

    # The step has to have done something, and something different per seed.
    assert not np.allclose(np.asarray(next_obs), np.asarray(obs))
    assert not np.allclose(np.asarray(next_obs)[0], np.asarray(next_obs)[1])
    for leaf in jax.tree_util.tree_leaves(next_state):
        assert np.asarray(leaf).shape[0] == N
