"""Is an episode long enough to measure the task it claims to measure?

``docs/rl-protocol.md`` states the rule::

    N >= max(10 * tau_actuator, 3 * T_period)

and six episodes were lengthened to satisfy it during the episode-length audit.
Nothing enforced it afterwards, so the other fifteen kept whatever the audit did
not look at, and the furnace drifted below it when its physics changed: adding
the regenerator, the reversal cycle, the thermocouple lag and the fuel dead time
took its crown response from 132 steps to 822, which turned a documented 12.1
tau episode into 1.9 without a single number in the repository moving.

``tau_actuator`` is the open-loop step response from actuator to tracked output,
the quantity relay tuning assumes, rather than the slowest mode in the state
matrix. For the 2D aircraft that mode is fuel burn at 29 694 s and for the
reactor it is xenon at 61 908 s, neither of which says how long the plant takes
to answer the stick or the rod.

Plants whose output integrates the input have no steady state to settle to, so
``tau_63`` is not defined for them and the rule's first clause does not apply.
They are reported as ``integrating`` and left to the period clause.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

#: Time constants that must fit inside an episode, from the protocol.
TAU_MULTIPLE = 10


def measure_response(spec, params=None) -> dict:
    """Open-loop step response of the tracked output to the actuator.

    Returns ``{"tau_steps", "shape", "episode"}``. ``tau_steps`` is ``nan`` for
    an integrating plant, whose ``shape`` is ``"integrating"``.

    Driven differentially, stepping the actuator to -0.4 and +0.4 and
    subtracting, so a plant that drifts under its own dynamics does not have
    that drift counted as a response.
    """
    from target_gym.runners.runners import _as_tuple

    if params is None:
        params = spec.make_test_params()
    env = spec.make_env()
    dt = float(getattr(params, "delta_t", 1.0))
    idx = list(_as_tuple(env.obs_value_index))[:1]
    step = jax.jit(env.step_env)
    adim = int(
        np.atleast_1d(env.action_space(params).sample(jax.random.PRNGKey(0))).shape[0]
    )
    n = int(params.max_steps_in_episode)

    def run(level):
        key = jax.random.PRNGKey(0)
        obs, state = env.reset_env(key, params)
        ys = [float(np.asarray(obs)[idx][0])]
        a = jnp.full((adim,), level, dtype=jnp.float32)
        for _ in range(n):
            key, sub = jax.random.split(key)
            obs, state, _, term, _ = step(sub, state, a, params)
            ys.append(float(np.asarray(obs)[idx][0]))
            if bool(term):
                break
        return np.array(ys)

    lo, hi = run(-0.4), run(0.4)
    m = min(len(lo), len(hi))
    d = hi[:m] - lo[:m]
    if m < 5 or not np.isfinite(d).all():
        return {"tau_steps": float("nan"), "shape": "unusable", "episode": n}

    total = d[-1] - d[0]
    if abs(total) < 1e-12:
        return {"tau_steps": float("nan"), "shape": "no response", "episode": n}

    # Integrating when the response is still climbing at a comparable rate at
    # the end of the window: compare the last quarter's slope with the first's.
    q = max(m // 4, 2)
    early = (d[q] - d[0]) / (q * dt)
    late = (d[-1] - d[-q]) / (q * dt)
    if abs(late) > 0.5 * abs(early) and abs(late) > 1e-12:
        return {"tau_steps": float("nan"), "shape": "integrating", "episode": n}

    cross = int(np.argmax((d - d[0]) / total >= 0.632))
    tau = float(cross) if cross > 0 else float("nan")
    return {"tau_steps": tau, "shape": "first-order", "episode": n}


def required_steps(response: dict) -> float:
    """Episode length the protocol's actuator clause demands, or nan."""
    tau = response["tau_steps"]
    return float("nan") if tau != tau else TAU_MULTIPLE * tau
