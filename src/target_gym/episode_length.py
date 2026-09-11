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

**The probe window is not the benchmark episode, and that distinction is the
whole point.** Measuring ``tau`` inside ``N`` makes the rule circular: a longer
episode sees more of the response, reports a larger ``tau``, and so demands a
longer episode. Measured that way the same aircraft reads ``integrating`` at
N=280, 217 steps at N=480 and 365 from N=800 onward, and ``plane``,
``plane_sine`` and ``plane_energy`` return identical numbers at equal N because
they are the same plant. Three apparent violations were nothing but three
episode lengths.

So the probe runs on a fixed long window and has to *converge*: the response is
measured twice, at ``PROBE_STEPS`` and at twice that, and a ``tau`` is only
reported when the two agree within ``CONVERGENCE_TOL``. A plant still climbing
at both, as the glass furnace is (822, 1520, 2383, 3415 steps at successive
doublings), has no time constant on this timescale and is reported as
``does not settle``. That is a fact worth knowing about the plant, and it is not
the same thing as an episode being too short.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

#: Time constants that must fit inside an episode, from the protocol.
TAU_MULTIPLE = 10

#: Steps the probe drives before it is doubled to test convergence. Generous on
#: purpose: this is a diagnostic, not a rollout, and it has to be long enough to
#: see the whole response of the slowest plant that has one.
PROBE_STEPS = 4000

#: Relative agreement required between the two probe windows.
CONVERGENCE_TOL = 0.10

#: Fraction of the response allowed to move against its overall direction before
#: it is called oscillatory rather than first-order. Some reversal is numerical;
#: a phugoid is half the trace.
MONOTONE_TOL = 0.15


def measure_response(spec, params=None) -> dict:
    """Open-loop step response of the tracked output to the actuator.

    Returns ``{"tau_steps", "shape", "episode"}``, where ``episode`` is the
    benchmark episode length and ``tau_steps`` is measured independently of it.
    ``tau_steps`` is ``nan`` unless ``shape`` is ``"first-order"``.

    Driven differentially, stepping the actuator to -0.4 and +0.4 and
    subtracting, so a plant that drifts under its own dynamics does not have
    that drift counted as a response.
    """
    if params is None:
        params = spec.make_test_params()
    episode = int(params.max_steps_in_episode)
    first = _probe(spec, params, PROBE_STEPS)
    if first["shape"] != "first-order":
        return {**first, "episode": episode}

    second = _probe(spec, params, 2 * PROBE_STEPS)
    a, b = first["tau_steps"], second["tau_steps"]
    if second["shape"] != "first-order" or abs(b - a) > CONVERGENCE_TOL * max(a, b):
        return {
            "tau_steps": float("nan"),
            "shape": "does not settle",
            "episode": episode,
            "probe": (a, b),
        }
    return {"tau_steps": b, "shape": "first-order", "episode": episode}


def _probe(spec, params, window: int) -> dict:
    """One differential step response over *window* steps."""
    from target_gym.runners.runners import _as_tuple

    env = spec.make_env()
    dt = float(getattr(params, "delta_t", 1.0))
    idx = list(_as_tuple(env.obs_value_index))[:1]
    step = jax.jit(env.step_env)
    adim = int(
        np.atleast_1d(env.action_space(params).sample(jax.random.PRNGKey(0))).shape[0]
    )
    n = int(window)

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
        return {"tau_steps": float("nan"), "shape": "unusable"}

    total = d[-1] - d[0]
    if abs(total) < 1e-12:
        return {"tau_steps": float("nan"), "shape": "no response"}

    # Integrating when the response is still climbing at a comparable rate at
    # the end of the window: compare the last quarter's slope with the first's.
    q = max(m // 4, 2)
    early = (d[q] - d[0]) / (q * dt)
    late = (d[-1] - d[-q]) / (q * dt)
    if abs(late) > 0.5 * abs(early) and abs(late) > 1e-12:
        return {"tau_steps": float("nan"), "shape": "integrating"}

    # A first-order step response is monotone. An oscillatory one has no tau_63
    # to read, and fitting one anyway is how a fixed elevator deflection on an
    # airliner -- which excites the phugoid, so altitude rises and falls -- came
    # out as a 365-step "time constant" that then demanded a 3650-step episode.
    # Measured on the fraction of steps that move against the overall direction.
    steps = np.diff(d)
    against = float(np.sum(np.sign(steps) != np.sign(total)) / max(len(steps), 1))
    if against > MONOTONE_TOL:
        return {"tau_steps": float("nan"), "shape": "oscillatory"}

    cross = int(np.argmax((d - d[0]) / total >= 0.632))
    tau = float(cross) if cross > 0 else float("nan")
    return {"tau_steps": tau, "shape": "first-order"}


def required_steps(response: dict) -> float:
    """Episode length the protocol's actuator clause demands, or nan."""
    tau = response["tau_steps"]
    return float("nan") if tau != tau else TAU_MULTIPLE * tau
