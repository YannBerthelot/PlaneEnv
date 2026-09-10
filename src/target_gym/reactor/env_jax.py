import os
from typing import Callable, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from gymnax.environments import environment, spaces

from target_gym.base import canonical_reset
from target_gym.reactor.env import (
    ReactorParams,
    ReactorState,
    check_is_terminal,
    compute_next_state,
    compute_reward,
    get_obs,
    steady_state_precursors,
    steady_state_xenon,
)
from target_gym.reactor.rendering import _render
from target_gym.utils import save_video

# Number of physics sub-steps per control step. Constant so JIT can treat it
# as static in `lax.scan(length=...)`. Change here only — `env.py` reads it
# via a delayed import in `check_is_terminal` / `get_target_from_schedule`.
CONTROL_PERIOD: int = 10


class Reactor(environment.Environment[ReactorState, ReactorParams]):
    """
    Nuclear reactor (point kinetics + thermal feedback).

    Observation (4,): [n, T_coolant, rho_ext_norm, target_n]
    Action      (1,): [rho_ext_norm] in [-1, 1] (control rod position)
    """

    render_reactor = classmethod(_render)
    screen_width = 700
    screen_height = 900
    # Number of physics sub-steps per env-step. Exposed so external
    # rollout code (eval scripts) can convert max_steps_in_episode
    # (physics units) to env-step counts.
    control_period: int = CONTROL_PERIOD

    # obs = [n, T_coolant, rho_ext_norm, target_n]
    obs_value_index: int = 0  # n (neutron density / normalised power)
    tracked_names: tuple = ("neutron power (normalised)",)
    obs_target_index: int = 3  # target_n

    def __init__(self, integration_method: str = "tr_bdf2_2"):
        self.obs_shape = (4,)
        self.integration_method = integration_method

    @property
    def default_params(self) -> ReactorParams:
        return ReactorParams()

    def compute_reward(self, state, params):
        return compute_reward(state, params)

    def step_env(
        self,
        key: chex.PRNGKey,
        state: ReactorState,
        action: jnp.ndarray,
        params: ReactorParams = None,
    ):
        if params is None:
            params = self.default_params

        rho_raw = action
        if not isinstance(action, float):
            rho_raw = action.reshape(())

        # Action is held constant for `control_period` physics sub-steps. Reward
        # is summed across the sub-steps; we freeze the state on termination so
        # the scan can still run for a fixed length under jit.
        def sub_step(carry, _):
            state, cum_reward, done, terminated = carry
            candidate, _metrics = compute_next_state(
                rho_raw, state, params, integration_method=self.integration_method
            )
            r = compute_reward(candidate, params, xp=jnp)
            term, trunc = check_is_terminal(candidate, params, xp=jnp)
            new_done = term | trunc
            # Freeze state once done; still accumulate the final-step reward.
            next_state = jax.tree.map(
                lambda a, b: jnp.where(done, a, b), state, candidate
            )
            next_reward = cum_reward + jnp.where(done, 0.0, r)
            next_done = done | new_done
            # Natural termination is tracked apart from `done` so `step_env`
            # can report it alone -- gymnax >= 1.0 derives `truncated` itself
            # from the returned state's `time`. The `~done` guard records only
            # the first crossing: once frozen, `term` keeps firing on the
            # held state.
            next_terminated = terminated | (term & jnp.logical_not(done))
            return (next_state, next_reward, next_done, next_terminated), None

        (new_state, reward, _done, terminated), _ = jax.lax.scan(
            sub_step,
            (state, jnp.float32(0.0), jnp.bool_(False), jnp.bool_(False)),
            xs=None,
            length=CONTROL_PERIOD,
        )

        # Mean over the sub-steps rather than the sum. The action is held for
        # ``CONTROL_PERIOD`` physics sub-steps, and summing made one environment
        # step of this plant worth several times a step of any other -- its
        # per-step reward peaked near 4.7 where every other environment caps
        # around 1.0, which is misleading the moment returns are read across
        # environments. Dividing by a positive constant leaves the optimal policy
        # and every within-environment comparison untouched.
        #
        # Terminating mid-period still costs: only the sub-steps before the stop
        # contribute to the sum, so the mean falls with them.
        obs = self.get_obs(new_state)
        return (
            obs,
            new_state,
            reward / CONTROL_PERIOD,
            terminated,
            {"last_state": new_state},
        )

    def get_obs(self, state: ReactorState, params: ReactorParams = None):
        if params is None:
            params = self.default_params
        return get_obs(state, params=params)

    def is_terminated(self, state: ReactorState, params: ReactorParams) -> jnp.ndarray:
        """Natural termination only; the time limit is gymnax's ``is_truncated``."""
        terminated, _ = check_is_terminal(state, params)
        return terminated

    @canonical_reset
    def reset_env(
        self, key: chex.PRNGKey, params: ReactorParams = None
    ) -> Tuple[jnp.ndarray, ReactorState]:
        if params is None:
            params = self.default_params

        key, n_key, target_key, demand_key = jax.random.split(key, 4)

        initial_n = jax.random.uniform(
            n_key,
            minval=params.initial_n_range[0],
            maxval=params.initial_n_range[1],
        )
        # Initial demand drawn from target range; OU process evolves it from here.
        initial_target = jax.random.uniform(
            target_key,
            minval=params.target_n_range[0],
            maxval=params.target_n_range[1],
        )
        # Precursors start at steady state for the initial neutron density;
        # without that there is a large transient in the first few seconds.
        initial_C = steady_state_precursors(initial_n, params)

        # Xenon and iodine start at the equilibrium for a *different*, recent
        # power level, not the current one. A reactor that has been
        # load-following is essentially never at xenon equilibrium, and starting
        # it there made the poison a constant bias: with dXe/dt = 0 at t = 0 and
        # a 13.2 h xenon time constant, the term the module docstring calls "the
        # dominant control challenge" contributed nothing an operator would have
        # to trim. Sampling the history instead makes it live from the first
        # step. The range is deliberately narrow, so the offset is one a real
        # unit would carry rather than an extreme.
        key, history_key = jax.random.split(key)
        n_recent = jax.random.uniform(
            history_key,
            minval=params.initial_xenon_power_range[0],
            maxval=params.initial_xenon_power_range[1],
        )
        initial_I_hat, initial_Xe_hat = steady_state_xenon(n_recent, params)

        state = ReactorState(
            time=0,
            n=initial_n,
            C=initial_C,
            T_fuel=jnp.asarray(params.initial_T_fuel, dtype=jnp.float32),
            T_coolant=jnp.asarray(params.initial_T_coolant, dtype=jnp.float32),
            I_hat=jnp.asarray(initial_I_hat, dtype=jnp.float32),
            Xe_hat=jnp.asarray(initial_Xe_hat, dtype=jnp.float32),
            target_n=initial_target,
            demand_key=demand_key,
            rho_ext=jnp.zeros((), dtype=jnp.float32),
            rho_ext_cmd=jnp.asarray(0.0, dtype=jnp.float32),
        )

        obs = self.get_obs(state)
        return obs, state

    def action_space(self, params: ReactorParams | None = None) -> spaces.Box:
        return spaces.Box(
            low=jnp.array([-1.0]),
            high=jnp.array([1.0]),
            shape=(1,),
            dtype=jnp.float32,
        )

    def observation_space(self, params: ReactorParams) -> spaces.Box:
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(-inf, inf, self.obs_shape, dtype=jnp.float32)

    def state_space(self, params: ReactorParams) -> spaces.Box:
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(
            -inf, inf, len(ReactorState.__dataclass_fields__), dtype=jnp.float32
        )

    @property
    def expert_policy(self):
        from target_gym.experts.pid import (
            FunctionalExpertPolicy,
            make_reactor_pid,
            pid_step,
        )

        params, zero_state = make_reactor_pid()
        return FunctionalExpertPolicy(params, zero_state, pid_step)

    def make_pid(self):
        """Return a ready-to-use StatefulPID for neutron-power tracking."""
        from target_gym.experts.pid import make_reactor_stateful_pid

        return make_reactor_stateful_pid()

    def make_mpc(self, params=None, **kwargs):
        """Return a CasADi MPC oracle for neutron-power tracking."""
        from target_gym.experts.mpc import make_reactor_mpc

        if params is None:
            params = self.default_params
        return make_reactor_mpc(self, params, **kwargs)

    def save_video(
        self,
        select_action: Callable[[jnp.ndarray], jnp.ndarray],
        seed: int,
        params=None,
        folder="videos",
        episode_index=0,
        FPS=60,
        format="mp4",
    ):
        return save_video(
            self,
            select_action,
            folder,
            episode_index,
            FPS,
            params,
            seed=seed,
            format=format,
        )

    def render(self, screen, state: ReactorState, params: ReactorParams, frames, clock):
        frames, screen, clock = self.render_reactor(
            screen, state, params, frames, clock
        )
        return frames, screen, clock


if __name__ == "__main__":
    env = Reactor()
    seed = 42
    env_params = ReactorParams(
        max_steps_in_episode=2000
    )  # 2000 physics = 200 control steps
    os.makedirs("videos/reactor", exist_ok=True)
    env.save_video(
        lambda o: np.random.uniform(-1, 1),
        seed,
        folder="videos/reactor",
        episode_index=0,
        params=env_params,
        format="gif",
    )
