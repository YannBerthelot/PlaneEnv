"""Console renderer for the 2D aircraft, on the shared instrument kit.

The sixteen process and industrial plants all render through
``render_kit.frame`` -- schematic panel, instrument stack, strip chart, one dark
palette. The aircraft did not: it drew a sky-blue pygame scene with a text box,
which beside a furnace or a reactor in the same gallery read as a different
project rather than a different environment.

This keeps the thing worth keeping -- an aircraft visibly flying toward a
commanded altitude -- and puts it inside that chrome, as the schematic panel,
with real gauges beside it and the tracking history underneath. The scene is
drawn in matplotlib rather than blitted from pygame, so the anti-aliasing,
typography and colours are the same ones every other environment uses.
"""

from __future__ import annotations

import numpy as np

from target_gym import render_kit as rk

HISTORY_KEYS = ("t", "altitude", "target", "speed", "pitch", "power", "reward")

# Side elevation of an airliner, in units of its own length, nose at +x. A side
# view shows one wing edge-on and a vertical fin -- mirroring the wing about the
# fuselage, as a first attempt did, draws a plan view and reads as a cross.
_FUSELAGE = np.array(
    [
        (0.52, 0.010),
        (0.44, 0.045),
        (0.10, 0.060),
        (-0.30, 0.055),
        (-0.46, 0.030),
        (-0.52, 0.004),
        (-0.46, -0.020),
        (-0.20, -0.045),
        (0.20, -0.048),
        (0.44, -0.030),
    ]
)
_WING = np.array([(0.10, -0.02), (-0.16, -0.20), (-0.26, -0.20), (-0.06, -0.02)])
_FIN = np.array([(-0.30, 0.05), (-0.40, 0.24), (-0.50, 0.24), (-0.46, 0.04)])
_HSTAB = np.array([(-0.42, 0.02), (-0.56, 0.10), (-0.62, 0.09), (-0.50, 0.01)])
_WINDOWS = np.linspace(-0.24, 0.30, 11)


def _rotate(points: np.ndarray, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return points @ np.array([[c, -s], [s, c]])


def _draw_aircraft(ax, x, y, theta, scale, color, shade):
    origin = np.array([x, y])
    for shape, fc, z in (
        (_WING, shade, 5),
        (_HSTAB, shade, 5),
        (_FIN, shade, 7),
        (_FUSELAGE, color, 6),
    ):
        pts = _rotate(shape * scale, theta) + origin
        ax.fill(pts[:, 0], pts[:, 1], color=fc, ec=rk.FRAME, lw=0.7, zorder=z)
    # Cabin windows: a row of dots is what makes a shape read as an airliner
    # rather than a dart, and it costs one scatter call.
    win = _rotate(
        np.column_stack([_WINDOWS, np.full_like(_WINDOWS, 0.012)]) * scale, theta
    )
    ax.scatter(win[:, 0] + x, win[:, 1] + y, s=1.6, color=rk.BG, zorder=8, linewidths=0)


def _draw_scene(ax, state, params, history):
    """The flight itself: aircraft, commanded altitude, ground, trail."""
    rk.schematic_axes(ax, xlim=(0.0, 1.0), ylim=(0.0, 1.0))

    alt = float(state.z)
    target = float(state.target_altitude)
    lo, hi = float(params.min_alt), float(params.max_alt)
    span = max(hi - lo, 1.0)

    def to_y(a: float) -> float:
        return 0.06 + 0.88 * float(np.clip((a - lo) / span, 0.0, 1.0))

    # Ground, and the altitude envelope the episode ends outside of.
    ax.axhspan(0.0, to_y(lo), color=rk.PANEL, zorder=0)
    ax.axhline(to_y(lo), color=rk.FRAME, lw=1.4, zorder=1)
    ax.axhline(to_y(hi), color=rk.FRAME, lw=0.8, ls=(0, (4, 4)), zorder=1)
    ax.text(0.012, to_y(hi) + 0.015, "CEILING", color=rk.DIM, size=6.0, family=rk.MONO)
    ax.text(0.012, to_y(lo) + 0.015, "GROUND", color=rk.DIM, size=6.0, family=rk.MONO)

    # Commanded altitude, and the precision floor the reward stops paying below.
    y_t = to_y(target)
    ax.axhline(y_t, color=rk.AMBER, lw=1.3, ls=(0, (6, 3)), zorder=3)
    floor = float(getattr(params, "precision_floor", 1.0))
    band = 0.80 * (50.0 * floor) / span
    ax.axhspan(y_t - band, y_t + band, color=rk.AMBER, alpha=0.10, zorder=2)
    ax.text(
        0.985,
        y_t + 0.02,
        f"TARGET {target:,.0f} m",
        color=rk.AMBER,
        size=6.8,
        family=rk.MONO,
        ha="right",
    )

    # Where it has been: the trail is the tracking record, which is the point.
    if len(history["altitude"]) > 1:
        n = len(history["altitude"])
        xs = np.linspace(max(0.62 - 0.018 * n, 0.06), 0.62, n)
        ys = [to_y(a) for a in history["altitude"]]
        ax.plot(xs, ys, color=rk.CYAN, lw=1.1, alpha=0.55, zorder=4)

    x_ac = 0.62
    y_ac = to_y(alt)
    rk.glow(ax, x_ac, y_ac, 0.045, color=rk.CYAN, strength=0.5, layers=5, zorder=5)
    _draw_aircraft(ax, x_ac, y_ac, float(state.theta), 0.26, rk.TEXT, "#9eb6ce")

    err = alt - target
    ax.annotate(
        "",
        xy=(x_ac + 0.10, y_ac),
        xytext=(x_ac + 0.10, y_t),
        arrowprops=dict(arrowstyle="<->", color=rk.DIM, lw=0.9, shrinkA=0, shrinkB=0),
        zorder=4,
    )
    ax.text(
        x_ac + 0.125,
        0.5 * (y_ac + y_t),
        f"{err:+,.0f} m",
        color=rk.DIM,
        size=6.5,
        family=rk.MONO,
        va="center",
    )


def render_plane_console(state, params, step, history):
    dt = float(params.delta_t)
    speed = float(np.hypot(float(state.x_dot), float(state.z_dot)))
    history["t"].append(step * dt / 60.0)
    history["altitude"].append(float(state.z))
    history["target"].append(float(state.target_altitude))
    history["speed"].append(speed)
    history["pitch"].append(float(np.rad2deg(state.theta)))
    history["power"].append(float(state.power))
    from target_gym.plane.env import compute_reward

    history["reward"].append(float(compute_reward(state, params)))

    lo, hi = float(params.min_alt), float(params.max_alt)
    span = max(hi - lo, 1.0)

    err = abs(float(state.z) - float(state.target_altitude))
    # Scale the bands to the envelope rather than to absolute metres: mid-climb
    # toward a commanded step is not an alarm condition, it is the task.
    rel = err / span
    status = rk.NOMINAL if rel < 0.03 else (rk.WATCH if rel < 0.15 else rk.ALARM)
    frac = lambda a: float(np.clip((float(a) - lo) / span, 0.0, 1.0))  # noqa: E731

    gauges = [
        rk.Gauge(
            "ALTITUDE",
            f"{float(state.z):,.0f} m",
            frac(state.z),
            rk.CYAN,
            target_frac=frac(state.target_altitude),
            limit_frac=1.0,
        ),
        rk.Gauge(
            "TARGET",
            f"{float(state.target_altitude):,.0f} m",
            frac(state.target_altitude),
            rk.AMBER,
        ),
        rk.Gauge(
            "AIRSPEED",
            f"{speed:,.0f} m/s",
            float(np.clip(speed / 300.0, 0, 1)),
            rk.BLUE,
        ),
        rk.Gauge(
            "PITCH",
            f"{np.rad2deg(float(state.theta)):+.1f}°",
            float(np.clip((np.rad2deg(float(state.theta)) + 20) / 40, 0, 1)),
            rk.TEAL,
        ),
        rk.Gauge(
            "POWER",
            f"{float(state.power) * 100:.0f} %",
            float(np.clip(float(state.power), 0, 1)),
            rk.GREEN,
        ),
        rk.Gauge(
            "FUEL",
            f"{float(state.fuel):,.0f} kg",
            float(
                np.clip(
                    float(state.fuel) / max(float(params.initial_fuel_quantity), 1),
                    0,
                    1,
                )
            ),
            rk.ORANGE,
            hidden=True,
        ),
    ]

    strip = rk.Strip(
        t=history["t"],
        series=[
            rk.Series(history["altitude"], "altitude", rk.CYAN),
            rk.Series(history["target"], "commanded", rk.AMBER, ls="--"),
        ],
        ylabel="altitude (m)",
        xlabel="minutes",
    )

    fig = rk.frame(
        title="AIRCRAFT  —  ALTITUDE  HOLD",
        step=step,
        elapsed_s=step * dt,
        schematic=lambda ax: _draw_scene(ax, state, params, history),
        schematic_title="FLIGHT  PROFILE",
        gauges=gauges,
        strips=[strip],
        status=status,
        subtitle="A320-like  ·  stall and shock-stall modelled  ·  fuel burn",
    )
    return rk.finish(fig), history


_render = rk.make_render_hook(render_plane_console, HISTORY_KEYS, stride=10)
