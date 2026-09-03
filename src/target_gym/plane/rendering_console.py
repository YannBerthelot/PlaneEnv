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

# How much recent flight the scene panel shows behind the aircraft.
_TRAIL_SAMPLES = 80


# Faint cloud layers, drifting with the distance actually flown. The settled
# scene is otherwise motionless -- the aircraft is pinned mid-panel holding a
# constant altitude -- so nothing in it says the aircraft is moving at 230 m/s.
# Three depths at different parallax rates give that back for the cost of a few
# ellipses. The alphas are deliberately near-invisible: a first attempt at
# 0.16 filled the panel with grey blobs that competed with the aircraft,
# which is the opposite of the point.
_CLOUD_LAYERS = ((0.045, 0.10, 2600.0), (0.030, 0.15, 5600.0), (0.020, 0.22, 9800.0))
_CLOUD_RNG = np.random.default_rng(7)
_CLOUD_SEEDS = [
    _CLOUD_RNG.uniform(0.0, 1.0, size=(4, 3)) for _ in range(len(_CLOUD_LAYERS))
]


# The aircraft is the A320 mesh that plane3d already flies -- fourteen faces in
# body frame, x forward, y right, z up -- rather than a second hand-drawn
# silhouette. One model means the 2D and 3D environments show the same aircraft,
# and the geometry is the sourced one from PHYSICS.md (37.57 m long) instead of
# a shape that merely looked about right.
def _rotate(points: np.ndarray, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return points @ np.array([[c, -s], [s, c]])


def _draw_clouds(ax, x_flown: float):
    """Parallax cloud layers, positioned by how far the aircraft has flown."""
    from matplotlib.patches import Ellipse

    for (alpha, scale, wavelength), seeds in zip(_CLOUD_LAYERS, _CLOUD_SEEDS):
        for u, v, w in seeds:
            # Wrap the layer so clouds re-enter from the right as they leave.
            x = ((u - x_flown / wavelength) % 1.15) - 0.075
            y = 0.24 + 0.62 * v
            for k, (dx, dy, sx) in enumerate(
                ((0.0, 0.0, 1.0), (0.55, 0.10, 0.72), (-0.5, 0.06, 0.66))
            ):
                ax.add_patch(
                    Ellipse(
                        (x + dx * scale, y + dy * scale),
                        width=scale * (1.5 + 0.5 * w) * sx,
                        height=scale * (0.55 + 0.2 * w) * sx,
                        facecolor=rk.TEXT,
                        edgecolor="none",
                        alpha=alpha,
                        zorder=1 + k * 0,
                    )
                )


def _mesh_side_view():
    """The shared A320 mesh, ready to project onto a side elevation."""
    from target_gym.plane3d.rendering import _build_plane_faces

    faces, length, _, _, _ = _build_plane_faces()
    return faces, float(length)


def _face_color(rgb) -> str:
    """Map the mesh's greyscale shading onto the console palette.

    The mesh carries plain greys (160-245) that encode which way a face points.
    Keeping that shading but re-tinting it is what makes the aircraft belong to
    the same drawing as the gauges beside it.
    """
    lum = float(np.mean(rgb)) / 255.0
    return rk.lerp_hex(rk.FRAME, "#eef4fc", float(np.clip((lum - 0.55) / 0.42, 0, 1)))


def _draw_aircraft(ax, x, y, theta, scale, color, shade):
    """Project the mesh in side elevation, pitched by ``theta``.

    Orthographic, camera looking along -y, so screen x is body x and screen y is
    body z. Faces are drawn back to front by their mean y, which is all the
    depth ordering a side view needs.
    """
    faces, length = _mesh_side_view()
    c, s_ = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s_], [s_, c]])
    unit = scale / length

    for verts, rgb, name in sorted(faces, key=lambda f: float(np.mean(f[0][:, 1]))):
        xy = np.column_stack([verts[:, 0], verts[:, 2]]) * unit
        xy = xy @ rot.T + np.array([x, y])
        ax.fill(
            xy[:, 0],
            xy[:, 1],
            facecolor=_face_color(rgb),
            edgecolor=rk.FRAME,
            lw=0.5,
            zorder=6,
            joinstyle="round",
        )

    # Cabin windows along the fuselage side.
    win = (
        np.column_stack(
            [np.linspace(-0.34, 0.30, 13) * scale, np.full(13, 0.012 * scale)]
        )
        @ rot.T
    )
    ax.scatter(win[:, 0] + x, win[:, 1] + y, s=1.4, color=rk.BG, zorder=8, linewidths=0)


def _draw_scene(ax, state, params, history):
    """The flight itself: aircraft, commanded altitude, ground, trail."""
    rk.schematic_axes(ax, xlim=(0.0, 1.0), ylim=(0.0, 1.0))

    alt = float(state.z)
    target = float(state.target_altitude)
    lo, hi = float(params.min_alt), float(params.max_alt)
    span = max(hi - lo, 1.0)

    def to_y(a: float) -> float:
        return 0.06 + 0.88 * float(np.clip((a - lo) / span, 0.0, 1.0))

    _draw_clouds(ax, float(state.x))

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

    # Where it has just been. A recent window, not the whole flight: over a
    # 10 000-step episode the history holds a thousand samples, and drawing all
    # of them compressed the climb into a vertical spike detached from the
    # aircraft. The strip chart underneath is what shows the whole episode.
    trail = history["altitude"][-_TRAIL_SAMPLES:]
    if len(trail) > 1:
        xs = np.linspace(0.62 - 0.34 * (len(trail) / _TRAIL_SAMPLES), 0.62, len(trail))
        ax.plot(
            xs,
            [to_y(a) for a in trail],
            color=rk.CYAN,
            lw=1.2,
            alpha=0.6,
            zorder=4,
            solid_capstyle="round",
        )

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
