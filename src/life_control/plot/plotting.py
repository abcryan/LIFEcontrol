"""
Plotting utilities for the LIFE N-spacecraft truth-model simulation.

State layout in X_hist (per row, length 14 * N):
    Chief   (i = 0): [r^I, v^I, q, ω, m]                   — absolute ICRF
    Deputy  (i ≥ 1): [δr^I, δv^I, q, ω, m]                  — relative to chief

Plots:
    plot_trajectory             : multi-panel summary (chief + formation geometry)
    plot_solar_system           : inner solar system with all spacecraft
    plot_l2_rotating_frame_zoom : Sun–Earth L2 rotating-frame zoom, all spacecraft

The "formation geometry" panels show the deputies in chief-centered ICRF,
which is the appropriate frame for visualizing the formation at mm-to-m scale.
"""
import numpy as np
import spiceypy as spice
from types import SimpleNamespace

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# State block size (must match main.py NX_PER_SC).
NX_PER_SC: int = 14


# ── Helpers: pull per-spacecraft sub-states out of the stacked history ──────

def _chief_state(X_hist: np.ndarray) -> dict[str, np.ndarray]:
    """Return chief's absolute states (r, v, q, ω, m) from stacked X_hist."""
    return dict(
        r = X_hist[:, 0:3],
        v = X_hist[:, 3:6],
        q = X_hist[:, 6:10],
        w = X_hist[:, 10:13],
        m = X_hist[:, 13],
    )


def _deputy_state(X_hist: np.ndarray, i: int) -> dict[str, np.ndarray]:
    """
    Return deputy i's states (δr, δv, q, ω, m).
    NOTE: δr and δv are *relative* to chief in ICRF. To get absolute
    positions, ADD chief's r/v.
    """
    base = NX_PER_SC * i
    return dict(
        dr = X_hist[:, base + 0 : base + 3],
        dv = X_hist[:, base + 3 : base + 6],
        q  = X_hist[:, base + 6 : base + 10],
        w  = X_hist[:, base + 10 : base + 13],
        m  = X_hist[:, base + 13],
    )


def _absolute_position(X_hist: np.ndarray, i: int) -> np.ndarray:
    """Absolute ICRF position of spacecraft i, materialized from stacked state."""
    if i == 0:
        return X_hist[:, 0:3]
    base = NX_PER_SC * i
    return X_hist[:, 0:3] + X_hist[:, base : base + 3]


# Color palette for spacecraft (consistent across all plots).
SC_COLORS = [
    "#FFFFFF",   # chief: white
    "#FF4D6D",   # deputy 1: red
    "#3A8DDE",   # deputy 2: blue
    "#5BC85B",   # deputy 3: green
    "#FFB347",   # deputy 4: orange
]


def _sc_color(i: int) -> str:
    return SC_COLORS[i % len(SC_COLORS)]


# ── Sun-Earth L2 rotating-frame helper ───────────────────────────────────────

L2_DIST_KM = 1.5e6


def sun_earth_l2_frame(
    et: float,
    l2_dist_km: float = L2_DIST_KM,
) -> dict[str, np.ndarray]:
    """Build an instantaneous Sun-Earth L2 rotating-frame triad at epoch et."""
    state_earth = spice.spkezr("EARTH", et, "J2000", "NONE", "SSB")[0]
    state_sun   = spice.spkezr("SUN",   et, "J2000", "NONE", "SSB")[0]
    r_earth, v_earth = state_earth[:3], state_earth[3:]
    r_sun,   v_sun   = state_sun[:3],   state_sun[3:]

    r_es = r_earth - r_sun
    v_es = v_earth - v_sun

    e_x = r_es / np.linalg.norm(r_es)
    h_es = np.cross(r_es, v_es)
    e_z = h_es / np.linalg.norm(h_es)
    e_y = np.cross(e_z, e_x)
    e_y = e_y / np.linalg.norm(e_y)

    omega_vec = h_es / np.dot(r_es, r_es)
    r_l2 = r_earth + l2_dist_km * e_x
    v_l2 = v_sun + np.cross(omega_vec, r_l2 - r_sun)

    return dict(r_l2=r_l2, v_l2=v_l2, e_x=e_x, e_y=e_y, e_z=e_z, omega_vec=omega_vec)



# ── Plot 0: Spacecraft List ────────────────────────────────────────────────

# Build the per-spacecraft list of label/J_B namespaces consumed by the
# plotting layer (kept minimal — only the two attributes the plots use).
def build_plot_spacecraft(param, n_sc):
    sc = [SimpleNamespace(label="leader", J_B=param.J_init_L)]
    for i in range(1, n_sc):
        sc.append(SimpleNamespace(label=f"follower_{i}", J_B=param.J_init_F))
    return sc



# ── Plot 1: trajectory summary (multi-spacecraft) ────────────────────────────

def plot_trajectory(
    t_hist:     np.ndarray,
    X_hist:     np.ndarray,
    et0:        float,
    spacecraft: list,
) -> None:
    """
    Multi-panel summary for an N-spacecraft propagation.

    Layout (3 x 3):
        (0,0)  Chief 3D trajectory in ICRF (with Sun and Earth)
        (0,1)  Formation geometry: deputies in chief-centered ICRF, 3D
        (0,2)  Baseline magnitudes |δr_i|(t) for each deputy
        (1,0)  Chief position components
        (1,1)  Chief velocity components
        (1,2)  Chief attitude quaternion components (q0, q1, q2, q3)
        (2,0)  Per-deputy relative δr in chief-centered ICRF, x/y/z vs time
        (2,1)  All spacecraft angular velocity magnitudes
        (2,2)  Chief conservation diagnostics
    """
    N        = len(spacecraft)
    chief    = _chief_state(X_hist)
    deputies = [_deputy_state(X_hist, i) for i in range(1, N)]
    t_h      = t_hist / 3600.0

    # Earth track for the chief 3D panel.
    n_ref = 200
    t_ref = np.linspace(t_hist[0], t_hist[-1], n_ref)
    r_earth_hist = np.array([
        spice.spkezr("EARTH", et0 + tk, "J2000", "NONE", "SSB")[0][:3] for tk in t_ref
    ])
    r_sun_pos = spice.spkezr("SUN", et0, "J2000", "NONE", "SSB")[0][:3]

    # --- Figure ----------------------------------------------------------
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        f"LIFE Truth Model — N = {N} spacecraft (1 chief + {N-1} deputies)",
        fontsize=14, fontweight="bold",
    )

    # (0,0) Chief 3D trajectory in ICRF with Sun/Earth context
    ax = plt.subplot2grid((3, 3), (0, 0), projection="3d", fig=fig)
    ax.plot(chief["r"][:, 0], chief["r"][:, 1], chief["r"][:, 2],
            color="C0", lw=1.5, label="chief")
    ax.plot(r_earth_hist[:, 0], r_earth_hist[:, 1], r_earth_hist[:, 2],
            "g-", lw=1.0, alpha=0.7, label="Earth")
    ax.scatter([r_sun_pos[0]], [r_sun_pos[1]], [r_sun_pos[2]],
               color="orange", s=140, marker="*", label="Sun")
    ax.scatter([chief["r"][0, 0]], [chief["r"][0, 1]], [chief["r"][0, 2]],
               color="blue", s=40, marker="o", label="start")
    ax.scatter([chief["r"][-1, 0]], [chief["r"][-1, 1]], [chief["r"][-1, 2]],
               color="red", s=40, marker="x", label="end")
    ax.set_xlabel("x [km]"); ax.set_ylabel("y [km]"); ax.set_zlabel("z [km]")
    ax.set_title("Chief trajectory in ICRF")
    ax.legend(loc="upper left", fontsize=7)

    # (0,1) Formation geometry: deputies in chief-centered ICRF (3D)
    ax = plt.subplot2grid((3, 3), (0, 1), projection="3d", fig=fig)
    ax.scatter([0], [0], [0], color=_sc_color(0), s=80, marker="*",
               edgecolors="black", linewidths=0.5, label=spacecraft[0].label)
    for k, d in enumerate(deputies, start=1):
        # δr is already in ICRF, in km. Convert to meters for the formation plot.
        dr_m = d["dr"] * 1e3
        ax.plot(dr_m[:, 0], dr_m[:, 1], dr_m[:, 2],
                color=_sc_color(k), lw=1.2, alpha=0.85, label=spacecraft[k].label)
        ax.scatter([dr_m[0, 0]], [dr_m[0, 1]], [dr_m[0, 2]],
                   color=_sc_color(k), s=25, marker="o", edgecolors="black", linewidths=0.4)
        ax.scatter([dr_m[-1, 0]], [dr_m[-1, 1]], [dr_m[-1, 2]],
                   color=_sc_color(k), s=35, marker="x")
    ax.set_xlabel("δx [m]"); ax.set_ylabel("δy [m]"); ax.set_zlabel("δz [m]")
    ax.set_title("Formation geometry (chief-centered ICRF)")
    ax.legend(loc="upper left", fontsize=7)

    # (0,2) Baseline magnitudes |δr_i|(t)
    ax = plt.subplot2grid((3, 3), (0, 2), fig=fig)
    for k, d in enumerate(deputies, start=1):
        baseline_m = np.linalg.norm(d["dr"], axis=1) * 1e3
        ax.plot(t_h, baseline_m, color=_sc_color(k), lw=1.4,
                label=f"{spacecraft[k].label}")
    ax.set_xlabel("time [h]"); ax.set_ylabel(r"$|\delta r|$ [m]")
    ax.set_title("Baseline magnitudes (chief → deputy)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    # (1,0) Chief position components
    ax = plt.subplot2grid((3, 3), (1, 0), fig=fig)
    ax.plot(t_h, chief["r"][:, 0], "r-", label="x")
    ax.plot(t_h, chief["r"][:, 1], "g-", label="y")
    ax.plot(t_h, chief["r"][:, 2], "b-", label="z")
    ax.set_xlabel("time [h]"); ax.set_ylabel("position [km]")
    ax.set_title("Chief position (ICRF)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    # (1,1) Chief velocity components
    ax = plt.subplot2grid((3, 3), (1, 1), fig=fig)
    ax.plot(t_h, chief["v"][:, 0], "r-", label=r"$v_x$")
    ax.plot(t_h, chief["v"][:, 1], "g-", label=r"$v_y$")
    ax.plot(t_h, chief["v"][:, 2], "b-", label=r"$v_z$")
    ax.set_xlabel("time [h]"); ax.set_ylabel("velocity [km/s]")
    ax.set_title("Chief velocity (ICRF)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    # (1,2) Chief attitude quaternion components
    # Plotted directly (no Euler decomposition) to avoid the gimbal-lock
    # artifacts that show up in ZYX Euler when the body rotates through
    # θ = ±π/2. Quaternions are smooth on SO(3).
    ax = plt.subplot2grid((3, 3), (1, 2), fig=fig)
    ax.plot(t_h, chief["q"][:, 0], color="#000000", lw=0.9, label=r"$q_0$ (scalar)")
    ax.plot(t_h, chief["q"][:, 1], color="#D62728", lw=0.9, label=r"$q_1$")
    ax.plot(t_h, chief["q"][:, 2], color="#2CA02C", lw=0.9, label=r"$q_2$")
    ax.plot(t_h, chief["q"][:, 3], color="#1F77B4", lw=0.9, label=r"$q_3$")
    ax.axhline(0.0, color="gray", lw=0.4, alpha=0.5)
    ax.set_xlabel("time [h]"); ax.set_ylabel("quaternion component")
    ax.set_title(r"Chief attitude quaternion $q_I^B$ components")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)

    # (2,0) Per-deputy δr components vs time (in chief-centered ICRF)
    ax = plt.subplot2grid((3, 3), (2, 0), fig=fig)
    for k, d in enumerate(deputies, start=1):
        dr_m = d["dr"] * 1e3
        ax.plot(t_h, dr_m[:, 0], color=_sc_color(k), lw=0.9, ls="-",
                label=f"{spacecraft[k].label} δx" if k == 1 else None)
        ax.plot(t_h, dr_m[:, 1], color=_sc_color(k), lw=0.9, ls="--")
        ax.plot(t_h, dr_m[:, 2], color=_sc_color(k), lw=0.9, ls=":")
    ax.set_xlabel("time [h]"); ax.set_ylabel(r"$\delta r$ components [m]")
    ax.set_title("Deputy δr (chief-centered ICRF): solid x, dashed y, dotted z")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=7)

    # (2,1) All spacecraft angular velocity magnitudes
    ax = plt.subplot2grid((3, 3), (2, 1), fig=fig)
    for i in range(N):
        w_mag = np.linalg.norm(_deputy_state(X_hist, i)["w"] if i > 0 else chief["w"], axis=1)
        ax.plot(t_h, w_mag, color=_sc_color(i), lw=1.2,
                label=spacecraft[i].label)
    ax.set_xlabel("time [h]"); ax.set_ylabel(r"$|\omega|$ [rad/s]")
    ax.set_title("Angular velocity magnitude (per spacecraft)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    # (2,2) Chief conservation diagnostics
    ax = plt.subplot2grid((3, 3), (2, 2), fig=fig)
    eps    = 1e-30
    q_norm = np.linalg.norm(chief["q"], axis=1)
    J_B    = spacecraft[0].J_B
    T_rot  = 0.5 * np.einsum("ni,ij,nj->n", chief["w"], J_B, chief["w"])
    L_mag  = np.linalg.norm((J_B @ chief["w"].T).T, axis=1)

    qn_dev = np.abs(q_norm - 1.0) + eps
    T_dev  = (np.abs((T_rot - T_rot[0]) / T_rot[0]) + eps
              if T_rot[0] != 0 else np.full_like(T_rot, eps))
    L_dev  = (np.abs((L_mag - L_mag[0]) / L_mag[0]) + eps
              if L_mag[0] != 0 else np.full_like(L_mag, eps))

    ax.semilogy(t_h, qn_dev, label=r"$||q| - 1|$")
    ax.semilogy(t_h, T_dev,  label=r"$|\Delta T_{rot}/T_0|$")
    ax.semilogy(t_h, L_dev,  label=r"$|\Delta |L|/|L_0||$")
    ax.set_xlabel("time [h]"); ax.set_ylabel("relative deviation")
    ax.set_title("Chief conservation diagnostics")
    ax.set_ylim(1e-17, 1e-8)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()


# ── Plot 2: Inner solar system (N spacecraft) ────────────────────────────────

_BODY_STYLE = {
    "SUN":     dict(color="#FFD24A", marker_size=260, trail_color="#FFB000", label="Sun"),
    "MERCURY": dict(color="#A9A9A9", marker_size=28,  trail_color="#888888", label="Mercury"),
    "VENUS":   dict(color="#E8B16D", marker_size=45,  trail_color="#C0824A", label="Venus"),
    "EARTH":   dict(color="#3A8DDE", marker_size=48,  trail_color="#1F5FA8", label="Earth"),
    "MOON":    dict(color="#D9D9D9", marker_size=20,  trail_color="#A0A0A0", label="Moon"),
}


def plot_solar_system(
    et0:             float,
    duration:        float,
    X_hist:          np.ndarray,
    t_hist:          np.ndarray,
    spacecraft:      list,
    n_trail_samples: int = 200,
    n_stars:         int = 800,
) -> None:
    """
    Inner solar system in a dark, slightly-tilted 3D view, with all N
    spacecraft. At solar-system scale the spacecraft tracks lie on top of
    each other (formation baseline ~100 m vs scene ~1.5e8 km), so we draw
    them slightly offset visually but they all share the same chief track.
    """
    N = len(spacecraft)

    # Body trails.
    t_samples = np.linspace(0.0, duration, n_trail_samples)
    body_keys = ("SUN", "MERCURY", "VENUS", "EARTH", "MOON")
    trails: dict[str, np.ndarray] = {}
    for body in body_keys:
        spice_name = body if body in ("SUN", "EARTH", "MOON") else f"{body} BARYCENTER"
        trail = np.array([
            spice.spkezr(spice_name, et0 + tk, "J2000", "NONE", "SSB")[0][:3]
            for tk in t_samples
        ])
        trails[body] = trail

    # Figure + dark style.
    fig = plt.figure(figsize=(14, 10), facecolor="black", dpi=150)
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("black")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor((0, 0, 0, 0))
        axis.line.set_color((1, 1, 1, 0.15))
        axis.label.set_color("white")
        for tick in axis.get_ticklabels():
            tick.set_color((1, 1, 1, 0.4))
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    # Background stars.
    rng = np.random.default_rng(seed=42)
    R_view = max(np.linalg.norm(trails["VENUS"][0]), 1.5e8) * 3.0
    phi   = rng.uniform(0.0, 2.0 * np.pi, n_stars)
    costh = rng.uniform(-1.0, 1.0, n_stars)
    sinth = np.sqrt(1.0 - costh ** 2)
    sx = R_view * sinth * np.cos(phi)
    sy = R_view * sinth * np.sin(phi)
    sz = R_view * costh
    star_brightness = rng.uniform(0.3, 1.0, n_stars)
    star_size       = rng.uniform(0.3, 2.0, n_stars)
    ax.scatter(sx, sy, sz, c=[(b, b, b) for b in star_brightness],
               s=star_size, marker=".", depthshade=False)

    # Body trails and final positions.
    for body in body_keys:
        style = _BODY_STYLE[body]
        trail = trails[body]
        ax.plot(trail[:, 0], trail[:, 1], trail[:, 2],
                color=style["trail_color"], lw=1.0, alpha=0.55)
        ax.scatter([trail[-1, 0]], [trail[-1, 1]], [trail[-1, 2]],
                   s=style["marker_size"], color=style["color"],
                   edgecolors="white", linewidths=0.6, label=style["label"],
                   depthshade=False, zorder=10)
        ax.scatter([trail[0, 0]], [trail[0, 1]], [trail[0, 2]],
                   s=0.35 * style["marker_size"], color=style["color"],
                   edgecolors="none", alpha=0.35, depthshade=False, zorder=8)

    # Sun glow.
    sun_pos = trails["SUN"][-1]
    for r_scale, alpha in [(2.0, 0.25), (3.5, 0.12), (6.0, 0.05)]:
        ax.scatter([sun_pos[0]], [sun_pos[1]], [sun_pos[2]],
                   s=_BODY_STYLE["SUN"]["marker_size"] * r_scale,
                   color="#FFD24A", alpha=alpha, edgecolors="none",
                   depthshade=False, zorder=9)

    # All spacecraft. At AU scale they overlap; we draw them with slightly
    # different alpha so the chief is the "main" track and deputies are visible
    # as faint companions. The legend identifies them by color.
    for i in range(N):
        r_abs = _absolute_position(X_hist, i)
        color = _sc_color(i) if i > 0 else "#FF4D6D"
        lw    = 1.6 if i == 0 else 0.9
        alpha = 0.95 if i == 0 else 0.7
        ax.plot(r_abs[:, 0], r_abs[:, 1], r_abs[:, 2],
                color=color, lw=lw, alpha=alpha, label=spacecraft[i].label)
        ax.scatter([r_abs[-1, 0]], [r_abs[-1, 1]], [r_abs[-1, 2]],
                   s=30 if i == 0 else 18, color=color,
                   edgecolors="white", linewidths=0.4, depthshade=False, zorder=12)

    # Framing.
    R_frame = np.linalg.norm(trails["VENUS"][0]) * 1.25
    ax.set_xlim(-R_frame, R_frame); ax.set_ylim(-R_frame, R_frame)
    ax.set_zlim(-R_frame * 0.6, R_frame * 0.6)
    ax.set_box_aspect((1, 1, 0.6))
    ax.view_init(elev=15.0, azim=-60.0)

    ax.set_title(
        f"Inner Solar System — {spice.et2utc(et0, 'ISOC', 0)}  "
        f"(+{duration/86400:.2f} d) — {N} spacecraft (formation baseline "
        "≪ scene, tracks overlap visually)",
        color="white", fontsize=12, pad=15,
    )
    leg = ax.legend(loc="upper left", fontsize=9, frameon=True,
                    facecolor=(0, 0, 0, 0.6), edgecolor=(1, 1, 1, 0.2),
                    labelcolor="white")
    for txt in leg.get_texts():
        txt.set_color("white")

    fig.tight_layout()
    plt.show()


# ── Plot 3: Sun-Earth L2 rotating-frame zoom (N spacecraft) ──────────────────

def plot_l2_rotating_frame_zoom(
    et0:        float,
    t_hist:     np.ndarray,
    X_hist:     np.ndarray,
    spacecraft: list,
    l2_dist_km: float = L2_DIST_KM,
) -> None:
    """
    Plot all N spacecraft, plus Earth and Moon, in the Sun-Earth L2
    rotating frame. At L2-zoom scale the formation (~100 m) is still
    much smaller than the L2 halo (~1e5 km), so the spacecraft tracks
    will overlap visually; each gets its own color in the legend.
    """
    N        = len(spacecraft)
    n_t      = len(t_hist)
    sc_rel   = [np.zeros((n_t, 3)) for _ in range(N)]
    earth_rel = np.zeros((n_t, 3))
    moon_rel  = np.zeros((n_t, 3))

    for k, tk in enumerate(t_hist):
        et = et0 + tk
        frame = sun_earth_l2_frame(et, l2_dist_km=l2_dist_km)
        r_l2 = frame["r_l2"]
        basis = np.vstack([frame["e_x"], frame["e_y"], frame["e_z"]])

        r_earth = spice.spkezr("EARTH", et, "J2000", "NONE", "SSB")[0][:3]
        r_moon  = spice.spkezr("MOON",  et, "J2000", "NONE", "SSB")[0][:3]
        earth_rel[k, :] = basis @ (r_earth - r_l2)
        moon_rel[k, :]  = basis @ (r_moon  - r_l2)

        for i in range(N):
            r_abs_i = _absolute_position(X_hist[k:k+1, :], i)[0]
            sc_rel[i][k, :] = basis @ (r_abs_i - r_l2)

    # Figure / dark style.
    fig = plt.figure(figsize=(10, 8), facecolor="black", dpi=150)
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("black")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor((0, 0, 0, 0))
        axis.line.set_color((1, 1, 1, 0.2))
        axis.label.set_color("white")
        for tick in axis.get_ticklabels():
            tick.set_color((1, 1, 1, 0.65))
    ax.grid(True, alpha=0.15)

    # L2 point.
    ax.scatter([0], [0], [0], s=55, color="#FFFFFF", edgecolors="#000000",
               linewidths=0.5, label="approx. L2", depthshade=False)

    # Earth and Moon.
    ax.plot(earth_rel[:, 0], earth_rel[:, 1], earth_rel[:, 2],
            color="#3A8DDE", lw=1.2, alpha=0.7, label="Earth")
    ax.scatter([earth_rel[-1, 0]], [earth_rel[-1, 1]], [earth_rel[-1, 2]],
               s=65, color="#3A8DDE", edgecolors="white", linewidths=0.5,
               depthshade=False)
    ax.plot(moon_rel[:, 0], moon_rel[:, 1], moon_rel[:, 2],
            color="#D9D9D9", lw=0.9, alpha=0.85, label="Moon")
    ax.scatter([moon_rel[-1, 0]], [moon_rel[-1, 1]], [moon_rel[-1, 2]],
               s=28, color="#D9D9D9", edgecolors="white", linewidths=0.4,
               depthshade=False)

    # All spacecraft.
    for i in range(N):
        color = _sc_color(i) if i > 0 else "#FF4D6D"
        lw    = 1.7 if i == 0 else 1.0
        alpha = 0.95 if i == 0 else 0.75
        ax.plot(sc_rel[i][:, 0], sc_rel[i][:, 1], sc_rel[i][:, 2],
                color=color, lw=lw, alpha=alpha, label=spacecraft[i].label)
        ax.scatter([sc_rel[i][0, 0]], [sc_rel[i][0, 1]], [sc_rel[i][0, 2]],
                   s=24, color=color, edgecolors="white", linewidths=0.4,
                   depthshade=False)
        ax.scatter([sc_rel[i][-1, 0]], [sc_rel[i][-1, 1]], [sc_rel[i][-1, 2]],
                   s=36, color=color, edgecolors="white", linewidths=0.5,
                   depthshade=False)

    ax.set_xlabel("x from L2 [km]")
    ax.set_ylabel("y from L2 [km]")
    ax.set_zlabel("z from L2 [km]")
    ax.set_title("Sun-Earth L2 Rotating-Frame Zoom — N spacecraft",
                 color="white", fontsize=13, pad=12)

    # Framing.
    pts = np.vstack([*sc_rel, earth_rel, moon_rel, np.zeros((1, 3))])
    pmin = pts.min(axis=0); pmax = pts.max(axis=0)
    ctr  = 0.5 * (pmin + pmax); half = 0.60 * max(pmax - pmin)
    ax.set_xlim(ctr[0] - half, ctr[0] + half)
    ax.set_ylim(ctr[1] - half, ctr[1] + half)
    ax.set_zlim(ctr[2] - half, ctr[2] + half)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=22.0, azim=-55.0)

    leg = ax.legend(loc="upper left", fontsize=9, frameon=True,
                    facecolor=(0, 0, 0, 0.6), edgecolor=(1, 1, 1, 0.2),
                    labelcolor="white")
    for txt in leg.get_texts():
        txt.set_color("white")

    fig.tight_layout()
    plt.show()