"""
Plotting utilities for the LIFE truth-model simulation.

This module collects the three diagnostic plots used in main.py:
    - plot_trajectory             : 3x3 summary panel for a propagation
    - plot_solar_system           : dark, slightly-tilted inner solar system view
    - plot_l2_rotating_frame_zoom : Sun-Earth L2 rotating-frame zoom

It also contains a small Sun-Earth L2 rotating-frame helper
(sun_earth_l2_frame + L2_DIST_KM) used only by the third plot.

Attitude in plot_trajectory is shown as Euler angles (roll/pitch/yaw,
Groves convention, in degrees) rather than raw quaternion components,
since Euler angles are much easier to interpret at a glance. Angles
are unwrapped along the trajectory so that a spinning spacecraft shows
a continuous ramp instead of saw-tooth jumps at +/- 180 deg.

The plots query SPICE via spkezr, so the caller must have already furnished
the required kernels (typically via load_kernels() in main.py) before any
of these functions are called.
"""

import numpy as np
import spiceypy as spice

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

# Attitude conversion (Groves quaternion -> Euler angles, eq. 2.37).
from utils.coordinate_trafos.Quat_to_Euler import quat_to_euler


# ── Sun-Earth L2 / rotating-frame helper ─────────────────────────────────────

L2_DIST_KM = 1.5e6


def sun_earth_l2_frame(
    et: float,
    l2_dist_km: float = L2_DIST_KM,
) -> dict[str, np.ndarray]:
    """
    Build an instantaneous Sun-Earth rotating-frame triad at epoch et.

    e_x : from Sun toward Earth, pointing outward past Earth toward L2
    e_z : normal to the instantaneous Sun-Earth orbital plane
    e_y : completes right-handed frame, approximately Earth's prograde direction

    Returns:
        r_l2      : approximate Sun-Earth L2 position in ICRF [km]
        v_l2      : inertial velocity of the co-rotating L2 point [km/s]
        e_x,e_y,e_z
        omega_vec : instantaneous Sun-Earth angular velocity vector [rad/s]
    """
    state_earth = spice.spkezr("EARTH", et, "J2000", "NONE", "SSB")[0]
    state_sun   = spice.spkezr("SUN",   et, "J2000", "NONE", "SSB")[0]

    r_earth = state_earth[:3]
    v_earth = state_earth[3:]

    r_sun = state_sun[:3]
    v_sun = state_sun[3:]

    r_es = r_earth - r_sun
    v_es = v_earth - v_sun

    e_x = r_es / np.linalg.norm(r_es)

    h_es = np.cross(r_es, v_es)
    e_z = h_es / np.linalg.norm(h_es)

    e_y = np.cross(e_z, e_x)
    e_y = e_y / np.linalg.norm(e_y)

    # Instantaneous angular velocity of the Sun-Earth line.
    omega_vec = h_es / np.dot(r_es, r_es)

    r_l2 = r_earth + l2_dist_km * e_x

    # Co-rotating inertial velocity of the approximate L2 point.
    # This is the important correction compared with simply using Earth's velocity.
    v_l2 = v_sun + np.cross(omega_vec, r_l2 - r_sun)

    return {
        "r_l2": r_l2,
        "v_l2": v_l2,
        "e_x": e_x,
        "e_y": e_y,
        "e_z": e_z,
        "omega_vec": omega_vec,
    }


# ── Plot 1: 3x3 trajectory summary ───────────────────────────────────────────

def plot_trajectory(
    t_hist: np.ndarray,
    X_hist: np.ndarray,
    et0:    float,
    J_B:    np.ndarray,
) -> None:
    """
    Build a 3x3 figure summarizing one propagation.

    Panels:
        (0,0)  3D trajectory in ICRF with Sun and Earth (full)
        (0,1)  3D trajectory zoomed to Earth + spacecraft
        (0,2)  Position components vs time
        (1,0)  Velocity components vs time
        (1,1)  Euler angles (roll/pitch/yaw, unwrapped) vs time, full window
        (1,2)  Euler angles, zoomed to first ~3 spin periods
        (2,0)  Angular velocity (body frame) vs time
        (2,1)  Conservation diagnostics (log scale)
        (2,2)  Mass vs time

    Args:
        t_hist : (N,)     time since et0 [s]
        X_hist : (N, 14)  state history
        et0    : initial ET (used to query Earth/Sun in ICRF)
        J_B    : 3x3 spacecraft inertia tensor in body frame [kg m^2],
                 used for the rotational-energy / angular-momentum diagnostics.
    """
    # Convenience slices.
    r = X_hist[:, 0:3]       # km
    v = X_hist[:, 3:6]       # km/s
    q = X_hist[:, 6:10]
    w = X_hist[:, 10:13]     # rad/s

    t_h = t_hist / 3600.0    # hours, readable axis labels

    # --- Quaternion -> Euler angles (Groves convention) ------------------
    # Per-sample conversion to (phi, theta, psi) = (roll, pitch, yaw) in
    # radians. We then unwrap along the trajectory so that a spinning axis
    # produces a continuous ramp instead of saw-tooth jumps at +/- pi, and
    # convert to degrees for display.
    #
    # Note: at theta = +/- pi/2 the Groves arctan2 formulas in eq. (2.37)
    # are degenerate (only phi -/+ psi is determined). For the JWST-like
    # demo initial conditions (small body-x spin) theta stays near zero,
    # so this is not an issue here.
    euler_rad = np.array([quat_to_euler(qk) for qk in q])         # (N, 3)
    euler_rad = np.unwrap(euler_rad, axis=0)
    euler_deg = np.degrees(euler_rad)                              # (N, 3)


    # --- Sample reference body positions over the time window ------------
    n_ref = 200
    t_ref = np.linspace(t_hist[0], t_hist[-1], n_ref)
    r_earth_hist = np.array([
        spice.spkezr("EARTH", et0 + tk, "J2000", "NONE", "SSB")[0][:3]
        for tk in t_ref
    ])
    r_sun_pos = spice.spkezr("SUN", et0, "J2000", "NONE", "SSB")[0][:3]

    # --- Figure ----------------------------------------------------------
    fig = plt.figure(figsize=(18, 11))
    fig.suptitle(
        "LIFE Truth Model — Single-Spacecraft Propagation",
        fontsize=14, fontweight="bold",
    )

    # (0, 0) 3D trajectory in ICRF — full view including the Sun
    ax3d = plt.subplot2grid((3, 3), (0, 0), projection="3d", fig=fig)
    ax3d.plot(r[:, 0], r[:, 1], r[:, 2], "b-", lw=1.5, label="spacecraft")
    ax3d.plot(r_earth_hist[:, 0], r_earth_hist[:, 1], r_earth_hist[:, 2],
              "g-", lw=1.0, alpha=0.7, label="Earth")
    ax3d.scatter([r_sun_pos[0]], [r_sun_pos[1]], [r_sun_pos[2]],
                 color="orange", s=140, marker="*", label="Sun")
    ax3d.scatter([r[0, 0]], [r[0, 1]], [r[0, 2]],
                 color="blue", s=40, marker="o", label="start")
    ax3d.scatter([r[-1, 0]], [r[-1, 1]], [r[-1, 2]],
                 color="red", s=40, marker="x", label="end")
    ax3d.set_xlabel("x [km]")
    ax3d.set_ylabel("y [km]")
    ax3d.set_zlabel("z [km]")
    ax3d.set_title("Trajectory in ICRF (full, with Sun)")
    ax3d.legend(loc="upper left", fontsize=7)

    # (0, 1) 3D trajectory zoomed to Earth + spacecraft only
    ax3z = plt.subplot2grid((3, 3), (0, 1), projection="3d", fig=fig)
    ax3z.plot(r[:, 0], r[:, 1], r[:, 2], "b-", lw=1.5, label="spacecraft")
    ax3z.plot(r_earth_hist[:, 0], r_earth_hist[:, 1], r_earth_hist[:, 2],
              "g-", lw=1.5, alpha=0.85, label="Earth")
    ax3z.scatter([r[0, 0]], [r[0, 1]], [r[0, 2]],
                 color="blue", s=40, marker="o", label="s/c start")
    ax3z.scatter([r[-1, 0]], [r[-1, 1]], [r[-1, 2]],
                 color="red", s=40, marker="x", label="s/c end")
    ax3z.scatter([r_earth_hist[0, 0]],  [r_earth_hist[0, 1]],  [r_earth_hist[0, 2]],
                 color="darkgreen", s=40, marker="o", label="Earth start")
    ax3z.scatter([r_earth_hist[-1, 0]], [r_earth_hist[-1, 1]], [r_earth_hist[-1, 2]],
                 color="darkgreen", s=40, marker="x", label="Earth end")

    # Tight bounding box around both tracks, with a small margin.
    pts = np.vstack([r, r_earth_hist])
    pmin, pmax = pts.min(axis=0), pts.max(axis=0)
    ctr  = 0.5 * (pmin + pmax)
    half = 0.55 * max(pmax - pmin)                 # cubic box, 10% margin
    ax3z.set_xlim(ctr[0] - half, ctr[0] + half)
    ax3z.set_ylim(ctr[1] - half, ctr[1] + half)
    ax3z.set_zlim(ctr[2] - half, ctr[2] + half)
    ax3z.set_xlabel("x [km]")
    ax3z.set_ylabel("y [km]")
    ax3z.set_zlabel("z [km]")
    ax3z.set_title("Trajectory zoom: Earth + spacecraft (ICRF)")
    ax3z.legend(loc="upper left", fontsize=7)

    # (0, 2) Position components
    ax_r = plt.subplot2grid((3, 3), (0, 2), fig=fig)
    ax_r.plot(t_h, r[:, 0], "r-", label="x")
    ax_r.plot(t_h, r[:, 1], "g-", label="y")
    ax_r.plot(t_h, r[:, 2], "b-", label="z")
    ax_r.set_xlabel("time [h]")
    ax_r.set_ylabel("position [km]")
    ax_r.set_title("Position components (ICRF)")
    ax_r.grid(True, alpha=0.3)
    ax_r.legend(loc="best", fontsize=9)

    # (1, 0) Velocity components
    ax_v = plt.subplot2grid((3, 3), (1, 0), fig=fig)
    ax_v.plot(t_h, v[:, 0], "r-", label="$v_x$")
    ax_v.plot(t_h, v[:, 1], "g-", label="$v_y$")
    ax_v.plot(t_h, v[:, 2], "b-", label="$v_z$")
    ax_v.set_xlabel("time [h]")
    ax_v.set_ylabel("velocity [km/s]")
    ax_v.set_title("Velocity components (ICRF)")
    ax_v.grid(True, alpha=0.3)
    ax_v.legend(loc="best", fontsize=9)

    # (1, 1) Euler angles, full window
    ax_e = plt.subplot2grid((3, 3), (1, 1), fig=fig)
    ax_e.plot(t_h, euler_deg[:, 0], "r-", lw=0.9, label=r"$\phi$ (roll)")
    ax_e.plot(t_h, euler_deg[:, 1], "g-", lw=0.9, label=r"$\theta$ (pitch)")
    ax_e.plot(t_h, euler_deg[:, 2], "b-", lw=0.9, label=r"$\psi$ (yaw)")
    ax_e.set_xlabel("time [h]")
    ax_e.set_ylabel("angle [deg]")
    ax_e.set_title("Euler angles $\\phi,\\theta,\\psi$ (full window, unwrapped)")
    ax_e.grid(True, alpha=0.3)
    ax_e.legend(loc="best", fontsize=8)

    # (1, 2) Euler angles, zoomed to first ~3 spin periods
    w0_mag    = np.linalg.norm(w[0]) if np.linalg.norm(w[0]) > 0 else 1.0
    t_zoom_s  = 3.0 * 2.0 * np.pi / w0_mag                 # 3 periods [s]
    t_zoom_h  = min(t_zoom_s / 3600.0, t_h[-1])            # cap at full window

    # Restrict y-range to what is actually visible in the zoom window so the
    # zoom is informative even when the full-window range is huge.
    zoom_mask = t_h <= t_zoom_h
    if zoom_mask.any():
        e_zoom_min = euler_deg[zoom_mask, :].min()
        e_zoom_max = euler_deg[zoom_mask, :].max()
        pad = max(5.0, 0.05 * (e_zoom_max - e_zoom_min))
    else:
        e_zoom_min, e_zoom_max, pad = -180.0, 180.0, 0.0

    ax_ez = plt.subplot2grid((3, 3), (1, 2), fig=fig)
    ax_ez.plot(t_h, euler_deg[:, 0], "r-", label=r"$\phi$ (roll)")
    ax_ez.plot(t_h, euler_deg[:, 1], "g-", label=r"$\theta$ (pitch)")
    ax_ez.plot(t_h, euler_deg[:, 2], "b-", label=r"$\psi$ (yaw)")
    ax_ez.set_xlim(0, t_zoom_h)
    ax_ez.set_ylim(e_zoom_min - pad, e_zoom_max + pad)
    ax_ez.set_xlabel("time [h]")
    ax_ez.set_ylabel("angle [deg]")
    ax_ez.set_title(f"Euler angles (zoom: first {t_zoom_h*60:.1f} min)")
    ax_ez.grid(True, alpha=0.3)
    ax_ez.legend(loc="best", fontsize=8)

    # (2, 0) Angular velocity in body frame
    ax_w = plt.subplot2grid((3, 3), (2, 0), fig=fig)
    ax_w.plot(t_h, w[:, 0], "r-", label=r"$\omega_x^B$")
    ax_w.plot(t_h, w[:, 1], "g-", label=r"$\omega_y^B$")
    ax_w.plot(t_h, w[:, 2], "b-", label=r"$\omega_z^B$")
    ax_w.set_xlabel("time [h]")
    ax_w.set_ylabel(r"$\omega$ [rad/s]")
    ax_w.set_title("Angular velocity (body frame)")
    ax_w.grid(True, alpha=0.3)
    ax_w.legend(loc="best", fontsize=9)

    # (2, 1) Conservation diagnostics, log scale
    eps    = 1e-30
    q_norm = np.linalg.norm(q, axis=1)
    T_rot  = 0.5 * np.einsum("ni,ij,nj->n", w, J_B, w)         # 1/2 w^T J w
    L_mag  = np.linalg.norm((J_B @ w.T).T, axis=1)             # |J w|

    qn_dev = np.abs(q_norm - 1.0) + eps
    T_dev  = (np.abs((T_rot - T_rot[0]) / T_rot[0]) + eps
              if T_rot[0] != 0 else np.full_like(T_rot, eps))
    L_dev  = (np.abs((L_mag - L_mag[0]) / L_mag[0]) + eps
              if L_mag[0] != 0 else np.full_like(L_mag, eps))

    ax_san = plt.subplot2grid((3, 3), (2, 1), fig=fig)
    ax_san.semilogy(t_h, qn_dev, label=r"$||q| - 1|$")
    ax_san.semilogy(t_h, T_dev,  label=r"$|\Delta T_{rot} / T_0|$")
    ax_san.semilogy(t_h, L_dev,  label=r"$|\Delta |L| / |L_0||$")
    ax_san.set_xlabel("time [h]")
    ax_san.set_ylabel("relative deviation")
    ax_san.set_title("Conservation diagnostics")
    ax_san.set_ylim(1e-17, 1e-8)                     # clamp to meaningful range
    ax_san.grid(True, alpha=0.3, which="both")
    ax_san.legend(loc="best", fontsize=8)

    # (2, 2) Mass over time (currently constant; placeholder for thrust-on case)
    ax_m = plt.subplot2grid((3, 3), (2, 2), fig=fig)
    ax_m.plot(t_h, X_hist[:, 13], "k-")
    ax_m.set_xlabel("time [h]")
    ax_m.set_ylabel("mass [kg]")
    ax_m.set_title("Spacecraft mass")
    ax_m.grid(True, alpha=0.3)
    # Give a visible y-range even when mass is exactly constant.
    m_mid = float(X_hist[0, 13])
    ax_m.set_ylim(m_mid - 1.0, m_mid + 1.0)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()


# ── Plot 2: Inner solar system beauty plot ───────────────────────────────────

# Body colors and (relative) display radii. Display radii are NOT physical —
# physical radii (Sun ~7e5 km, Mercury ~2440 km) are invisible at a scale that
# includes Mercury's orbit, so dots are drawn at exaggerated sizes for visibility.
# Orbital positions remain exactly to scale via SPICE.
_BODY_STYLE = {
    "SUN":     dict(color="#FFD24A", marker_size=260, trail_color="#FFB000", label="Sun"),
    "MERCURY": dict(color="#A9A9A9", marker_size=28,  trail_color="#888888", label="Mercury"),
    "VENUS":   dict(color="#E8B16D", marker_size=45,  trail_color="#C0824A", label="Venus"),
    "EARTH":   dict(color="#3A8DDE", marker_size=48,  trail_color="#1F5FA8", label="Earth"),
    "MOON":    dict(color="#D9D9D9", marker_size=20,  trail_color="#A0A0A0", label="Moon"),
}


def plot_solar_system(
    et0:       float,
    duration:  float,
    spacecraft_r:    np.ndarray | None = None,
    spacecraft_t:    np.ndarray | None = None,
    n_trail_samples: int = 200,
    n_stars:         int = 800,
) -> None:
    """
    Render the inner solar system in a dark, slightly-tilted 3D view.

    Bodies drawn: Sun, Mercury, Venus, Earth, Moon (and the spacecraft if
    provided). Distances are exactly to scale; body marker sizes are
    exaggerated so the smaller bodies are visible alongside the Sun.

    Args:
        et0          : initial SPICE ET [s past J2000 TDB]
        duration     : window length [s] (typically same as the sim)
        spacecraft_r : optional spacecraft position history (N, 3) in ICRF [km]
        spacecraft_t : optional spacecraft time vector (N,) in seconds since et0
        n_trail_samples : how many SPICE samples to draw each body's trail with
        n_stars      : number of background stars
    """
    # --- Sample body positions along the window ---------------------------
    t_samples = np.linspace(0.0, duration, n_trail_samples)
    body_keys = ("SUN", "MERCURY", "VENUS", "EARTH", "MOON")

    trails: dict[str, np.ndarray] = {}
    for body in body_keys:
        # SPICE name lookup: planets stored as "X BARYCENTER" or directly.
        spice_name = body if body in ("SUN", "EARTH", "MOON") else f"{body} BARYCENTER"
        trail = np.array([
            spice.spkezr(spice_name, et0 + tk, "J2000", "NONE", "SSB")[0][:3]
            for tk in t_samples
        ])
        trails[body] = trail

    # --- Figure & dark style ---------------------------------------------
    fig = plt.figure(figsize=(14, 10), facecolor="black", dpi=150)
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("black")

    # Hide panes & axes for a clean space look.
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor((0, 0, 0, 0))
        axis.line.set_color((1, 1, 1, 0.15))
        axis.label.set_color("white")
        for tick in axis.get_ticklabels():
            tick.set_color((1, 1, 1, 0.4))
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    # --- Background stars (random, in a large sphere around the view) ----
    rng = np.random.default_rng(seed=42)
    # Place stars on a large bounding sphere ~3x the outermost orbit.
    R_view = max(np.linalg.norm(trails["VENUS"][0]), 1.5e8) * 3.0
    phi   = rng.uniform(0.0, 2.0 * np.pi, n_stars)
    costh = rng.uniform(-1.0, 1.0, n_stars)
    sinth = np.sqrt(1.0 - costh ** 2)
    sx = R_view * sinth * np.cos(phi)
    sy = R_view * sinth * np.sin(phi)
    sz = R_view * costh
    star_brightness = rng.uniform(0.3, 1.0, n_stars)
    star_size       = rng.uniform(0.3, 2.0, n_stars)
    ax.scatter(sx, sy, sz,
               c=[(b, b, b) for b in star_brightness],
               s=star_size, marker=".", depthshade=False)

    # --- Body trails and current positions -------------------------------
    for body in body_keys:
        style = _BODY_STYLE[body]
        trail = trails[body]
        # Trail (faint, behind the body).
        ax.plot(trail[:, 0], trail[:, 1], trail[:, 2],
                color=style["trail_color"], lw=1.0, alpha=0.55)
        # Body at final epoch position.
        ax.scatter(
            [trail[-1, 0]], [trail[-1, 1]], [trail[-1, 2]],
            s=style["marker_size"], color=style["color"],
            edgecolors="white", linewidths=0.6,
            label=style["label"], depthshade=False, zorder=10,
        )

        # Optional faint start marker.
        ax.scatter(
            [trail[0, 0]], [trail[0, 1]], [trail[0, 2]],
            s=0.35 * style["marker_size"], color=style["color"],
            edgecolors="none", alpha=0.35,
            depthshade=False, zorder=8,
        )

    # Sun glow: stack of fading concentric markers.
    sun_pos = trails["SUN"][-1]
    for r_scale, alpha in [(2.0, 0.25), (3.5, 0.12), (6.0, 0.05)]:
        ax.scatter(
            [sun_pos[0]], [sun_pos[1]], [sun_pos[2]],
            s=_BODY_STYLE["SUN"]["marker_size"] * r_scale,
            color="#FFD24A", alpha=alpha, edgecolors="none",
            depthshade=False, zorder=9,
        )

    # --- Spacecraft (optional) -------------------------------------------
    if spacecraft_r is not None:
        ax.plot(
            spacecraft_r[:, 0], spacecraft_r[:, 1], spacecraft_r[:, 2],
            color="#FF4D6D", lw=1.4, alpha=0.9, label="spacecraft"
        )

        # Start marker.
        ax.scatter(
            [spacecraft_r[0, 0]], [spacecraft_r[0, 1]], [spacecraft_r[0, 2]],
            s=22, color="#FF4D6D", edgecolors="white", linewidths=0.4,
            depthshade=False, zorder=11,
        )

        # End marker.
        ax.scatter(
            [spacecraft_r[-1, 0]], [spacecraft_r[-1, 1]], [spacecraft_r[-1, 2]],
            s=35, color="#FF4D6D", edgecolors="white", linewidths=0.5,
            depthshade=False, zorder=12,
        )

        # A few small direction markers along the trajectory.
        n_arrows = 8
        idxs = np.linspace(0, len(spacecraft_r) - 1, n_arrows + 2, dtype=int)[1:-1]
        ax.scatter(
            spacecraft_r[idxs, 0], spacecraft_r[idxs, 1], spacecraft_r[idxs, 2],
            s=10, color="#FF8FA3", edgecolors="none",
            depthshade=False, zorder=12,
        )

    # --- Camera & framing -------------------------------------------------
    # Frame to Venus' orbit (outer of the inner bodies of interest).
    R_frame = np.linalg.norm(trails["VENUS"][0]) * 1.25
    ax.set_xlim(-R_frame, R_frame)
    ax.set_ylim(-R_frame, R_frame)
    ax.set_zlim(-R_frame * 0.6, R_frame * 0.6)
    ax.set_box_aspect((1, 1, 0.6))                # keep z compressed for the tilt
    ax.view_init(elev=15.0, azim=-60.0)            # slight tilt above ecliptic

    # --- Title & legend ---------------------------------------------------
    ax.set_title(
        f"Inner Solar System — {spice.et2utc(et0, 'ISOC', 0)}  "
        f"(+{duration/86400:.2f} d)",
        color="white", fontsize=13, pad=15,
    )
    leg = ax.legend(
        loc="upper left", fontsize=9, frameon=True,
        facecolor=(0, 0, 0, 0.6), edgecolor=(1, 1, 1, 0.2),
        labelcolor="white",
    )
    for txt in leg.get_texts():
        txt.set_color("white")

    fig.tight_layout()
    plt.show()


# ── Plot 3: Sun-Earth L2 rotating-frame zoom ─────────────────────────────────

def plot_l2_rotating_frame_zoom(
    et0: float,
    t_hist: np.ndarray,
    spacecraft_r: np.ndarray,
    l2_dist_km: float = L2_DIST_KM,
) -> None:
    """
    Plot spacecraft, Earth, and Moon in a Sun-Earth L2 rotating frame.

    Coordinates:
        origin : instantaneous approximate L2 point
        x      : Sun -> Earth -> L2 direction
        y      : prograde direction
        z      : ecliptic-normal direction

    This is the correct diagnostic plot for checking whether the spacecraft
    is behaving like an L2 halo/Lissajous trajectory.
    """
    sc_rel = np.zeros_like(spacecraft_r)
    earth_rel = np.zeros_like(spacecraft_r)
    moon_rel = np.zeros_like(spacecraft_r)

    for k, tk in enumerate(t_hist):
        et = et0 + tk

        frame = sun_earth_l2_frame(et, l2_dist_km=l2_dist_km)

        r_l2 = frame["r_l2"]
        e_x = frame["e_x"]
        e_y = frame["e_y"]
        e_z = frame["e_z"]

        basis = np.vstack([e_x, e_y, e_z])

        r_earth = spice.spkezr("EARTH", et, "J2000", "NONE", "SSB")[0][:3]
        r_moon  = spice.spkezr("MOON",  et, "J2000", "NONE", "SSB")[0][:3]

        sc_rel[k, :] = basis @ (spacecraft_r[k, :] - r_l2)
        earth_rel[k, :] = basis @ (r_earth - r_l2)
        moon_rel[k, :] = basis @ (r_moon - r_l2)

    fig = plt.figure(figsize=(10, 8), facecolor="black", dpi=150)
    ax = fig.add_subplot(111, projection="3d")
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
    ax.scatter(
        [0], [0], [0],
        s=55, color="#FFFFFF", edgecolors="#000000",
        linewidths=0.5, label="approx. L2", depthshade=False,
    )

    # Earth and Moon relative to L2.
    ax.plot(
        earth_rel[:, 0], earth_rel[:, 1], earth_rel[:, 2],
        color="#3A8DDE", lw=1.2, alpha=0.7, label="Earth"
    )
    ax.scatter(
        [earth_rel[-1, 0]], [earth_rel[-1, 1]], [earth_rel[-1, 2]],
        s=65, color="#3A8DDE", edgecolors="white",
        linewidths=0.5, depthshade=False,
    )

    ax.plot(
        moon_rel[:, 0], moon_rel[:, 1], moon_rel[:, 2],
        color="#D9D9D9", lw=0.9, alpha=0.85, label="Moon"
    )
    ax.scatter(
        [moon_rel[-1, 0]], [moon_rel[-1, 1]], [moon_rel[-1, 2]],
        s=28, color="#D9D9D9", edgecolors="white",
        linewidths=0.4, depthshade=False,
    )

    # Spacecraft relative trajectory.
    ax.plot(
        sc_rel[:, 0], sc_rel[:, 1], sc_rel[:, 2],
        color="#FF4D6D", lw=1.6, alpha=0.95, label="spacecraft"
    )
    ax.scatter(
        [sc_rel[0, 0]], [sc_rel[0, 1]], [sc_rel[0, 2]],
        s=30, color="#FF4D6D", edgecolors="white",
        linewidths=0.4, depthshade=False,
    )
    ax.scatter(
        [sc_rel[-1, 0]], [sc_rel[-1, 1]], [sc_rel[-1, 2]],
        s=45, color="#FF4D6D", edgecolors="white",
        linewidths=0.5, depthshade=False,
    )

    ax.set_xlabel("x from L2 [km]")
    ax.set_ylabel("y from L2 [km]")
    ax.set_zlabel("z from L2 [km]")

    ax.set_title(
        "Sun-Earth L2 Rotating-Frame Zoom",
        color="white", fontsize=13, pad=12,
    )

    # Symmetric-ish box around spacecraft, Earth, Moon, and L2.
    pts = np.vstack([sc_rel, earth_rel, moon_rel, np.zeros((1, 3))])
    pmin = pts.min(axis=0)
    pmax = pts.max(axis=0)
    ctr = 0.5 * (pmin + pmax)
    half = 0.60 * max(pmax - pmin)

    ax.set_xlim(ctr[0] - half, ctr[0] + half)
    ax.set_ylim(ctr[1] - half, ctr[1] + half)
    ax.set_zlim(ctr[2] - half, ctr[2] + half)
    ax.set_box_aspect((1, 1, 1))

    ax.view_init(elev=22.0, azim=-55.0)

    leg = ax.legend(
        loc="upper left", fontsize=9, frameon=True,
        facecolor=(0, 0, 0, 0.6), edgecolor=(1, 1, 1, 0.2),
        labelcolor="white",
    )
    for txt in leg.get_texts():
        txt.set_color("white")

    fig.tight_layout()
    plt.show()
