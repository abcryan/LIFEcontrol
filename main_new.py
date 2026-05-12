"""
LIFE Mission – High-Fidelity Translational + Rotational Dynamics (Truth Model)

Single-spacecraft state (per Eq. (5) of the design doc):
    x_i = [ r_IB^I  (3) ,    position of body B wrt inertial I, resolved in I  [km]
            v_IB^I  (3) ,    velocity of body B wrt inertial I, resolved in I  [km/s]
            q_I^B   (4) ,    attitude quaternion, inertial -> body
            w_IB^B  (3) ,    angular rate of B wrt I, resolved in B            [rad/s]
            m       (1) ]    spacecraft mass                                   [kg]
    -> shape (14,)

Control vector u_i = [a_ctrl^I (3), tau_ctrl^B (3)]  -> shape (6,)
    a_ctrl   : control acceleration, inertial frame   [km/s^2]
    tau_ctrl : control torque,       body frame       [N*m]

Frames:
    I : ICRF J2000, origin at solar system barycenter (SSB)
    B : spacecraft body frame

Time:   SPICE Ephemeris Time (ET) = seconds past J2000 TDB.
Units:  km, km/s, s, kg, rad, rad/s, N, N*m  (SPICE-native for kinematics).

Dynamics per Eq. (6); terms not yet modelled are kept as named zero variables
so the structure of the truth model is explicit in the code.
"""
import sys
sys.dont_write_bytecode = True

import numpy as np
import spiceypy as spice
from scipy.integrate import solve_ivp
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

# Quaternion kinematics matrix Omega(omega) such that q_dot = 0.5 * Omega * q.
# (Already implemented in your utils.)
from utils.other.Omega_omega import Omega_omega


# ── SPICE kernels ─────────────────────────────────────────────────────────────

KERNELS = [
    "data/spice_kernels/naif0012.tls",
    "data/spice_kernels/de440.bsp",
    "data/spice_kernels/gm_de440.tpc",
]


def _check_kernels_exist() -> bool:
    return all(Path(k).exists() for k in KERNELS)


# ── Gravitational parameters [km^3/s^2] ──────────────────────────────────────

BODIES: tuple[str, ...] = (
    "SUN",
    "MERCURY BARYCENTER",
    "VENUS BARYCENTER",
    "EARTH",
    "MOON",
    "MARS BARYCENTER",
    "JUPITER BARYCENTER",
    "SATURN BARYCENTER",
    "URANUS BARYCENTER",
    "NEPTUNE BARYCENTER",
)

GM: dict[str, float] = {}   # populated by load_kernels()


def load_kernels() -> None:
    """Furnish SPICE kernels and read GM values from the PCK."""
    if not _check_kernels_exist():
        raise FileNotFoundError(
            "\n" + "=" * 70 + "\n"
            "SPICE kernels not found!\n\n"
            "Required kernels are missing from data/spice_kernels/\n"
            "Download them automatically using:\n\n"
            "  pip install -e .\n"
            "  lifecontrol-setup-kernels\n\n"
            "Or manually from NAIF:\n"
            "  https://naif.jpl.nasa.gov/pub/naif/generic_kernels/\n"
            "=" * 70
        )
    for k in KERNELS:
        spice.furnsh(k)
    GM.clear()
    for body in BODIES:
        GM[body] = float(spice.bodvrd(body, "GM", 1)[1][0])


# ── Spacecraft constant parameters (single spacecraft, for the moment) ───────
# These will be made per-spacecraft once multi-spacecraft support is added.

# Inertia tensor in body frame [kg * m^2]. Slightly off-diagonal so that an
# x-axis spin couples into the other body axes (visible in the plots).
J_B = np.diag([100.0, 100.0, 100.0])
J_B[1, 2] = 10.0
J_B[2, 1] = 10.0   # mirror to keep J symmetric
J_B_inv: np.ndarray = np.linalg.inv(J_B)

# Inertia rate dJ/dt in body frame [kg * m^2 / s].
# Per Eq. (16) this is driven by mass depletion: J_dot = m_dot * K_r.
# With zero control (no thrust), m_dot = 0, so J_dot = 0.
J_B_dot: np.ndarray = np.zeros((3, 3))             # [kg * m^2 / s]

# SRP parameters (kept here for when a_SRP is enabled). Currently unused
# because a_SRP is set to zero per the doc's "set to 0 for the moment".
C_R: float = 1.3                                    # reflectivity in [1, 2]
A_SRP: float = 10.0                                 # effective area [m^2]
P_SUN: float = 4.56e-6                              # solar pressure at 1 AU [N/m^2]
R_SUN_REF_KM: float = 1.495978707e8                 # reference distance [km] (1 AU)

# Specific impulse and standard gravity (for mass depletion when control on).
ISP: float = 220.0                                  # [s]
G0: float = 9.80665e-3                              # [km/s^2]  (note: km units)


# ── Gravitational acceleration ───────────────────────────────────────────────

def a_grav(r_I: np.ndarray, et: float) -> np.ndarray:
    """
    N-body gravitational acceleration on the spacecraft, in ICRF [km/s^2].

    Implements Eq. (7):
        a_grav^I = sum_b  -mu_b * (r_IB^I - r_Ib^I) / |r_IB^I - r_Ib^I|^3
    """
    a = np.zeros(3)
    for body, mu in GM.items():
        r_b = spice.spkezr(body, et, "J2000", "NONE", "SSB")[0][:3]
        dr = r_I - r_b
        a -= mu * dr / np.linalg.norm(dr) ** 3
    return a


# ── ODE right-hand side (single spacecraft, 14-state) ────────────────────────

def x_dot_i(t: float, x: np.ndarray, et0: float, u: np.ndarray) -> np.ndarray:
    """
    ODE right-hand side for the coupled translational + rotational dynamics
    of a single spacecraft.

    State x (14,):
        x[0:3]   r_IB^I    position           [km]
        x[3:6]   v_IB^I    velocity           [km/s]
        x[6:10]  q_I^B     attitude quaternion (inertial -> body)
        x[10:13] w_IB^B    angular rate, body frame   [rad/s]
        x[13]    m         mass               [kg]

    Control u (6,):
        u[0:3]   a_ctrl^I  control acceleration, inertial    [km/s^2]
        u[3:6]   tau_ctrl^B control torque, body             [N*m]

    Returns dx/dt of shape (14,).
    """
    # --- Unpack state ---
    r     = x[0:3]
    v     = x[3:6]
    q     = x[6:10]
    omega = x[10:13]
    m     = x[13]

    # --- Unpack control ---
    a_ctrl   = u[0:3]   # [km/s^2] in inertial frame
    tau_ctrl = u[3:6]   # [N*m]    in body frame

    # --- Translational dynamics (Eq. 6, row 2) ----------------------------
    # dot r = v
    # dot v = a_grav + a_SRP + a_ion + a_grav(isc) + a_ctrl + a_p
    a_gravity     = a_grav(r, et0 + t)              # full N-body, active
    a_SRP         = np.zeros(3)                     # Eq. (8), set to 0 for now
    a_ion         = np.zeros(3)                     # Eq. (9), set to 0
    a_grav_isc    = np.zeros(3)                     # Eq. (10), inter-s/c gravity, set to 0
    a_p           = np.zeros(3)                     # Eq. (12), process noise, set to 0

    r_dot = v
    v_dot = a_gravity + a_SRP + a_ion + a_grav_isc + a_ctrl + a_p

    # --- Attitude kinematics (Eq. 6, row 3) -------------------------------
    # dot q = 0.5 * Omega(omega) * q
    q_dot = 0.5 * Omega_omega(omega) @ q

    # --- Rotational dynamics (Eq. 6, row 4) -------------------------------
    # J * dot omega = tau_SRP + tau_ctrl + tau_p - omega x (J omega) - J_dot omega
    tau_SRP = np.zeros(3)                           # Eq. (14), set to 0
    tau_p   = np.zeros(3)                           # Eq. (15), set to 0

    tau_total = tau_SRP + tau_ctrl + tau_p
    omega_dot = J_B_inv @ (
        tau_total
        - np.cross(omega, J_B @ omega)
        - J_B_dot @ omega
    )

    # --- Mass dynamics (Eq. 6, row 5) -------------------------------------
    # dot m = - sum_l |f_ctrl,l| / (Isp_l * g0)
    # No control thrust -> no mass change for the moment.
    m_dot = 0.0

    return np.concatenate([
        r_dot,
        v_dot,
        q_dot,
        omega_dot,
        np.array([m_dot]),
    ])


# ── Propagator / single-step integrator ──────────────────────────────────────

def step(
    x:           np.ndarray,
    dt:          float,
    et:          float,
    u:           np.ndarray | None = None,
    rtol:        float = 1e-12,
    atol:        float = 1e-12,
    renormalize: bool = True,
) -> np.ndarray:
    """
    Advance the 14-state x by one control sample dt [s] from epoch et,
    with zero-order-hold control u over the interval.

    renormalize : if True (default), normalize the quaternion after the step
                  to fight numerical drift. Set False when measuring raw
                  ODE drift in tests.

    Returns x_next (14,).
    """
    if u is None:
        u = np.zeros(6)

    sol = solve_ivp(
        x_dot_i, (0.0, dt),
        x,
        method       = "DOP853",
        args         = (et, u),
        rtol         = rtol,
        atol         = atol,
        dense_output = False,
    )
    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    x_next = sol.y[:, -1]

    if renormalize:
        q = x_next[6:10]
        x_next[6:10] = q / np.linalg.norm(q)

    return x_next


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_trajectory(
    t_hist: np.ndarray,
    X_hist: np.ndarray,
    et0:    float,
) -> None:
    """
    Build a 2x3 figure summarizing one propagation.

    Panels:
        (1,1)  3D trajectory in ICRF with Sun and Earth
        (1,2)  Position components vs time
        (1,3)  Velocity components vs time
        (2,1)  Quaternion components vs time
        (2,2)  Angular velocity (body frame) vs time
        (2,3)  Conservation diagnostics (log scale)

    t_hist : (N,)     time since et0 [s]
    X_hist : (N, 14)  state history
    et0    : float    initial ET (used to query Earth/Sun in ICRF)
    """
    # Convenience slices.
    r = X_hist[:, 0:3]       # km
    v = X_hist[:, 3:6]       # km/s
    q = X_hist[:, 6:10]
    w = X_hist[:, 10:13]     # rad/s

    t_h = t_hist / 3600.0    # hours, readable axis labels

    # --- Sample reference body positions over the time window ------------
    n_ref = 200
    t_ref = np.linspace(t_hist[0], t_hist[-1], n_ref)
    r_earth_hist = np.array([
        spice.spkezr("EARTH", et0 + tk, "J2000", "NONE", "SSB")[0][:3]
        for tk in t_ref
    ])
    r_sun_pos = spice.spkezr("SUN", et0, "J2000", "NONE", "SSB")[0][:3]

    # --- Figure ----------------------------------------------------------
    # 3x3 grid:
    #   row 0: [ 3D full ICRF | 3D zoom (Earth + s/c) | position ]
    #   row 1: [ velocity     | quaternion full       | quaternion zoom ]
    #   row 2: [ omega        | conservation          | mass ]
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
    # Auto-frame to a tight box around both tracks.
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

    # (1, 1) Quaternion components, full window
    ax_q = plt.subplot2grid((3, 3), (1, 1), fig=fig)
    ax_q.plot(t_h, q[:, 0], "k-", lw=0.8, label="$q_w$ (scalar)")
    ax_q.plot(t_h, q[:, 1], "r-", lw=0.8, label="$q_x$")
    ax_q.plot(t_h, q[:, 2], "g-", lw=0.8, label="$q_y$")
    ax_q.plot(t_h, q[:, 3], "b-", lw=0.8, label="$q_z$")
    ax_q.set_xlabel("time [h]")
    ax_q.set_ylabel("quaternion")
    ax_q.set_title("Attitude quaternion $q_I^B$ (full window)")
    ax_q.grid(True, alpha=0.3)
    ax_q.legend(loc="best", fontsize=8)
    ax_q.set_ylim(-1.05, 1.05)

    # (1, 2) Quaternion components, zoomed to first ~3 spin periods
    # Spin period T = 2*pi / |omega| with the initial omega.
    w0_mag    = np.linalg.norm(w[0]) if np.linalg.norm(w[0]) > 0 else 1.0
    t_zoom_s  = 3.0 * 2.0 * np.pi / w0_mag                 # 3 periods [s]
    t_zoom_h  = min(t_zoom_s / 3600.0, t_h[-1])            # cap at full window

    ax_qz = plt.subplot2grid((3, 3), (1, 2), fig=fig)
    ax_qz.plot(t_h, q[:, 0], "k-", label="$q_w$")
    ax_qz.plot(t_h, q[:, 1], "r-", label="$q_x$")
    ax_qz.plot(t_h, q[:, 2], "g-", label="$q_y$")
    ax_qz.plot(t_h, q[:, 3], "b-", label="$q_z$")
    ax_qz.set_xlim(0, t_zoom_h)
    ax_qz.set_xlabel("time [h]")
    ax_qz.set_ylabel("quaternion")
    ax_qz.set_title(f"Quaternion (zoom: first {t_zoom_h*60:.1f} min)")
    ax_qz.grid(True, alpha=0.3)
    ax_qz.legend(loc="best", fontsize=8)
    ax_qz.set_ylim(-1.05, 1.05)

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

    # (2,3) Conservation diagnostics, log scale
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


# ── Demo: epoch-stepping main loop ───────────────────────────────────────────

if __name__ == "__main__":
    load_kernels()

    print("GM values loaded from gm_de440.tpc [km^3/s^2]:")
    for body, mu in GM.items():
        print(f"  {body:<20s} {mu: .10e}")

    # --- Initial epoch ----------------------------------------------------
    et0 = spice.str2et("2027-01-01T00:00:00")

    # --- Initial spacecraft state -----------------------------------------
    # Position: approximate Sun-Earth L2 (Earth + 1.5e6 km along Sun-Earth line)
    r_earth = spice.spkezr("EARTH", et0, "J2000", "NONE", "SSB")[0]
    r_sun   = spice.spkezr("SUN",   et0, "J2000", "NONE", "SSB")[0]
    earth_hat = (r_earth[:3] - r_sun[:3]) / np.linalg.norm(r_earth[:3] - r_sun[:3])

    r0 = r_earth[:3] + 1.5e6 * earth_hat          # [km]
    v0 = r_earth[3:].copy()                       # [km/s]  (inherit Earth's velocity)
    q0 = np.array([1.0, 0.0, 0.0, 0.0])           # identity quaternion (scalar-first)
    w0 = np.array([0.01, 0.0, 0.0])               # [rad/s]  small body-x spin
    m0 = 500.0                                    # [kg]

    x = np.concatenate([r0, v0, q0, w0, np.array([m0])])

    print(f"\nEpoch  : {spice.et2utc(et0, 'ISOC', 3)}")
    print(f"r0     : {x[0:3]}  km")
    print(f"v0     : {x[3:6]}  km/s")
    print(f"q0     : {x[6:10]}")
    print(f"w0     : {x[10:13]}  rad/s")
    print(f"m0     : {x[13]}  kg")

    # --- Main control loop -----------------------------------------------
    # 1 day at 20 s sampling -> 4320 samples. With omega ~ 0.01 rad/s the spin
    # period is ~628 s, so we get ~300 samples per period -> smooth curves.
    dt        = 20.0
    n_steps   = int(86400 / dt)
    et        = et0

    # History buffers, including the initial sample.
    t_hist        = np.zeros(n_steps + 1)
    X_hist        = np.zeros((n_steps + 1, 14))
    t_hist[0]     = 0.0
    X_hist[0, :]  = x

    print(f"\n--- Epoch-stepping simulation: {n_steps} steps of {dt:.1f} s "
          f"({n_steps * dt / 3600:.2f} h total) ---")

    print_every = max(1, n_steps // 20)            # ~20 progress lines
    for k in range(n_steps):
        # Specify control input for this epoch.
        # Layout: [a_ctrl^I (3) [km/s^2], tau_ctrl^B (3) [N*m]]
        u = np.zeros(6)

        # Propagate one step forward.
        x_next = step(x, dt=dt, et=et, u=u)

        # Store history.
        t_hist[k + 1]    = (k + 1) * dt
        X_hist[k + 1, :] = x_next

        # Sparse progress diagnostics.
        if (k + 1) % print_every == 0 or k == n_steps - 1:
            print(
                f"  k={k+1:5d}/{n_steps}  t={(k+1)*dt/3600:6.3f} h   "
                f"|r|={np.linalg.norm(x_next[0:3]):.6e} km   "
                f"|v|={np.linalg.norm(x_next[3:6]):.6f} km/s   "
                f"|q|={np.linalg.norm(x_next[6:10]):.12f}   "
                f"|w|={np.linalg.norm(x_next[10:13]):.6e} rad/s"
            )

        # Advance.
        x   = x_next
        et += dt

    # --- Plot ------------------------------------------------------------
    print("\nGenerating plots ...")
    plot_trajectory(t_hist, X_hist, et0)
