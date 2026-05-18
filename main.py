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

# Quaternion kinematics matrix Omega(omega) such that q_dot = 0.5 * Omega * q.
# (Already implemented in your utils.)
from utils.other.Omega_omega import Omega_omega

# Plotting routines (moved out of this file for clarity).
from utils.plotting.plotting import (
    plot_trajectory,
    plot_solar_system,
    plot_l2_rotating_frame_zoom,
)


# ── SPICE defaults ───────────────────────────────────────────────────────────

KERNELS = [
    "data/spice_kernels/naif0012.tls",
    "data/spice_kernels/de440.bsp",
    "data/spice_kernels/gm_de440.tpc",
]

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


# ── SpiceEnv: kernels, GM, and ephemeris lookups ─────────────────────────────

class SpiceEnv:
    """
    Bundles everything SPICE-side that stays constant for a simulation run:
    kernel set, body list, reference frame, aberration correction, observer.
    Loads the kernels and reads GM at construction so the rest of the code
    just calls `env.body_position(body, et)` / `env.GM[body]` etc.

    Note: SPICE state is process-global (spice.furnsh registers kernels in a
    shared pool), so only one SpiceEnv per process is meaningful — but having
    it as an object keeps the universal args out of every call site.
    """

    def __init__(
        self,
        kernels:  list[str]        = KERNELS,
        bodies:   tuple[str, ...]  = BODIES,
        frame:    str              = "J2000",
        abcorr:   str              = "NONE",
        observer: str              = "SSB",
    ):
        self.kernels  = kernels
        self.bodies   = bodies
        self.frame    = frame
        self.abcorr   = abcorr
        self.observer = observer

        self._furnish()
        # GM dict [km^3/s^2], read from the PCK once at startup.
        self.GM: dict[str, float] = {
            body: float(spice.bodvrd(body, "GM", 1)[1][0]) for body in self.bodies
        }

    def _furnish(self) -> None:
        """Furnish SPICE kernels (raises a helpful error if any are missing)."""
        if not all(Path(k).exists() for k in self.kernels):
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
        for k in self.kernels:
            spice.furnsh(k)

    # ── Ephemeris lookup ─────────────────────────────────────────────────
    def body_position(self, body: str, et: float) -> np.ndarray:
        """Position of `body` at epoch `et`, in (frame, observer) [km]."""
        return spice.spkpos(body, et, self.frame, self.abcorr, self.observer)[0]

    # ── Time conversions (thin wrappers, kept for symmetry) ──────────────
    def str2et(self, s: str) -> float:
        return spice.str2et(s)

    def et2utc(self, et: float, fmt: str = "ISOC", prec: int = 3) -> str:
        return spice.et2utc(et, fmt, prec)


# ── Spacecraft constant parameters (single spacecraft, for the moment) ───────
# These will be made per-spacecraft once multi-spacecraft support is added.

# Inertia tensor in body frame [kg * m^2]. Slightly off-diagonal so that an
# x-axis spin couples into the other body axes (visible in the plots).
J_B = np.diag([100.0, 100.0, 100.0])
J_B[1, 2] = 10.0
J_B[2, 1] = 10.0   # mirror to keep J symmetric --> edit --> SEE Scharf et a. Overivew of TPF...

# SRP parameters (kept here for when a_SRP is enabled). Currently unused
# because a_SRP is set to zero per the doc's "set to 0 for the moment".
C_R: float = 1.8                                    # reflectivity in [1, 2], (JWST cannonball, Farres & Petersen (2019), AAS 19-657), https://ntrs.nasa.gov/api/citations/20190029609/downloads/20190029609.pdf
A_SRP: float = 40.0                                 # effective area [m^2], rough estimate for LIFE collector spacecraft
P_SUN: float = 4.53e-6                              # solar pressure at 1 AU [N/m^2], (JWST cannonball, Farres & Petersen (2019), AAS 19-657), https://ntrs.nasa.gov/api/citations/20190029609/downloads/20190029609.pdf
R_SUN_REF_KM: float = 1.495978707e8                 # reference distance [km] (1 AU),

# Unit conversion: SRP gives m/s^2 natively; rest of dynamics is in km.
KM_PER_M: float = 1.0e-3

# Specific impulse and standard gravity (for mass depletion when control on).
ISP: float = 220.0                                  # [s]
G0: float = 9.80665e-3                              # [km/s^2]  (note: km units)


# ── Plant: single-spacecraft dynamics + integrator ───────────────────────────

class Plant:
    """
    Dynamics of one spacecraft (14-state, Eq. (6)) bundled with a single-step
    integrator. Holds the spacecraft constants as instance attributes so that
    multi-spacecraft support is a matter of instantiating more Plants. Takes
    a SpiceEnv so all ephemeris lookups go through one configured object.
    """

    def __init__(
        self,
        env:   SpiceEnv,
        J_B:   np.ndarray = J_B,
        C_R:   float      = C_R,
        A_SRP: float      = A_SRP,
        ISP:   float      = ISP,
    ):
        self.env     = env
        self.J_B     = J_B
        self.J_B_inv = np.linalg.inv(J_B)
        # Inertia rate dJ/dt in body frame [kg * m^2 / s].
        # Per Eq. (16) this is driven by mass depletion: J_dot = m_dot * K_r.
        # With zero control (no thrust), m_dot = 0, so J_dot = 0.
        self.J_B_dot = np.zeros((3, 3))
        self.C_R     = C_R
        self.A_SRP   = A_SRP
        self.ISP     = ISP

    # ── Gravitational acceleration ───────────────────────────────────────
    def a_grav(self, r_I: np.ndarray, et: float) -> np.ndarray:
        """
        N-body gravitational acceleration on the spacecraft, in ICRF [km/s^2].

        Implements Eq. (7):
            a_grav^I = sum_b  -mu_b * (r_IB^I - r_Ib^I) / |r_IB^I - r_Ib^I|^3
        """
        a = np.zeros(3)
        for body, mu in self.env.GM.items():
            r_b = self.env.body_position(body, et)
            dr  = r_I - r_b
            a  -= mu * dr / np.linalg.norm(dr) ** 3
        return a

    # ── Solar radiation pressure ─────────────────────────────────────────
    def a_srp(self, r_I: np.ndarray, m: float, et: float) -> np.ndarray:
        """
        Cannonball solar radiation pressure acceleration on the spacecraft,
        in ICRF [km/s^2].

        Implements (the corrected form of) Eq. (8):

            a_srp^I = C_R * (A / m) * P_sun * (r_sun_ref / r)^2 * u_hat

        where:
            u_hat = (r_IB^I - r_Is^I) / |r_IB^I - r_Is^I|
                    is the unit vector from the Sun toward the spacecraft
                    (so SRP pushes the spacecraft *away* from the Sun),
            r     = |r_IB^I - r_Is^I| is the Sun-spacecraft distance,
            P_sun is the solar pressure at the reference distance r_sun_ref = 1 AU,
                  so "P_sun * (r_sun_ref / r)^2" reproduces the 1/r^2 falloff
                  of solar flux.

        No eclipse / shadow function is included, matching the doc's Eq. (8).
        For an L2 halo orbit this is essentially fine (s/c is in continuous
        sunlight); it would need to be added for trajectories that cross
        Earth's umbra.
        """
        r_sun  = self.env.body_position("SUN", et)
        d      = r_I - r_sun                            # Sun -> spacecraft
        d_norm = np.linalg.norm(d)
        u_hat  = d / d_norm

        # Pressure scaling with distance: P(r) = P_SUN * (R_SUN_REF / r)^2 [N/m^2].
        P_at_r          = P_SUN * (R_SUN_REF_KM / d_norm) ** 2
        a_mag_km_per_s2 = self.C_R * (self.A_SRP / m) * P_at_r * KM_PER_M
        return a_mag_km_per_s2 * u_hat

    # ── ODE right-hand side (14-state) ───────────────────────────────────
    def x_dot(self, tau: float, x: np.ndarray, et0: float, u: np.ndarray) -> np.ndarray:
        """
        State x (14,):
            x[0:3]   r_IB^I    position           [km]
            x[3:6]   v_IB^I    velocity           [km/s]
            x[6:10]  q_I^B     attitude quaternion (inertial -> body)
            x[10:13] w_IB^B    angular rate, body frame   [rad/s]
            x[13]    m         mass               [kg]

        Control u (6,):
            u[0:3]   a_ctrl^I  control acceleration, inertial    [km/s^2]
            u[3:6]   tau_ctrl^B control torque, body             [N*m]
        """
        r, v, q, omega, m = x[0:3], x[3:6], x[6:10], x[10:13], x[13]
        a_ctrl, tau_ctrl  = u[0:3], u[3:6]

        # --- Translational dynamics (Eq. 6, row 2) ----------------------------
        a_gravity  = self.a_grav(r, et0 + tau)          # full N-body, active
        a_rad      = self.a_srp(r, m, et0 + tau)        # Eq. (8), cannonball SRP
        a_ion      = np.zeros(3)                        # Eq. (9), set to 0
        a_grav_isc = np.zeros(3)                        # Eq. (10), inter-s/c gravity, set to 0
        a_p        = np.zeros(3)                        # Eq. (12), process noise, set to 0

        r_dot = v
        v_dot = a_gravity + a_rad + a_ion + a_grav_isc + a_ctrl + a_p

        # --- Attitude kinematics (Eq. 6, row 3) -------------------------------
        q_dot = 0.5 * Omega_omega(omega) @ q

        # --- Rotational dynamics (Eq. 6, row 4) -------------------------------
        # J * dot omega = tau_SRP + tau_ctrl + tau_p - omega x (J omega) - J_dot omega
        tau_SRP = np.zeros(3)                           # Eq. (14), set to 0
        tau_p   = np.zeros(3)                           # Eq. (15), set to 0
        omega_dot = self.J_B_inv @ (
            tau_SRP + tau_ctrl + tau_p
            - np.cross(omega, self.J_B @ omega)
            - self.J_B_dot @ omega
        )

        # --- Mass dynamics (Eq. 6, row 5) -------------------------------------
        # dot m = - sum_l |f_ctrl,l| / (Isp_l * g0). No control thrust -> 0.
        m_dot = 0.0

        return np.concatenate([r_dot, v_dot, q_dot, omega_dot, [m_dot]])

    # ── Single-step propagator ───────────────────────────────────────────
    def step(
        self,
        x:           np.ndarray,
        u:           np.ndarray,
        t:           float,
        dt:          float,
        rtol:        float = 1e-12,
        atol:        float = 1e-12,
        renormalize: bool  = True,
    ) -> np.ndarray:
        """
        Advance x by one control sample dt [s] from epoch t [SPICE ET],
        with zero-order-hold control u over the interval. Returns x_next (14,).
        """
        sol = solve_ivp(
            self.x_dot, (0.0, dt), x,
            method = "DOP853",
            args   = (t, u),
            rtol   = rtol,
            atol   = atol,
        )
        if not sol.success:
            raise RuntimeError(f"Integration failed: {sol.message}")

        x_next = sol.y[:, -1]
        if renormalize:                                # fight quaternion drift
            x_next[6:10] = x_next[6:10] / np.linalg.norm(x_next[6:10])
        return x_next


# ── Demo: epoch-stepping main loop ───────────────────────────────────────────

if __name__ == "__main__":
    # SPICE environment: loads kernels and reads GM at construction.
    env = SpiceEnv()

    print("GM values loaded from gm_de440.tpc [km^3/s^2]:")
    for body, mu in env.GM.items():
        print(f"  {body:<20s} {mu: .10e}")

    # --- Initial epoch ----------------------------------------------------
    et0 = env.str2et("2026-05-12T00:00:00")

    # --- Initial spacecraft state -----------------------------------------
    # Approximate Sun-Earth L2 halo-like initial condition from JWST: https://ssd.jpl.nasa.gov/horizons/app.html#/
    r0 = np.array([-9.594503991242750e7, -1.098032827423822e8,-4.778858640538428e7 ])   # [km]
    v0 = np.array([2.279303677102495e1, -1.735582913273474e1, -7.640659579588683e0])    # [km/s]
    q0 = np.array([1.0, 0.0, 0.0, 0.0])           # identity quaternion (scalar-first)
    w0 = np.array([0.001, 0.01, 0.0])               # [rad/s]  small body-x spin and some small y to couple into the other axes; not based on any real data, just for demo purposes
    m0 = 3000.0                                    # [kg]

    x = np.concatenate([r0, v0, q0, w0, np.array([m0])])

    print(f"\nEpoch  : {env.et2utc(et0)}")
    print(f"r0     : {x[0:3]}  km")
    print(f"v0     : {x[3:6]}  km/s")
    print(f"q0     : {x[6:10]}")
    print(f"w0     : {x[10:13]}  rad/s")
    print(f"m0     : {x[13]}  kg")

    # --- Instantiate the spacecraft plant --------------------------------
    plant = Plant(env)

    # --- Main control loop -----------------------------------------------
    # 1 day at 2 s sampling -> 43200 samples. With omega ~ 0.01 rad/s the spin
    # period is ~628 s, so we get ~300 samples per period -> smooth curves.
    dt        = 200.0
    n_steps   = int(10*86400 / dt)
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
        x_next = plant.step(x, u, et, dt)

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
    plot_trajectory(t_hist, X_hist, et0, J_B)
    plot_solar_system(
        et0          = et0,
        duration     = t_hist[-1],
        spacecraft_r = X_hist[:, 0:3],
        spacecraft_t = t_hist,
    )
    plot_l2_rotating_frame_zoom(
        et0          = et0,
        t_hist       = t_hist,
        spacecraft_r = X_hist[:, 0:3],
    )
