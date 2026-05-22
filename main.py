"""
LIFE Mission – High-Fidelity Translational + Rotational Dynamics (Truth Model)
N-spacecraft formation: 1 chief + (N-1) deputies, chief-plus-relative form.

State layout (per Eq. (5) of the design doc, extended to N spacecraft):

    Spacecraft 0 (CHIEF, absolute state in ICRF):
        x[ 0: 3]   r_IB^I    position           [km]
        x[ 3: 6]   v_IB^I    velocity           [km/s]
        x[ 6:10]   q_I^B     attitude quaternion (inertial -> body)
        x[10:13]   ω_IB^B    angular rate, body frame   [rad/s]
        x[13]      m         mass               [kg]

    Spacecraft i = 1..N-1 (DEPUTY, RELATIVE translational state, ABSOLUTE attitude):
        x[14*i +  0:14*i +  3]   δr^I = r_dep - r_chief   [km]     (ICRF)
        x[14*i +  3:14*i +  6]   δv^I = v_dep - v_chief   [km/s]
        x[14*i +  6:14*i + 10]   q_I^B,i                            (absolute)
        x[14*i + 10:14*i + 13]   ω_IB^B,i                  [rad/s]
        x[14*i + 13]             m_i                       [kg]

    Total length: 14 * N
    For N = 5: x.shape = (70,)

Control layout (per spacecraft):
    u[6*i + 0:6*i + 3]   a_ctrl^I,i   control acceleration, inertial    [km/s^2]
    u[6*i + 3:6*i + 6]   τ_ctrl^B,i   control torque, body              [N*m]

    Total length: 6 * N. For N = 5: u.shape = (30,)

WHY THIS LAYOUT?
The experiment in test_num_integration.py established that absolute-state
differencing of two ~1.5e8 km SSB-relative position vectors hits a numerical
floor of ~200 μm/day at 1 AU, growing to ~mm-scale over a month. This is set
by double-precision cancellation and cannot be reduced by tighter integration.
For mm-level relative formation control, deputies MUST be propagated in
relative (δr, δv) form. Attitude states remain absolute since unit-magnitude
quaternions have no large-number cancellation issue.

The core simulation loop is preserved exactly:

    x_next = plant.step(x, u, et, dt)

x is the full stacked 14*N vector; u is the full stacked 6*N control.
"""
import sys
sys.dont_write_bytecode = True

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import spiceypy as spice
from scipy.integrate import solve_ivp

from utils.other.Omega_omega import Omega_omega

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
    """Kernel set + GM cache + ephemeris lookups. Unchanged from single-s/c version."""

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
        self.GM: dict[str, float] = {
            body: float(spice.bodvrd(body, "GM", 1)[1][0]) for body in self.bodies
        }

    def _furnish(self) -> None:
        if not all(Path(k).exists() for k in self.kernels):
            raise FileNotFoundError(
                "\n" + "=" * 70 + "\n"
                "SPICE kernels not found!\n\n"
                "Required kernels are missing from data/spice_kernels/\n"
                "Download them with:\n\n"
                "  pip install -e .\n"
                "  lifecontrol-setup-kernels\n\n"
                "Or manually from NAIF:\n"
                "  https://naif.jpl.nasa.gov/pub/naif/generic_kernels/\n"
                "=" * 70
            )
        for k in self.kernels:
            spice.furnsh(k)

    def body_position(self, body: str, et: float) -> np.ndarray:
        return spice.spkpos(body, et, self.frame, self.abcorr, self.observer)[0]

    def str2et(self, s: str) -> float:
        return spice.str2et(s)

    def et2utc(self, et: float, fmt: str = "ISOC", prec: int = 3) -> str:
        return spice.et2utc(et, fmt, prec)


# ── Per-spacecraft parameters (one instance per spacecraft) ──────────────────

# Default inertia tensor — slightly off-diagonal so y-z coupling is visible.
_J_B_DEFAULT = np.diag([100.0, 100.0, 100.0])
_J_B_DEFAULT[1, 2] = 10.0
_J_B_DEFAULT[2, 1] = 10.0

C_R: float           = 1.8                                 # reflectivity
A_SRP: float         = 40.0                                # effective area [m^2]
P_SUN: float         = 4.53e-6                             # solar pressure at 1 AU [N/m^2]
R_SUN_REF_KM: float  = 1.495978707e8                       # 1 AU [km]
KM_PER_M: float      = 1.0e-3
ISP: float           = 220.0                               # [s]
G0: float            = 9.80665e-3                          # [km/s^2]


@dataclass
class SpacecraftParams:
    """
    Per-spacecraft constants. One instance per spacecraft so that each can
    have its own inertia tensor, reflective area, Isp, etc.
    """
    J_B:   np.ndarray = field(default_factory=lambda: _J_B_DEFAULT.copy())
    C_R:   float      = C_R
    A_SRP: float      = A_SRP                              # [m^2]
    ISP:   float      = ISP                                # [s]
    label: str        = "spacecraft"                       # for plotting/printing

    def __post_init__(self):
        # Precompute J_B^{-1} once; J_B_dot is zero whenever m_dot is zero,
        # which is the case for the current zero-thrust scenario. When
        # thrust is enabled, J_B_dot will be computed in the RHS from m_dot.
        self.J_B_inv = np.linalg.inv(self.J_B)


# ── Plant: N-spacecraft dynamics + integrator ────────────────────────────────

# State and control block sizes per spacecraft.
NX_PER_SC: int = 14
NU_PER_SC: int = 6


class Plant:
    """
    Dynamics for an N-spacecraft formation:
      - Spacecraft 0 is the CHIEF: absolute state in SSB-centered ICRF.
      - Spacecraft 1..N-1 are DEPUTIES: relative translational state (δr, δv)
        w.r.t. the chief in ICRF, plus absolute attitude.

    The chief's translational dynamics use the standard N-body gravity sum
    (unchanged from the single-spacecraft Plant). Deputy dynamics use the
    DIFFERENTIAL form a_grav(r_chief + δr) - a_grav(r_chief), evaluated
    body-by-body so that the cancellation is done at well-conditioned scale.

    Attitude, mass, and rotational dynamics are identical for chief and
    deputies — only the translational equations differ by role.

    Public API:
        plant.step(x, u, et, dt) -> x_next     # x shape (14*N,), u shape (6*N,)
    """

    def __init__(
        self,
        env:           SpiceEnv,
        spacecraft:    list[SpacecraftParams],
    ):
        self.env        = env
        self.spacecraft = spacecraft
        self.N          = len(spacecraft)
        self.nx         = NX_PER_SC * self.N
        self.nu         = NU_PER_SC * self.N

    # ── State / control accessors ────────────────────────────────────────
    @staticmethod
    def _slice_x(i: int) -> slice:
        """State slice for spacecraft i (length 14)."""
        return slice(NX_PER_SC * i, NX_PER_SC * (i + 1))

    @staticmethod
    def _slice_u(i: int) -> slice:
        """Control slice for spacecraft i (length 6)."""
        return slice(NU_PER_SC * i, NU_PER_SC * (i + 1))

    # ── Acceleration: absolute N-body gravity (chief) ────────────────────
    def a_grav_absolute(self, r_I: np.ndarray, et: float) -> np.ndarray:
        """
        N-body gravitational acceleration at absolute position r_I [km] in ICRF.
        Eq. (7). Used by the chief.
        """
        a = np.zeros(3)
        for body, mu in self.env.GM.items():
            r_b = self.env.body_position(body, et)
            dr  = r_I - r_b
            a  -= mu * dr / np.linalg.norm(dr) ** 3
        return a

    # ── Acceleration: DIFFERENTIAL N-body gravity (deputy) ───────────────
    def a_grav_relative(
        self,
        r_chief: np.ndarray,
        delta_r: np.ndarray,
        et:      float,
    ) -> np.ndarray:
        """
        Differential N-body gravitational acceleration on a deputy at
        r_dep = r_chief + δr, computed body-by-body as

            a_rel = Σ_b [ a_b(r_chief + δr) - a_b(r_chief) ]

        with each per-body subtraction performed at well-conditioned scale
        (the body-relative vector d1 = r_chief - r_b is reused; the deputy's
        body-relative vector is d2 = d1 + δr, never reconstructed from
        SSB-based absolutes). This is the numerically-stable formulation
        validated in test_num_integration.py.
        """
        a = np.zeros(3)
        for body, mu in self.env.GM.items():
            r_b = self.env.body_position(body, et)
            d1  = r_chief - r_b                                  # body -> chief
            d2  = d1 + delta_r                                   # body -> deputy
            a  += -mu * (d2 / np.linalg.norm(d2) ** 3
                       - d1 / np.linalg.norm(d1) ** 3)
        return a

    # ── Acceleration: absolute SRP (chief OR deputy at absolute position) ─
    def a_srp(self, r_I: np.ndarray, m: float, A: float, C_R_loc: float,
              et: float) -> np.ndarray:
        """
        Cannonball SRP at absolute position r_I, body of mass m, area A,
        reflectivity C_R_loc. Eq. (8). [km/s^2 in ICRF]
        """
        r_sun  = self.env.body_position("SUN", et)
        d      = r_I - r_sun
        d_norm = np.linalg.norm(d)
        u_hat  = d / d_norm
        P_at_r = P_SUN * (R_SUN_REF_KM / d_norm) ** 2
        a_mag  = C_R_loc * (A / m) * P_at_r * KM_PER_M
        return a_mag * u_hat

    # ── Acceleration: DIFFERENTIAL SRP (deputy) ──────────────────────────
    def a_srp_relative(
        self,
        r_chief: np.ndarray,
        delta_r: np.ndarray,
        m_chief: float, A_chief: float, CR_chief: float,
        m_dep:   float, A_dep:   float, CR_dep:   float,
        et:      float,
    ) -> np.ndarray:
        """
        Differential cannonball SRP: a_srp(deputy) - a_srp(chief).

        Note: SRP differs between chief and deputy for TWO reasons:
        (i) Position-dependent: 1/r^2 falloff + different sun-pointing
            unit vector at r_chief vs r_chief + δr. For δr ≪ r_sun this is
            tiny (~ |δr|/AU times nominal SRP), but we include it for
            consistency with the mm-level precision goal.
        (ii) Parameter-dependent: chief and deputy can have different
             (m, A, C_R). This is typically the dominant differential.

        Implementing it as the literal difference of two a_srp() calls is
        fine numerically because SRP is itself a small acceleration
        (~1e-13 km/s^2); no large-number cancellation is involved.
        """
        a_dep   = self.a_srp(r_chief + delta_r, m_dep, A_dep, CR_dep, et)
        a_chf   = self.a_srp(r_chief,           m_chief, A_chief, CR_chief, et)
        return a_dep - a_chf

    # ── Per-spacecraft rotational/mass dynamics ──────────────────────────
    def _attitude_and_mass_rhs(
        self,
        q:        np.ndarray,
        omega:    np.ndarray,
        m:        float,
        tau_ctrl: np.ndarray,
        params:   SpacecraftParams,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Returns (q_dot, omega_dot, m_dot) for one spacecraft. Identical
        formulation for chief and deputies — attitude is absolute in both
        cases and rotational dynamics doesn't care about translational frame.
        """
        # Quaternion kinematics.
        q_dot = 0.5 * Omega_omega(omega) @ q

        # Rotational dynamics — currently no SRP torque, no process noise.
        # Eq. (6, row 4): J ω̇ = τ_SRP + τ_ctrl + τ_p - ω × (J ω) - J̇ ω
        tau_SRP = np.zeros(3)
        tau_p   = np.zeros(3)
        # J_B_dot = 0 while m_dot = 0 (current case: zero thrust).
        J_B_dot = np.zeros((3, 3))
        omega_dot = params.J_B_inv @ (
            tau_SRP + tau_ctrl + tau_p
            - np.cross(omega, params.J_B @ omega)
            - J_B_dot @ omega
        )

        # Mass dynamics — zero while no thrust.
        m_dot = 0.0
        return q_dot, omega_dot, m_dot

    # ── ODE right-hand side (full 14*N state) ────────────────────────────
    def x_dot(self, tau: float, x: np.ndarray, et0: float, u: np.ndarray) -> np.ndarray:
        """
        RHS of the stacked N-spacecraft state. Layout in module docstring.
        """
        et = et0 + tau
        xdot = np.zeros_like(x)

        # ── Chief (index 0): absolute dynamics ───────────────────────────
        sc0       = self.spacecraft[0]
        x0        = x[self._slice_x(0)]
        u0        = u[self._slice_u(0)]
        r_chief   = x0[0:3]
        v_chief   = x0[3:6]
        q_chief   = x0[6:10]
        w_chief   = x0[10:13]
        m_chief   = x0[13]
        a_ctrl_0  = u0[0:3]
        tau_ctrl_0 = u0[3:6]

        a_grav_chief = self.a_grav_absolute(r_chief, et)
        a_srp_chief  = self.a_srp(r_chief, m_chief, sc0.A_SRP, sc0.C_R, et)
        a_p_chief    = np.zeros(3)                               # process noise (off)
        a_grav_isc_0 = np.zeros(3)                               # inter-s/c gravity (off)
        a_ion_0      = np.zeros(3)                               # ion engine (off)

        r_dot_chief = v_chief
        v_dot_chief = (a_grav_chief + a_srp_chief + a_ion_0
                       + a_grav_isc_0 + a_ctrl_0 + a_p_chief)

        q_dot_0, w_dot_0, m_dot_0 = self._attitude_and_mass_rhs(
            q_chief, w_chief, m_chief, tau_ctrl_0, sc0
        )

        xdot[self._slice_x(0)] = np.concatenate([
            r_dot_chief, v_dot_chief, q_dot_0, w_dot_0, [m_dot_0]
        ])

        # ── Deputies (indices 1..N-1): RELATIVE translational dynamics ───
        for i in range(1, self.N):
            sc_i      = self.spacecraft[i]
            xi        = x[self._slice_x(i)]
            ui        = u[self._slice_u(i)]
            delta_r   = xi[0:3]
            delta_v   = xi[3:6]
            q_i       = xi[6:10]
            w_i       = xi[10:13]
            m_i       = xi[13]
            a_ctrl_i  = ui[0:3]
            tau_ctrl_i = ui[3:6]

            # δa from gravity: differential N-body, numerically stable.
            da_grav = self.a_grav_relative(r_chief, delta_r, et)

            # δa from SRP: difference of two cannonball evaluations. Safe
            # because SRP is itself a tiny acceleration — no catastrophic
            # cancellation.
            da_srp  = self.a_srp_relative(
                r_chief, delta_r,
                m_chief, sc0.A_SRP, sc0.C_R,
                m_i,     sc_i.A_SRP, sc_i.C_R,
                et,
            )

            # Differential control acceleration: deputy's a_ctrl minus chief's a_ctrl.
            # Both are commanded in ICRF, so the difference is well-defined.
            da_ctrl = a_ctrl_i - a_ctrl_0

            # Other differential terms (all currently zero).
            da_ion      = np.zeros(3)
            da_grav_isc = np.zeros(3)
            da_p        = np.zeros(3)

            dr_dot = delta_v
            dv_dot = da_grav + da_srp + da_ion + da_grav_isc + da_ctrl + da_p

            # Attitude/mass dynamics: identical formulation for deputies.
            q_dot_i, w_dot_i, m_dot_i = self._attitude_and_mass_rhs(
                q_i, w_i, m_i, tau_ctrl_i, sc_i
            )

            xdot[self._slice_x(i)] = np.concatenate([
                dr_dot, dv_dot, q_dot_i, w_dot_i, [m_dot_i]
            ])

        return xdot

    # ── Single-step propagator (signature UNCHANGED) ─────────────────────
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
        Advance the full stacked state x by dt [s] from epoch t [SPICE ET]
        under zero-order-hold control u. Returns x_next, same shape as x.

        Identical signature to the single-spacecraft version — only the
        dimension of x and u changed.
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
        if renormalize:
            # Renormalize each spacecraft's quaternion independently.
            for i in range(self.N):
                q_slice = slice(NX_PER_SC * i + 6, NX_PER_SC * i + 10)
                x_next[q_slice] = x_next[q_slice] / np.linalg.norm(x_next[q_slice])
        return x_next


# ── Convenience: reconstruct absolute states from stacked relative form ──────

def deputy_absolute_position(x: np.ndarray, i: int) -> np.ndarray:
    """
    Return the absolute ICRF position of deputy i (i >= 1) from a stacked
    state vector x. For i = 0, returns the chief's absolute position directly.
    """
    if i == 0:
        return x[0:3].copy()
    return x[0:3] + x[NX_PER_SC * i : NX_PER_SC * i + 3]


def deputy_absolute_velocity(x: np.ndarray, i: int) -> np.ndarray:
    """As deputy_absolute_position but for velocity."""
    if i == 0:
        return x[3:6].copy()
    return x[3:6] + x[NX_PER_SC * i + 3 : NX_PER_SC * i + 6]


# ── Demo: epoch-stepping main loop ───────────────────────────────────────────

if __name__ == "__main__":
    env = SpiceEnv()

    print("GM values loaded from gm_de440.tpc [km^3/s^2]:")
    for body, mu in env.GM.items():
        print(f"  {body:<20s} {mu: .10e}")

    # --- Formation configuration ----------------------------------------
    # 5 spacecraft: 1 chief + 4 deputies in a planar "X" pattern around the
    # chief, 100 m baseline along the ICRF x-y plane. This is a placeholder
    # geometry — in the real mission the formation plane is normal to the
    # target star's line of sight.
    N_SC = 5

    spacecraft_params = [
        SpacecraftParams(label="chief"),
        SpacecraftParams(label="deputy_1"),
        SpacecraftParams(label="deputy_2"),
        SpacecraftParams(label="deputy_3"),
        SpacecraftParams(label="deputy_4"),
    ]

    plant = Plant(env, spacecraft_params)

    # --- Initial epoch ---------------------------------------------------
    et0 = env.str2et("2026-05-12T00:00:00")

    # --- Chief initial state (JWST-like halo IC) -------------------------
    r0_chief = np.array([-9.594503991242750e7, -1.098032827423822e8, -4.778858640538428e7])
    v0_chief = np.array([ 2.279303677102495e1, -1.735582913273474e1, -7.640659579588683e0])
    q0_chief = np.array([1.0, 0.0, 0.0, 0.0])
    w0_chief = np.array([0.001, 0.01, 0.0])
    m0_chief = 3000.0
    x0_chief = np.concatenate([r0_chief, v0_chief, q0_chief, w0_chief, [m0_chief]])

    # --- Deputy initial states: 100 m planar X formation around chief ----
    BASELINE_KM = 0.1   # 100 m
    delta_r0_list = [
        np.array([ BASELINE_KM,  0.0,         0.0]),   # deputy_1: +x
        np.array([-BASELINE_KM,  0.0,         0.0]),   # deputy_2: -x
        np.array([ 0.0,          BASELINE_KM, 0.0]),   # deputy_3: +y
        np.array([ 0.0,         -BASELINE_KM, 0.0]),   # deputy_4: -y
    ]

    # Same initial attitude and rate for all deputies for the demo.
    q0_dep = np.array([1.0, 0.0, 0.0, 0.0])
    w0_dep = np.array([0.001, 0.01, 0.0])
    m0_dep = 3000.0

    x0 = [x0_chief]
    for i, dr0 in enumerate(delta_r0_list):
        dv0 = np.zeros(3)
        x0.append(np.concatenate([dr0, dv0, q0_dep, w0_dep, [m0_dep]]))
    x = np.concatenate(x0)

    # --- Initial-state printout ------------------------------------------
    print(f"\nEpoch  : {env.et2utc(et0)}")
    print(f"N_SC   : {N_SC} ({'chief + ' + str(N_SC - 1) + ' deputies'})")
    print(f"Chief r0 : {x[0:3]}  km")
    print(f"Chief v0 : {x[3:6]}  km/s")
    print(f"Chief m0 : {x[13]}  kg")
    print(f"Baselines (initial, [m]):")
    for i in range(1, N_SC):
        dr_m = np.linalg.norm(x[NX_PER_SC * i : NX_PER_SC * i + 3]) * 1e3
        print(f"  deputy_{i}: |δr0| = {dr_m:.3f} m")

    # --- Main control loop (signature UNCHANGED) -------------------------
    dt       = 200.0
    n_steps  = int(3 * 86400 / dt)
    et       = et0

    t_hist        = np.zeros(n_steps + 1)
    X_hist        = np.zeros((n_steps + 1, plant.nx))
    X_hist[0, :]  = x

    print(f"\n--- Epoch-stepping simulation: {n_steps} steps of {dt:.1f} s "
          f"({n_steps * dt / 3600:.2f} h total) ---")

    print_every = max(1, n_steps // 20)
    for k in range(n_steps):
        # Specify control input for this epoch. Layout per spacecraft:
        #   [a_ctrl^I (3) km/s^2, tau_ctrl^B (3) N*m]
        u = np.zeros(plant.nu)

        x_next = plant.step(x, u, et, dt)

        t_hist[k + 1]    = (k + 1) * dt
        X_hist[k + 1, :] = x_next

        if (k + 1) % print_every == 0 or k == n_steps - 1:
            # Print chief norms + deputy_1 baseline as a quick diagnostic.
            dr1_m = np.linalg.norm(x_next[NX_PER_SC : NX_PER_SC + 3]) * 1e3
            print(
                f"  k={k+1:5d}/{n_steps}  t={(k+1)*dt/3600:6.3f} h   "
                f"|r_chief|={np.linalg.norm(x_next[0:3]):.6e} km   "
                f"|q_chief|={np.linalg.norm(x_next[6:10]):.12f}   "
                f"|δr_1|={dr1_m:.6f} m"
            )

        x   = x_next
        et += dt


    # --- Per-deputy diagnostic dump --------------------------------------
    # Helps disambiguate "curves overlap on the |δr| plot" — is it because
    # two deputies genuinely have identical dynamics (would be a bug) or
    # because the asymmetric tidal field gives them different evolutions
    # that just happen to look similar at this y-axis scale?
    print("\n--- Final baselines and drift over the run ---")
    for i in range(1, N_SC):
        dr_init = X_hist[0,  NX_PER_SC * i : NX_PER_SC * i + 3]
        dr_final = X_hist[-1, NX_PER_SC * i : NX_PER_SC * i + 3]
        mag_init  = np.linalg.norm(dr_init)  * 1e3   # m
        mag_final = np.linalg.norm(dr_final) * 1e3   # m
        drift     = mag_final - mag_init             # m  (signed)
        print(
            f"  {spacecraft_params[i].label:<10s} "
            f"|δr0| = {mag_init:.6f} m   "
            f"|δr(T)| = {mag_final:.9f} m   "
            f"Δ|δr| = {drift*1e6:+.3f} μm   "
            f"δr_final = [{dr_final[0]*1e3:+.6f}, {dr_final[1]*1e3:+.6f}, "
            f"{dr_final[2]*1e3:+.6f}] m"
        )

    # --- Pairwise check: which deputies have ~identical trajectories? ----
    # If any pair shows ~machine-precision agreement, that means the
    # dynamics produced literally the same numbers for both — which is
    # physically expected only if their initial conditions were identical.
    # Symmetric ±x or ±y deputies should NOT be identical (their absolute
    # positions in the asymmetric tidal field differ).
    print("\n--- Pairwise deputy-trajectory differences over full run ---")
    for i in range(1, N_SC):
        for j in range(i + 1, N_SC):
            dr_i = X_hist[:, NX_PER_SC * i : NX_PER_SC * i + 3]
            dr_j = X_hist[:, NX_PER_SC * j : NX_PER_SC * j + 3]
            # Max difference in |δr| over time (in meters).
            diff_mag = np.max(np.abs(np.linalg.norm(dr_i, axis=1)
                                     - np.linalg.norm(dr_j, axis=1))) * 1e3
            # Max componentwise difference (in meters).
            diff_vec = np.max(np.linalg.norm(dr_i - dr_j, axis=1)) * 1e3
            print(
                f"  {spacecraft_params[i].label} vs {spacecraft_params[j].label}: "
                f"max ||δr_i| - |δr_j|| = {diff_mag:.3e} m, "
                f"max |δr_i - δr_j| = {diff_vec:.3e} m"
            )

    # --- Plots -----------------------------------------------------------
    print("\nGenerating plots ...")
    plot_trajectory(
        t_hist        = t_hist,
        X_hist        = X_hist,
        et0           = et0,
        spacecraft    = spacecraft_params,
    )
    plot_solar_system(
        et0          = et0,
        duration     = t_hist[-1],
        X_hist       = X_hist,
        t_hist       = t_hist,
        spacecraft   = spacecraft_params,
    )
    plot_l2_rotating_frame_zoom(
        et0          = et0,
        t_hist       = t_hist,
        X_hist       = X_hist,
        spacecraft   = spacecraft_params,
    )
