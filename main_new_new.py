
import numpy as np
from scipy.integrate import solve_ivp
import spiceypy as spice
from pathlib import Path
from dataclasses import dataclass, field
from types import SimpleNamespace

# File Imports
from config.mission.config import KERNELS, BODIES, FRAME, ABCORR, OBSERVER
from utils.other.Omega_omega import Omega_omega
from utils.plotting.plotting import (
    plot_trajectory,
    plot_solar_system,
    plot_l2_rotating_frame_zoom,
)


##############################################
# Classes
##############################################

class SpiceEnv:
    """Kernel set + GM cache + ephemeris lookups."""

    def __init__(self, kernels, bodies, frame, abcorr, observer):
        self.kernels  = kernels
        self.bodies   = bodies
        self.frame    = frame
        self.abcorr   = abcorr
        self.observer = observer

        self._furnish()

        self.GM = {}
        for body in self.bodies:
            _, gm_values = spice.bodvrd(body, "GM", 1)
            self.GM[body] = float(gm_values[0])

        self._print_gm()

    def _furnish(self):
        if not all(Path(k).exists() for k in self.kernels):
            raise FileNotFoundError("SPICE kernels not found!")
        for k in self.kernels:
            spice.furnsh(k)

    def _print_gm(self):
        print("Gravitational Parameters (GM) [km^3/s^2]:")
        for body, gm in self.GM.items():
            print(f"  {body:<20s} {gm: .10e}")

    def body_position(self, body: str, et: float) -> np.ndarray:
        return spice.spkpos(body, et, self.frame, self.abcorr, self.observer)[0]

    def str2et(self, s: str) -> float:
        return spice.str2et(s)

    def et2utc(self, et: float, fmt: str = "ISOC", prec: int = 3) -> str:
        return spice.et2utc(et, fmt, prec)


@dataclass(frozen=True)
class Parameters:

    # ── Physical constants ──────────────────────────────────────────────
    P_SUN           = 4.53e-6           # solar pressure at 1 AU [N/m^2]
    R_SUN_REF_KM    = 1.495978707e8     # 1 AU [km]
    G_0             = 9.80665e-3        # standard Earth surface gravity [km/s^2]

    # ── Spacecraft parameters (Leader / Follower) ───────────────────────
    m_init_L        = 4000.0  # [kg]
    m_init_F        = 3000.0  # [kg]

    m_cylinder_L    = 3000.0  # [kg]  (fuel-bearing section, mass varies)
    m_cylinder_F    = 2000.0  # [kg]

    h_cylinder_L    = 5.0     # [m]
    h_cylinder_F    = 4.8     # [m]

    h_ring          = 1.2     # [m]
    r_cylinder      = 2.57    # [m]
    r_in            = 2.7     # [m]
    r_out           = 3.8     # [m]

    c_reflect       = 1.8     # SRP reflectivity coefficient

    Isp_L           = 220.0   # [s]
    Isp_F           = 220.0   # [s]

    # ── Derived (computed in __post_init__) ─────────────────────────────
    m_ring_L:           float      = field(init=False)
    m_ring_F:           float      = field(init=False)
    J_cylinder_L:       np.ndarray = field(init=False)
    J_cylinder_F:       np.ndarray = field(init=False)
    J_ring_L:           np.ndarray = field(init=False)
    J_ring_F:           np.ndarray = field(init=False)
    J_init_L:           np.ndarray = field(init=False)
    J_init_F:           np.ndarray = field(init=False)
    J_ring_per_mass_L:  np.ndarray = field(init=False)
    J_ring_per_mass_F:  np.ndarray = field(init=False)
    SRP_area_L:         float      = field(init=False)
    SRP_area_F:         float      = field(init=False)

    def __post_init__(self):
        # frozen=True needs object.__setattr__ for derived fields
        s = lambda k, v: object.__setattr__(self, k, v)

        # Constant ring mass = initial total - cylinder mass (only cylinder burns)
        m_ring_L = mass_ring(self.m_init_L, self.m_cylinder_L)
        m_ring_F = mass_ring(self.m_init_F, self.m_cylinder_F)
        s("m_ring_L", m_ring_L)
        s("m_ring_F", m_ring_F)

        # Cylinder (fuel-bearing) inertia at initial mass
        J_cyl_L = inertia_cylinder(self.m_cylinder_L, self.r_cylinder, self.h_cylinder_L)
        J_cyl_F = inertia_cylinder(self.m_cylinder_F, self.r_cylinder, self.h_cylinder_F)
        s("J_cylinder_L", J_cyl_L)
        s("J_cylinder_F", J_cyl_F)

        # Ring inertia (constant, geometry+mass fixed)
        J_ring_L = inertia_ring(m_ring_L, self.r_in, self.r_out, self.h_ring)
        J_ring_F = inertia_ring(m_ring_F, self.r_in, self.r_out, self.h_ring)
        s("J_ring_L", J_ring_L)
        s("J_ring_F", J_ring_F)

        # Total initial inertia (ring + cylinder)
        s("J_init_L", J_ring_L + J_cyl_L)
        s("J_init_F", J_ring_F + J_cyl_F)

        # Per-unit-mass RING inertia (constant geometry). Per the design doc the
        # ring is the fuel-bearing section, so m_r,i is the time-varying mass and
        #   J_ring(t) = m_r,i(t) * J_ring_per_mass   =>   J̇ = ṁ * J_ring_per_mass
        s("J_ring_per_mass_L", inertia_ring(1.0, self.r_in, self.r_out, self.h_ring))
        s("J_ring_per_mass_F", inertia_ring(1.0, self.r_in, self.r_out, self.h_ring))

        # Effective SRP cross-section (projected rectangle of the cylinder)
        s("SRP_area_L", 2 * self.r_cylinder * self.h_cylinder_L)
        s("SRP_area_F", 2 * self.r_cylinder * self.h_cylinder_F)


class Sensor:
    """Navigation Class"""

    def __init__(self):
        # TODO
        pass

    def measure(self, x, t):
        # TODO
        return x.copy()


class Guidance:
    """Guidance Class"""

    def __init__(self, dim_y):
        # TODO
        self.dim_y = dim_y

    def reference(self, et):
        # TODO
        return np.zeros(self.dim_y)


class Controller:
    """Controller Class"""

    def __init__(self, u_dim):
        self.u_dim = u_dim

    def compute(self, y_hat, y_ref, t):
        # 1. Build OCP over [t, t + N*dt] using y_hat as x0 and y_ref as target.
        # 2. Solve (e.g. CasADi / acados / cvxpy).
        # 3. Cache the tail as warm-start for next step.
        # 4. Return u_0.
        return np.zeros(self.u_dim)


class Plant:
    """
    Plant Truth Model of the System.

    N-spacecraft formation:
      - Spacecraft 0 is the LEADER: absolute state in SSB-centered ICRF.
      - Spacecraft 1..N-1 are FOLLOWERS: relative translational state (δr, δv)
        w.r.t. the Leader in ICRF, plus absolute attitude.

    Relative form for the follower's translation is required to avoid the
    ~200 µm/day numerical floor that comes from differencing two ~1 AU
    SSB-relative position vectors at double precision. Attitude states stay
    absolute (unit-magnitude quaternions don't suffer that cancellation).
    """

    def __init__(self, env, param, n_sc, dim_x_sc, dim_u_sc):
        self.env        = env
        self.param      = param
        self.n_sc       = n_sc
        self.dim_x_sc   = dim_x_sc
        self.dim_u_sc   = dim_u_sc
        self.dim_x      = n_sc * dim_x_sc
        self.dim_u      = n_sc * dim_u_sc

    def step(self, x, u, t, dt,
             rtol = 1e-12, atol = 1e-12, renormalize = True):

        # Propagate full dynamics
        sol = solve_ivp(self.x_dot, (0.0, dt), x,
                        method = "DOP853", args = (t, u),
                        rtol = rtol, atol = atol, t_eval = [dt])

        if not sol.success:
            raise RuntimeError(f"Integration failed: {sol.message}")

        x_next = sol.y[:, -1]

        if renormalize:
            x_next = self._renormalize_quat(x_next, self.n_sc, self.dim_x_sc)

        return x_next

    # Implements RHS of x_dot = f(t, x)
    def x_dot(self, tau, x, et0, u):

        xdot = np.zeros_like(x)
        et   = et0 + tau

        # ── Leader Dynamics (i=0) ──────
        #              /\
        #             /  \
        #            |LEAD|
        #           [|====|]
        #            /|  |\
        #           o-+--+-o
        # ────────────────────────────────
        x_L         = x[self._slice_x(0, self.dim_x_sc)]
        u_L         = u[self._slice_u(0, self.dim_u_sc)]

        r_L         = x_L[0:3]
        v_L         = x_L[3:6]
        q_L         = x_L[6:10]
        w_L         = x_L[10:13]
        m_L         = x_L[13]

        a_ctrl_L    = u_L[0:3]
        tau_ctrl_L  = u_L[3:6]

        # Spacecraft mass derivative (rocket equation)
        m_dot_L     = self.mass_rhs(m_L, a_ctrl_L, self.param.Isp_L)

        # Time-varying inertia and its derivative
        J_L         = self._J_current(m_L, self.param.m_cylinder_L,
                                      self.param.J_cylinder_L,
                                      self.param.J_ring_per_mass_L)
        J_dot_L     = self.J_dot(m_dot_L, self.param.J_ring_per_mass_L)

        # Absolute translational dynamics
        r_dot_L     = v_L
        v_dot_L     = self.acc_rhs_abs(r_L, m_L, self.param.SRP_area_L, a_ctrl_L, et)

        # Rotational kinematics
        q_dot_L     = self.attitude_kin_rhs(q_L, w_L)

        # Rotational dynamics
        w_dot_L     = self.attitude_dyn_rhs(w_L, tau_ctrl_L, J_L, J_dot_L)

        # Write
        xdot[self._slice_x(0, self.dim_x_sc)] = np.concatenate([r_dot_L, v_dot_L, q_dot_L, w_dot_L, [m_dot_L]])

        # ── Follower Dynamics (i>0) ────
        #             ____
        #            |FLWR|
        #           [|o  o|]
        #            |____|
        #            /    \
        #           o      o
        # ────────────────────────────────
        for i in range(1, self.n_sc):

            x_i         = x[self._slice_x(i, self.dim_x_sc)]
            u_i         = u[self._slice_u(i, self.dim_u_sc)]

            dr_i        = x_i[0:3]      # δr w.r.t. Leader [km, ICRF]
            dv_i        = x_i[3:6]      # δv w.r.t. Leader [km/s, ICRF]
            q_i         = x_i[6:10]
            w_i         = x_i[10:13]
            m_i         = x_i[13]

            a_ctrl_i    = u_i[0:3]
            tau_ctrl_i  = u_i[3:6]

            # Spacecraft mass derivative
            m_dot_i     = self.mass_rhs(m_i, a_ctrl_i, self.param.Isp_F)

            # Time-varying inertia and its derivative
            J_i         = self._J_current(m_i, self.param.m_cylinder_F,
                                          self.param.J_cylinder_F,
                                          self.param.J_ring_per_mass_F)
            J_dot_i     = self.J_dot(m_dot_i, self.param.J_ring_per_mass_F)

            # Relative translational dynamics (numerically stable diff form)
            dr_dot_i    = dv_i
            dv_dot_i    = self.acc_rhs_rel(r_L, dr_i, m_L, m_i, a_ctrl_L, a_ctrl_i, et)

            # Rotational kinematics (absolute attitude)
            q_dot_i     = self.attitude_kin_rhs(q_i, w_i)

            # Rotational dynamics
            w_dot_i     = self.attitude_dyn_rhs(w_i, tau_ctrl_i, J_i, J_dot_i)

            xdot[self._slice_x(i, self.dim_x_sc)] = np.concatenate(
                [dr_dot_i, dv_dot_i, q_dot_i, w_dot_i, [m_dot_i]]
            )

        return xdot

    # ── Translational acceleration ───────────────────────────────────────

    def acc_rhs_abs(self, r_I, m, area, a_ctrl, et):
        """
        Total absolute acceleration on the Leader (ICRF) [km/s^2].
        """
        a_tot = (self.acc_grav_abs(r_I, et)
                 + self.acc_srp_abs(r_I, m, area, et)
                 + self.acc_grav_isc_abs()
                 + self.acc_ion_abs()
                 + self.acc_p_abs()
                 + a_ctrl)
        return a_tot

    def acc_rhs_rel(self, r_L, dr, m_L, m_F, a_ctrl_L, a_ctrl_F, et):
        """
        Total differential acceleration of a Follower w.r.t. the Leader
        (ICRF) [km/s^2]. Each contribution is formed at well-conditioned
        scale (body-by-body for gravity; direct subtraction for SRP since
        it's tiny in absolute magnitude).
        """
        a_tot = (self.acc_grav_rel(r_L, dr, et)
                 + self.acc_srp_rel(r_L, dr, m_L, m_F, et)
                 + self.acc_grav_isc_rel()
                 + self.acc_ion_rel()
                 + self.acc_p_rel()
                 + (a_ctrl_F - a_ctrl_L))
        return a_tot

    def acc_grav_abs(self, r_I, et):
        """
        N-body gravitational acceleration at absolute position r_I [km] in ICRF.
        """
        v_dot = np.zeros(3)
        for body, mu in self.env.GM.items():
            r_b = self.env.body_position(body, et)
            dr  = r_I - r_b
            v_dot -= mu * dr / np.linalg.norm(dr) ** 3
        return v_dot

    def acc_grav_rel(self, r_L, dr, et):
        """
        Differential N-body gravity on a Follower at r_F = r_L + δr,
        computed body-by-body as

            a_rel = Σ_b [ a_b(r_L + δr) - a_b(r_L) ]

        with each per-body subtraction performed at well-conditioned scale
        (d1 = r_L - r_b is reused; d2 = d1 + δr is never reconstructed from
        SSB-relative absolutes).
        """
        a = np.zeros(3)
        for body, mu in self.env.GM.items():
            r_b = self.env.body_position(body, et)
            d1  = r_L - r_b                  # body -> Leader
            d2  = d1 + dr                    # body -> Follower
            a  += -mu * (d2 / np.linalg.norm(d2) ** 3
                       - d1 / np.linalg.norm(d1) ** 3)
        return a

    def acc_srp_abs(self, r_I, m, area, et):
        """
        Cannonball SRP at absolute position r_I, body of mass m, projected
        area `area`, reflectivity self.param.c_reflect. [km/s^2 in ICRF]
        """
        r_sun  = self.env.body_position("SUN", et)
        d      = r_I - r_sun
        d_norm = np.linalg.norm(d)
        u_hat  = d / d_norm
        P_at_r = self.param.P_SUN * (self.param.R_SUN_REF_KM / d_norm) ** 2
        # 1e-3 converts m/s^2 -> km/s^2 (P in N/m^2 = kg/(m·s^2))
        a_mag  = self.param.c_reflect * (area / m) * P_at_r * 1.0e-3
        return a_mag * u_hat

    def acc_srp_rel(self, r_L, dr, m_L, m_F, et):
        """
        Differential cannonball SRP: a_srp(Follower) - a_srp(Leader).
        SRP at 1 AU is ~1e-13 km/s^2 — the literal difference is fine
        numerically (no large-number cancellation).
        """
        a_F = self.acc_srp_abs(r_L + dr, m_F, self.param.SRP_area_F, et)
        a_L = self.acc_srp_abs(r_L,      m_L, self.param.SRP_area_L, et)
        return a_F - a_L

    def acc_grav_isc_abs(self):
        # TODO: inter-spacecraft gravity (currently negligible)
        return np.zeros(3)

    def acc_grav_isc_rel(self):
        # TODO
        return np.zeros(3)

    def acc_ion_abs(self):
        # TODO: ion engine continuous thrust (off)
        return np.zeros(3)

    def acc_ion_rel(self):
        # TODO
        return np.zeros(3)

    def acc_p_abs(self):
        # TODO: stochastic process noise (off)
        return np.zeros(3)

    def acc_p_rel(self):
        # TODO
        return np.zeros(3)

    # ── Attitude ─────────────────────────────────────────────────────────

    def attitude_kin_rhs(self, q, omega):
        """
        Quaternion kinematics: q_dot = (1/2) Ω(ω) q.
        """
        return 0.5 * Omega_omega(omega) @ q

    def attitude_dyn_rhs(self, omega, tau_ctrl, J, J_dot):
        """
        Rigid-body rotational dynamics in body frame with time-varying inertia:

            J ω̇ = τ_ctrl + τ_SRP + τ_p − ω × (J ω) − J̇ ω

        Solve for ω̇. `J` and `J_dot` are passed in because they depend on the
        current (time-varying) mass of the spacecraft.
        """
        tau_tot = (tau_ctrl
                   + self.attitude_tau_SRP()
                   + self.attitude_tau_p()
                   - np.cross(omega, J @ omega)
                   - J_dot @ omega)
        return np.linalg.solve(J, tau_tot)

    def attitude_tau_SRP(self):
        # TODO: SRP torque (cp offset etc.) — currently zero
        return np.zeros(3)

    def attitude_tau_p(self):
        # TODO: stochastic torque disturbance — currently zero
        return np.zeros(3)

    # ── Mass & inertia derivatives ───────────────────────────────────────

    def mass_rhs(self, m, a_ctrl, Isp):
        """
        Rocket-equation mass flow from commanded thrust acceleration.

            |T| = m · |a_ctrl|              (Newton's 2nd law)
            ṁ   = − |T| / (Isp · g_0)       (rocket equation)

        Units are consistent in km: |a_ctrl| [km/s^2], g_0 [km/s^2],
        Isp [s] → ṁ [kg/s].
        """
        thrust_mag = m * np.linalg.norm(a_ctrl)         # [kg · km/s^2]
        return -thrust_mag / (Isp * self.param.G_0)     # [kg/s]

    def J_dot(self, m_dot, J_ring_per_mass):
        """
        Inertia tensor time derivative.

        Per the design doc (Eq. 23–24) the RING is the fuel-bearing section, so
        its mass m_r,i = m_i − m_c,i varies in time while the cylindrical bus
        mass m_c,i is constant. J_ring scales linearly with m_ring for fixed
        ring geometry:

            J_ring(t) = m_r,i(t) · J_ring_per_mass
            ⇒  J̇    = ṁ_r,i · J_ring_per_mass = ṁ_i · J_ring_per_mass

        (m_cylinder is constant, so ṁ_r,i = ṁ_i = ṁ.)
        """
        return m_dot * J_ring_per_mass

    @staticmethod
    def _J_current(m, m_cylinder, J_cylinder, J_ring_per_mass):
        """
        Reconstruct the current total inertia from the current total mass:

            m_r,i(t) = m(t) − m_cylinder          (constant bus mass)
            J(t)     = J_cylinder + m_r,i(t) · J_ring_per_mass
        """
        return J_cylinder + (m - m_cylinder) * J_ring_per_mass

    # ── Utilities ────────────────────────────────────────────────────────

    @staticmethod
    def _renormalize_quat(x_next, n_sc, dim_x_sc):
        for i in range(n_sc):
            q_slice = slice(dim_x_sc * i + 6, dim_x_sc * i + 10)
            x_next[q_slice] = x_next[q_slice] / np.linalg.norm(x_next[q_slice])
        return x_next

    @staticmethod
    def _slice_x(i, dim_x_sc):
        """State slice for spacecraft i (length dim_x_sc)."""
        return slice(dim_x_sc * i, dim_x_sc * (i + 1))

    @staticmethod
    def _slice_u(i, dim_u_sc):
        """Control slice for spacecraft i (length dim_u_sc)."""
        return slice(dim_u_sc * i, dim_u_sc * (i + 1))


##############################################
# Methods
##############################################


def inertia_ring(m_ring, r_in, r_out, h_ring) -> np.ndarray:
    """Hollow-cylindrical-ring (annular cylinder) inertia tensor."""
    J = np.zeros((3, 3))
    J[0, 0] = (1/12) * m_ring * (3 * (r_in**2 + r_out**2) + h_ring**2)
    J[1, 1] = J[0, 0]
    J[2, 2] = (1/2)  * m_ring * (r_in**2 + r_out**2)
    return J


def inertia_cylinder(m_cylinder, r_cylinder, h_cylinder) -> np.ndarray:
    """Solid cylinder inertia tensor about its principal axes."""
    J = np.zeros((3, 3))
    J[0, 0] = (1/12) * m_cylinder * (3 * r_cylinder**2 + h_cylinder**2)
    J[1, 1] = J[0, 0]
    J[2, 2] = (1/2)  * m_cylinder * r_cylinder**2
    return J


def inertia(m_ring, r_in, r_out, h_ring, J_cylinder) -> np.ndarray:
    """Combined ring + cylinder inertia tensor."""
    return inertia_ring(m_ring, r_in, r_out, h_ring) + J_cylinder


def mass_ring(m_init, m_cylinder):
    return m_init - m_cylinder


def initialize_state(formation, baseline, att_F, m_init_F, x_init_L, n_sc):
    """Build the stacked initial state. Followers are stored in RELATIVE form
    (δr, δv) w.r.t. the Leader."""

    if formation == "square planar" and att_F == "same as leader" and n_sc == 5:

        delta_r0_list = [
            np.array([ baseline,  0.0,     0.0]),   # follower_1: +x
            np.array([-baseline,  0.0,     0.0]),   # follower_2: -x
            np.array([ 0.0,       baseline, 0.0]),  # follower_3: +y
            np.array([ 0.0,      -baseline, 0.0]),  # follower_4: -y
        ]

        q_init_F = np.array([1.0, 0.0, 0.0, 0.0])
        w_init_F = np.array([0.001, 0.01, 0.0])

        x0 = [x_init_L]
        for dr0 in delta_r0_list:
            dv0 = np.zeros(3)
            x0.append(np.concatenate([dr0, dv0, q_init_F, w_init_F, [m_init_F]]))
        x = np.concatenate(x0)

    else:
        raise NotImplementedError(
            "Formation geometry or attitude initialization not implemented "
            "for the given parameters."
        )

    return x


##############################################
# Main Function
##############################################

def main():

    # Load SPICE kernels into environment
    env = SpiceEnv(KERNELS, BODIES, FRAME, ABCORR, OBSERVER)

    # Load Spacecraft Parameters and Constants
    param = Parameters()

    # ---------------------------------------
    # MODEL Settings:

    # Number of spacecrafts ( n_sc = (1 Leader) + (n_sc - 1 Followers) )
    n_sc = 5

    # State space dimensions
    dim_x_sc = 14           # [x, y, z, vx, vy, vz, q1, q2, q3, q4, wx, wy, wz, m]
    dim_y_sc = dim_x_sc     # same for now
    dim_u_sc = 6            # [fx, fy, fz, taux, tauy, tauz]

    dim_x = n_sc * dim_x_sc
    dim_y = n_sc * dim_y_sc
    dim_u = n_sc * dim_u_sc

    # Initialize the Classes
    sensor      = Sensor()
    ctrl        = Controller(dim_u)
    guidance    = Guidance(dim_y)
    plant       = Plant(env, param, n_sc, dim_x_sc, dim_u_sc)

    # ---------------------------------------
    # SIMULATION Parameters:

    # Initial Epoch (ET)
    et_init     = env.str2et("2026-05-12T00:00:00")

    # Leader Initial State (r0, v0, q0, w0, m0) — Webb-like halo around L2
    r_init_L    = np.array([-9.594503991242750e7, -1.098032827423822e8, -4.778858640538428e7])   # [km]
    v_init_L    = np.array([ 2.279303677102495e1, -1.735582913273474e1, -7.640659579588683e0])   # [km/s]
    q_init_L    = np.array([1.0, 0.0, 0.0, 0.0])
    w_init_L    = np.array([0.001, 0.01, 0.0])     # [rad/s]
    x_init_L    = np.concatenate((r_init_L, v_init_L, q_init_L, w_init_L,
                                  [param.m_init_L]))

    # Initial Formation
    formation   = "square planar"
    baseline    = 0.1                # [km] — 100 m
    att_F       = "same as leader"

    x_init = initialize_state(formation, baseline, att_F,
                              param.m_init_F, x_init_L, n_sc)

    # ── Initial-state printout ──────────────────────────────────────────
    print(f"\nEpoch  : {env.et2utc(et_init)}")
    print(f"n_sc   : {n_sc}  (1 leader + {n_sc - 1} followers)")
    print(f"Leader r0 : {x_init[0:3]}  km")
    print(f"Leader v0 : {x_init[3:6]}  km/s")
    print(f"Leader m0 : {x_init[13]}  kg")
    print(f"Initial follower baselines [m]:")
    for i in range(1, n_sc):
        dr_m = np.linalg.norm(x_init[dim_x_sc * i : dim_x_sc * i + 3]) * 1e3
        print(f"  follower_{i}: |δr0| = {dr_m:.3f} m")

    # ---------------------------------------
    # Time loop
    dt          = 200.0           # [s]
    t_tot       = 3 * 86400       # [s]
    n_steps     = int(t_tot / dt)
    et          = et_init
    x           = x_init

    t_hist        = np.zeros(n_steps + 1)
    X_hist        = np.zeros((n_steps + 1, dim_x))
    X_hist[0, :]  = x
    print_every   = max(1, n_steps // 20)

    print(f"\n--- Epoch-stepping simulation: {n_steps} steps of {dt:.1f} s "
          f"({n_steps * dt / 3600:.2f} h total) ---")

    for k in range(n_steps):

        # ------------  Sense ----------- #
        y_hat   = sensor.measure(x, et)

        # ------------  Plan  ----------- #
        y_ref   = guidance.reference(et)
        u       = ctrl.compute(y_hat, y_ref, et)

        # ------------   Act  ----------- #
        x_next  = plant.step(x, u, et, dt)

        # ------------  Data  ----------- #
        t_hist[k + 1]     = (k + 1) * dt
        X_hist[k + 1, :]  = x_next

        if (k + 1) % print_every == 0 or k == n_steps - 1:
            dr1_m = np.linalg.norm(x_next[dim_x_sc : dim_x_sc + 3]) * 1e3
            print(
                f"  k={k+1:5d}/{n_steps}  t={(k+1)*dt/3600:6.3f} h   "
                f"|r_L|={np.linalg.norm(x_next[0:3]):.6e} km   "
                f"|q_L|={np.linalg.norm(x_next[6:10]):.12f}   "
                f"|δr_1|={dr1_m:.6f} m"
            )

        x   = x_next
        et += dt

    # ── Per-follower diagnostic dump ────────────────────────────────────
    labels = ["leader"] + [f"follower_{i}" for i in range(1, n_sc)]

    print("\n--- Final baselines and drift over the run ---")
    for i in range(1, n_sc):
        dr_init  = X_hist[0,  dim_x_sc * i : dim_x_sc * i + 3]
        dr_final = X_hist[-1, dim_x_sc * i : dim_x_sc * i + 3]
        mag_init   = np.linalg.norm(dr_init)  * 1e3   # m
        mag_final  = np.linalg.norm(dr_final) * 1e3   # m
        drift      = mag_final - mag_init             # m  (signed)
        print(
            f"  {labels[i]:<11s} "
            f"|δr0| = {mag_init:.6f} m   "
            f"|δr(T)| = {mag_final:.9f} m   "
            f"Δ|δr| = {drift*1e6:+.3f} μm   "
            f"δr_final = [{dr_final[0]*1e3:+.6f}, {dr_final[1]*1e3:+.6f}, "
            f"{dr_final[2]*1e3:+.6f}] m"
        )

    # ── Pairwise follower trajectory differences ────────────────────────
    print("\n--- Pairwise follower-trajectory differences over full run ---")
    for i in range(1, n_sc):
        for j in range(i + 1, n_sc):
            dr_i = X_hist[:, dim_x_sc * i : dim_x_sc * i + 3]
            dr_j = X_hist[:, dim_x_sc * j : dim_x_sc * j + 3]
            diff_mag = np.max(np.abs(np.linalg.norm(dr_i, axis=1)
                                     - np.linalg.norm(dr_j, axis=1))) * 1e3
            diff_vec = np.max(np.linalg.norm(dr_i - dr_j, axis=1)) * 1e3
            print(
                f"  {labels[i]} vs {labels[j]}: "
                f"max ||δr_i| - |δr_j|| = {diff_mag:.3e} m, "
                f"max |δr_i - δr_j| = {diff_vec:.3e} m"
            )

    # ── Plots ───────────────────────────────────────────────────────────
    # Plotting utilities from main.py expect a list of objects with `.label`.
    # Wrap our flat label list in SimpleNamespace so they still work.
    spacecraft_for_plot = [SimpleNamespace(label=l) for l in labels]

    print("\nGenerating plots ...")
    plot_trajectory(
        t_hist     = t_hist,
        X_hist     = X_hist,
        et0        = et_init,
        spacecraft = spacecraft_for_plot,
    )
    plot_solar_system(
        et0        = et_init,
        duration   = t_hist[-1],
        X_hist     = X_hist,
        t_hist     = t_hist,
        spacecraft = spacecraft_for_plot,
    )
    plot_l2_rotating_frame_zoom(
        et0        = et_init,
        t_hist     = t_hist,
        X_hist     = X_hist,
        spacecraft = spacecraft_for_plot,
    )


if __name__ == "__main__":
    main()
