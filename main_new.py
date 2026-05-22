"""
LIFE Mission — High-Fidelity Truth Model (Translational + Rotational Dynamics)

N-spacecraft formation: 1 LEADER + (N-1) FOLLOWERS, leader-plus-relative form.

Per spacecraft state layout (14 components):
    x[0:3]    r or δr        position           [km]    (leader: absolute,
                                                         follower: relative to leader)
    x[3:6]    v or δv        velocity           [km/s]  (same convention)
    x[6:10]   q_I^B          quaternion (inertial → body, absolute)
    x[10:13]  ω_IB^B         angular rate, body frame              [rad/s]
    x[13]     m              total mass                            [kg]

Per spacecraft control (6 components):
    u[0:3]    a_ctrl^I       control acceleration, inertial        [km/s^2]
    u[3:6]    τ_ctrl^B       control torque, body frame            [N·m]

Spacecraft physical model (per role):
    Outer ring  (variable mass, propellant tanks)  +  inner solid cylinder
    (constant mass).  Both are concentric → total inertia adds directly with
    no parallel-axis term.  Only the ring mass changes with propellant burn,
    so J̇ comes entirely from the ring contribution.
"""

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

    # ── Universal constants ─────────────────────────────────────
    P_SUN           = 4.53e-6           # solar pressure at 1 AU   [N/m^2]
    R_SUN_AU        = 1.495978707e8     # 1 AU                     [km]
    G0              = 9.80665e-3        # standard gravity         [km/s^2]
    KM_PER_M        = 1.0e-3            # km / m

    # ── Spacecraft masses [kg] ──────────────────────────────────
    m_init_L        = 4000.0            # leader   total initial mass
    m_init_F        = 3000.0            # follower total initial mass
    m_cylinder_L    = 3000.0            # leader   cylinder (constant) mass
    m_cylinder_F    = 2000.0            # follower cylinder (constant) mass

    # ── Spacecraft geometry [m] ─────────────────────────────────
    h_cylinder_L    = 5.0               # leader   cylinder height
    h_cylinder_F    = 4.8               # follower cylinder height
    h_ring          = 1.2               # ring height (shared)
    r_cylinder      = 2.57              # cylinder radius (shared)
    r_in            = 2.7               # ring inner radius (shared)
    r_out           = 3.8               # ring outer radius (shared)

    # ── Propulsion & SRP ────────────────────────────────────────
    c_reflect       = 1.8               # SRP reflectivity coefficient (Webb-like)
    ISP_L           = 220.0             # leader   specific impulse [s]
    ISP_F           = 220.0             # follower specific impulse [s]

    # ── Derived (computed in __post_init__) ─────────────────────
    J_cylinder_L:   np.ndarray = field(init=False)
    J_cylinder_F:   np.ndarray = field(init=False)
    J_init_L:       np.ndarray = field(init=False)
    J_init_F:       np.ndarray = field(init=False)
    SRP_area_L:     float      = field(init=False)
    SRP_area_F:     float      = field(init=False)
    m_ring_init_L:  float      = field(init=False)
    m_ring_init_F:  float      = field(init=False)

    def __post_init__(self):
        # frozen=True forbids normal assignment — bypass via object.__setattr__
        s = lambda k, v: object.__setattr__(self, k, v)

        J_cyl_L = inertia_cylinder(self.m_cylinder_L, self.r_cylinder, self.h_cylinder_L)
        J_cyl_F = inertia_cylinder(self.m_cylinder_F, self.r_cylinder, self.h_cylinder_F)
        s("J_cylinder_L", J_cyl_L)
        s("J_cylinder_F", J_cyl_F)

        m_ring_L = mass_ring(self.m_init_L, self.m_cylinder_L)
        m_ring_F = mass_ring(self.m_init_F, self.m_cylinder_F)
        s("m_ring_init_L", m_ring_L)
        s("m_ring_init_F", m_ring_F)

        s("J_init_L", inertia(m_ring_L, self.r_in, self.r_out, self.h_ring, J_cyl_L))
        s("J_init_F", inertia(m_ring_F, self.r_in, self.r_out, self.h_ring, J_cyl_F))

        # Cylindrical "cannonball" projected area (rectangle 2r × h)
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
    Truth model dynamics for an N-spacecraft formation.

      - Spacecraft 0 = LEADER:    absolute (r, v) in SSB-centred ICRF.
      - Spacecraft i > 0 = FOLLOWER: relative (δr, δv) w.r.t. leader in ICRF,
        plus ABSOLUTE attitude.  Relative form avoids the ~mm-level numerical
        floor incurred by differencing two ~1.5e8 km SSB-relative vectors.

    Per-spacecraft physical model:
        outer ring   (variable mass — fuel)
        inner cylinder (constant mass)
      Inertia tensor is computed as J(m) = J_cyl + J_ring(m_ring(m)) and
      tracks m through the burn.  J̇ comes solely from the ring term.
    """

    def __init__(self, env, param, n_sc, dim_x_sc, dim_u_sc):
        self.env       = env
        self.param     = param
        self.n_sc      = n_sc
        self.dim_x_sc  = dim_x_sc
        self.dim_u_sc  = dim_u_sc

    # ── Single-step propagator ──────────────────────────────────────────────

    def step(self, x, u, t, dt,
             rtol = 1e-12, atol = 1e-12, renormalize = True):

        sol = solve_ivp(self.x_dot, (0.0, dt), x,
                        method  = "DOP853",
                        args    = (t, u),
                        rtol    = rtol,
                        atol    = atol,
                        t_eval  = [dt])

        if not sol.success:
            raise RuntimeError(f"Integration failed: {sol.message}")

        x_next = sol.y[:, -1]

        if renormalize:
            x_next = self._renormalize_quat(x_next, self.n_sc, self.dim_x_sc)

        return x_next

    # ── ODE RHS ─────────────────────────────────────────────────────────────

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

        # Absolute Translational Dynamics
        r_dot_L     = v_L
        v_dot_L     = self.acc_rhs_abs(r_L, m_L, a_ctrl_L, 'L', et)

        # Mass Derivative (Tsiolkovsky)
        m_dot_L     = self.mass_rhs(m_L, a_ctrl_L, 'L')

        # Rotational Kinematics + Dynamics
        q_dot_L     = self.attitude_kin_rhs(q_L, w_L)
        w_dot_L     = self.attitude_dyn_rhs(w_L, m_L, m_dot_L, tau_ctrl_L, 'L')

        xdot[self._slice_x(0, self.dim_x_sc)] = np.concatenate(
            [r_dot_L, v_dot_L, q_dot_L, w_dot_L, [m_dot_L]]
        )

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

            delta_r     = x_i[0:3]
            delta_v     = x_i[3:6]
            q_i         = x_i[6:10]
            w_i         = x_i[10:13]
            m_i         = x_i[13]

            a_ctrl_i    = u_i[0:3]
            tau_ctrl_i  = u_i[3:6]

            # Relative Translational Dynamics (numerically stable per-body diff)
            r_dot_i     = delta_v
            v_dot_i     = self.acc_rhs_rel(r_L, delta_r,
                                           m_L, m_i,
                                           a_ctrl_L, a_ctrl_i, et)

            # Mass Derivative
            m_dot_i     = self.mass_rhs(m_i, a_ctrl_i, 'F')

            # Rotational Kinematics + Dynamics (attitude is absolute)
            q_dot_i     = self.attitude_kin_rhs(q_i, w_i)
            w_dot_i     = self.attitude_dyn_rhs(w_i, m_i, m_dot_i, tau_ctrl_i, 'F')

            xdot[self._slice_x(i, self.dim_x_sc)] = np.concatenate(
                [r_dot_i, v_dot_i, q_dot_i, w_dot_i, [m_dot_i]]
            )

        return xdot

    # ── Translational dynamics ──────────────────────────────────────────────

    def acc_rhs_abs(self, r_I, m, a_ctrl, role, et):
        """Total absolute acceleration on a spacecraft (used by leader)."""
        a_tot = (self.acc_grav_abs(r_I, et)
                 + self.acc_srp_abs(r_I, m, role, et)
                 + self.acc_grav_isc_abs()
                 + self.acc_ion_abs()
                 + self.acc_p_abs()
                 + a_ctrl)
        return a_tot

    def acc_rhs_rel(self, r_L, delta_r, m_L, m_F, a_ctrl_L, a_ctrl_F, et):
        """
        Total DIFFERENTIAL acceleration on a follower w.r.t. the leader.
        Each per-body subtraction is evaluated at well-conditioned scale to
        avoid catastrophic cancellation when differencing SSB-scale vectors.
        """
        da_tot = (self.acc_grav_rel(r_L, delta_r, et)
                  + self.acc_srp_rel(r_L, delta_r, m_L, m_F, et)
                  + self.acc_grav_isc_rel()
                  + self.acc_ion_rel()
                  + self.acc_p_rel()
                  + (a_ctrl_F - a_ctrl_L))
        return da_tot

    def acc_grav_abs(self, r_I, et):
        """N-body Solar System gravity at absolute r_I [km]. → [km/s^2]"""
        v_dot = np.zeros(3)
        for body, mu in self.env.GM.items():
            r_b   = self.env.body_position(body, et)
            dr    = r_I - r_b
            v_dot -= mu * dr / np.linalg.norm(dr) ** 3
        return v_dot

    def acc_grav_rel(self, r_L, delta_r, et):
        """
        Differential N-body gravity on follower at r_L + δr.  Evaluated as
            Σ_b [ a_b(r_L + δr) − a_b(r_L) ]
        with each term kept at body-relative scale (d2 = d1 + δr, never
        d2 = r_F − r_b reconstructed from SSB-based absolutes).
        """
        da = np.zeros(3)
        for body, mu in self.env.GM.items():
            r_b = self.env.body_position(body, et)
            d1  = r_L - r_b                       # body → leader
            d2  = d1 + delta_r                    # body → follower
            da -= mu * (d2 / np.linalg.norm(d2) ** 3
                      - d1 / np.linalg.norm(d1) ** 3)
        return da

    def acc_srp_abs(self, r_I, m, role, et):
        """
        Cannonball SRP at absolute position r_I; body of mass m, projected
        area A(role), reflectivity c_reflect.  → [km/s^2]
        """
        A   = self.param.SRP_area_L if role == 'L' else self.param.SRP_area_F
        c_R = self.param.c_reflect

        r_sun  = self.env.body_position("SUN", et)
        d      = r_I - r_sun
        d_norm = np.linalg.norm(d)
        u_hat  = d / d_norm                                            # sun → s/c (outward)
        P_at_r = self.param.P_SUN * (self.param.R_SUN_AU / d_norm) ** 2
        a_mag  = c_R * (A / m) * P_at_r * self.param.KM_PER_M
        return a_mag * u_hat

    def acc_srp_rel(self, r_L, delta_r, m_L, m_F, et):
        """
        Differential cannonball SRP: SRP(follower) − SRP(leader).
        Done as a literal difference of two SRP evaluations — safe because
        SRP itself is tiny (~1e-13 km/s^2), no large-number cancellation.
        Two physical sources of mismatch:
          (i)  position-dependent (1/r^2, sun-line) — ~|δr|/AU of nominal
          (ii) parameter-dependent (m, A may differ between leader/follower)
        """
        a_F = self.acc_srp_abs(r_L + delta_r, m_F, 'F', et)
        a_L = self.acc_srp_abs(r_L,           m_L, 'L', et)
        return a_F - a_L

    def acc_grav_isc_abs(self):
        # TODO: inter-spacecraft gravity (negligible at 100 m, 3000 kg)
        return np.zeros(3)

    def acc_grav_isc_rel(self):
        # TODO
        return np.zeros(3)

    def acc_ion_abs(self):
        # TODO: continuous ion thrust (separate from a_ctrl chemical impulse)
        return np.zeros(3)

    def acc_ion_rel(self):
        # TODO
        return np.zeros(3)

    def acc_p_abs(self):
        # TODO: process noise
        return np.zeros(3)

    def acc_p_rel(self):
        # TODO
        return np.zeros(3)

    # ── Rotational dynamics ─────────────────────────────────────────────────

    def attitude_kin_rhs(self, q, omega):
        """Quaternion kinematics: q̇ = ½ Ω(ω) q."""
        return 0.5 * Omega_omega(omega) @ q

    def attitude_dyn_rhs(self, omega, m, m_dot, tau_ctrl, role):
        """
        Euler's rotational equation with a TIME-VARYING inertia:
            J(m) ω̇  =  τ_SRP + τ_p + τ_ctrl  −  ω × (J ω)  −  J̇ ω

        J(m)   = J_cyl(role) + J_ring(m − m_cyl(role))   ← cylinder constant
        J̇      = ∂J_ring/∂m_ring · ṁ_ring   = ∂J_ring/∂m_ring · ṁ
                 (m_cyl constant ⇒ ṁ_ring = ṁ; J_ring linear in m_ring)
        """
        J     = self._inertia_now(m, role)
        J_dot = self._inertia_dot_now(m_dot)
        J_inv = np.linalg.inv(J)

        tau_tot = (self.attitude_tau_SRP()
                   + self.attitude_tau_p()
                   + tau_ctrl
                   - np.cross(omega, J @ omega)
                   - J_dot @ omega)

        return J_inv @ tau_tot

    def attitude_tau_SRP(self):
        # TODO: SRP torque (depends on attitude and CoP offset)
        return np.zeros(3)

    def attitude_tau_p(self):
        # TODO: process-noise torque
        return np.zeros(3)

    # ── Mass dynamics ───────────────────────────────────────────────────────

    def mass_rhs(self, m, a_ctrl, role):
        """
        Tsiolkovsky mass-flow:   ṁ = −|F| / (Isp · g0)
        with thrust F = m · a_ctrl (control is commanded as acceleration).

        Units check (km / s / kg system):
            |F| = m · |a|    [kg · km/s^2]  ≡ [kN]
            Isp · g0         [s · km/s^2]   = [km/s]
            ṁ = −|F| / (Isp · g0)            [kg / s]   ✓

        Only the ring (propellant) mass changes; the cylinder is structural.
        The scalar m carried in the state IS the total mass m_cyl + m_ring,
        and since m_cyl is constant, ṁ = ṁ_ring.
        """
        Isp   = self.param.ISP_L if role == 'L' else self.param.ISP_F
        F_mag = m * np.linalg.norm(a_ctrl)
        return -F_mag / (Isp * self.param.G0)

    # ── Inertia helpers (mass-dependent) ────────────────────────────────────

    def _inertia_now(self, m, role):
        """Current J_B(m) for the given role (cylinder + ring(m − m_cyl))."""
        if role == 'L':
            m_cyl, J_cyl = self.param.m_cylinder_L, self.param.J_cylinder_L
        else:
            m_cyl, J_cyl = self.param.m_cylinder_F, self.param.J_cylinder_F
        m_ring = m - m_cyl
        return inertia(m_ring, self.param.r_in, self.param.r_out,
                       self.param.h_ring, J_cyl)

    def _inertia_dot_now(self, m_dot):
        """
        dJ/dt.  Cylinder mass is constant ⇒ only the ring contributes.
        J_ring is linear in m_ring, so J̇_ring is the same expression with
        m_ring → ṁ_ring (= ṁ).  Geometry (r_in, r_out, h_ring) is fixed —
        propellant always fills the same annular volume in this model — and
        is shared between leader and follower, so no role dispatch is needed.
        """
        return inertia_dot(m_dot, self.param.r_in, self.param.r_out, self.param.h_ring)

    # ── Slicing / quaternion utilities ──────────────────────────────────────

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

# Total inertia tensor: outer annular ring (variable mass) + inner solid cylinder.
def inertia(m_ring, r_in, r_out, h_ring, J_cylinder) -> np.ndarray:
    J_ring        = np.zeros((3, 3))
    J_ring[0, 0]  = (1/12) * m_ring * (3 * (r_in**2 + r_out**2) + h_ring**2)
    J_ring[1, 1]  = J_ring[0, 0]
    J_ring[2, 2]  = (1/2)  * m_ring * (r_in**2 + r_out**2)
    return J_ring + J_cylinder


# Time derivative of total inertia: only the ring's mass changes with burn.
def inertia_dot(m_dot, r_in, r_out, h_ring) -> np.ndarray:
    J_dot        = np.zeros((3, 3))
    J_dot[0, 0]  = (1/12) * m_dot * (3 * (r_in**2 + r_out**2) + h_ring**2)
    J_dot[1, 1]  = J_dot[0, 0]
    J_dot[2, 2]  = (1/2)  * m_dot * (r_in**2 + r_out**2)
    return J_dot


# Inertia tensor of the constant-mass inner cylinder (z = symmetry axis).
def inertia_cylinder(m_cylinder, r_cylinder, h_cylinder) -> np.ndarray:
    J_cylinder         = np.zeros((3, 3))
    J_cylinder[0, 0]   = (1/12) * m_cylinder * (3 * r_cylinder**2 + h_cylinder**2)
    J_cylinder[1, 1]   = J_cylinder[0, 0]
    J_cylinder[2, 2]   = (1/2)  * m_cylinder * r_cylinder**2
    return J_cylinder


# Propellant ring mass at a given total mass.
def mass_ring(m_init, m_cylinder):
    return m_init - m_cylinder


# Initialize the stacked state vector for leader + followers.
def initialize_state(formation, baseline, att_F, m_init_F, x_init_L, n_sc):

    if formation == "square planar" and att_F == "same as leader" and n_sc == 5:

        delta_r0_list = [
            np.array([ baseline,  0.0,      0.0]),   # follower_1: +x
            np.array([-baseline,  0.0,      0.0]),   # follower_2: -x
            np.array([ 0.0,       baseline, 0.0]),   # follower_3: +y
            np.array([ 0.0,      -baseline, 0.0]),   # follower_4: -y
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
            "Formation geometry or attitude initialization not implemented."
        )

    return x


# Build the per-spacecraft list of label/J_B namespaces consumed by the
# plotting layer (kept minimal — only the two attributes the plots use).
def build_plot_spacecraft(param, n_sc):
    sc = [SimpleNamespace(label="leader", J_B=param.J_init_L)]
    for i in range(1, n_sc):
        sc.append(SimpleNamespace(label=f"follower_{i}", J_B=param.J_init_F))
    return sc


##############################################
# Main Function
##############################################

def main():

    # ─── Environment + Parameters ────────────────────────────────────
    env   = SpiceEnv(KERNELS, BODIES, FRAME, ABCORR, OBSERVER)
    param = Parameters()

    # ─── MODEL Settings ──────────────────────────────────────────────
    # n_sc = 1 leader + (n_sc − 1) followers
    n_sc       = 5

    # State / measurement / control dimensions
    dim_x_sc   = 14       # [x, y, z, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz, m]
    dim_y_sc   = dim_x_sc # same for now
    dim_u_sc   = 6        # [ax, ay, az, taux, tauy, tauz]

    dim_x      = n_sc * dim_x_sc
    dim_y      = n_sc * dim_y_sc
    dim_u      = n_sc * dim_u_sc

    # ─── Pipeline classes ────────────────────────────────────────────
    sensor     = Sensor()
    ctrl       = Controller(dim_u)
    guidance   = Guidance(dim_y)
    plant      = Plant(env, param, n_sc, dim_x_sc, dim_u_sc)

    # ─── SIMULATION Parameters ───────────────────────────────────────
    et_init    = env.str2et("2026-05-12T00:00:00")

    # Leader Initial State (r0, v0, q0, w0, m0)  — Webb-like halo about L2
    r_init_L   = np.array([-9.594503991242750e7, -1.098032827423822e8, -4.778858640538428e7])   # [km]
    v_init_L   = np.array([ 2.279303677102495e1, -1.735582913273474e1, -7.640659579588683e0])   # [km/s]
    q_init_L   = np.array([1.0, 0.0, 0.0, 0.0])
    w_init_L   = np.array([0.001, 0.01, 0.0])
    x_init_L   = np.concatenate((r_init_L, v_init_L, q_init_L, w_init_L, [param.m_init_L]))

    # Initial formation
    formation  = "square planar"
    baseline   = 0.1                       # 100 m
    att_F      = "same as leader"

    x_init     = initialize_state(formation, baseline, att_F,
                                  param.m_init_F, x_init_L, n_sc)

    # ─── Initial-state printout ──────────────────────────────────────
    print(f"\nEpoch  : {env.et2utc(et_init)}")
    print(f"N_SC   : {n_sc} (1 leader + {n_sc - 1} followers)")
    print(f"Leader r0 : {x_init[0:3]}  km")
    print(f"Leader v0 : {x_init[3:6]}  km/s")
    print(f"Leader m0 : {x_init[13]}  kg")
    print("Baselines (initial, [m]):")
    for i in range(1, n_sc):
        dr_m = np.linalg.norm(x_init[dim_x_sc*i : dim_x_sc*i + 3]) * 1e3
        print(f"  follower_{i}: |δr0| = {dr_m:.3f} m")

    # ─── Time grid + history buffers ─────────────────────────────────
    dt          = 200.0
    t_tot       = 3 * 86400
    n_steps     = int(t_tot / dt)
    print_every = max(1, n_steps // 20)

    t_hist          = np.zeros(n_steps + 1)
    X_hist          = np.zeros((n_steps + 1, dim_x))
    X_hist[0, :]    = x_init

    print(f"\n--- Epoch-stepping simulation: {n_steps} steps of {dt:.1f} s "
          f"({n_steps * dt / 3600:.2f} h total) ---")

    # ─── Main control loop ───────────────────────────────────────────
    et = et_init
    x  = x_init
    for k in range(n_steps):

        # ------------  Sense ----------- #
        y_hat   = sensor.measure(x, et)

        # ------------  Plan  ----------- #
        y_ref   = guidance.reference(et)
        u       = ctrl.compute(y_hat, y_ref, et)

        # ------------   Act  ----------- #
        x_next  = plant.step(x, u, et, dt)

        # ------------  Data  ----------- #
        t_hist[k + 1]    = (k + 1) * dt
        X_hist[k + 1, :] = x_next

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

    # ─── Per-follower diagnostic dump ────────────────────────────────
    print("\n--- Final baselines and drift over the run ---")
    for i in range(1, n_sc):
        dr_init   = X_hist[0,  dim_x_sc*i : dim_x_sc*i + 3]
        dr_final  = X_hist[-1, dim_x_sc*i : dim_x_sc*i + 3]
        mag_init  = np.linalg.norm(dr_init)  * 1e3
        mag_final = np.linalg.norm(dr_final) * 1e3
        drift     = mag_final - mag_init
        print(
            f"  follower_{i:<2d} "
            f"|δr0| = {mag_init:.6f} m   "
            f"|δr(T)| = {mag_final:.9f} m   "
            f"Δ|δr| = {drift*1e6:+.3f} μm   "
            f"δr_final = [{dr_final[0]*1e3:+.6f}, {dr_final[1]*1e3:+.6f}, "
            f"{dr_final[2]*1e3:+.6f}] m"
        )

    # ─── Pairwise follower comparison ────────────────────────────────
    print("\n--- Pairwise follower-trajectory differences over full run ---")
    for i in range(1, n_sc):
        for j in range(i + 1, n_sc):
            dr_i     = X_hist[:, dim_x_sc*i : dim_x_sc*i + 3]
            dr_j     = X_hist[:, dim_x_sc*j : dim_x_sc*j + 3]
            diff_mag = np.max(np.abs(np.linalg.norm(dr_i, axis=1)
                                     - np.linalg.norm(dr_j, axis=1))) * 1e3
            diff_vec = np.max(np.linalg.norm(dr_i - dr_j, axis=1)) * 1e3
            print(
                f"  follower_{i} vs follower_{j}: "
                f"max ||δr_i| - |δr_j|| = {diff_mag:.3e} m, "
                f"max |δr_i - δr_j| = {diff_vec:.3e} m"
            )

    # ─── Plots ───────────────────────────────────────────────────────
    print("\nGenerating plots ...")
    spacecraft = build_plot_spacecraft(param, n_sc)
    plot_trajectory(
        t_hist     = t_hist,
        X_hist     = X_hist,
        et0        = et_init,
        spacecraft = spacecraft,
    )
    plot_solar_system(
        et0        = et_init,
        duration   = t_hist[-1],
        X_hist     = X_hist,
        t_hist     = t_hist,
        spacecraft = spacecraft,
    )
    plot_l2_rotating_frame_zoom(
        et0        = et_init,
        t_hist     = t_hist,
        X_hist     = X_hist,
        spacecraft = spacecraft,
    )


if __name__ == "__main__":
    main()
