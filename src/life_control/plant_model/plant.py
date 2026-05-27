import numpy as np
from scipy.integrate import solve_ivp

from life_control.utils.other import Omega_omega
from life_control.utils.coordinate_trafos import quat_to_CTM
from life_control.plant_model.thrusters import (
    THRUSTER_POSITIONS,
    THRUSTER_NORMALS,
    B_F,
    B_TAU,
    N_THRUSTERS,
    clamp_thrust,
    force_torque_body,
)

from life_control.utils.physics import J_ring_unit, J_cylinder_unit

import life_control.utils.constants as const 


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
        T_cmd_L     = u[self._slice_u(0, self.dim_u_sc)]

        r_L         = x_L[0:3]
        v_L         = x_L[3:6]
        q_L         = x_L[6:10]
        w_L         = x_L[10:13]
        m_prop_L    = x_L[13]

        # Saturate thrust to physical range [0, T_max] N per thruster.
        # The integrator may probe the RHS at small sub-steps; saturating here
        # keeps the dynamics consistent with what the actuator can deliver.
        T_L         = clamp_thrust(T_cmd_L, self.param.T_MAX)

        # Body-frame net force [N] and torque [N·m] from the allocation.
        # If the propellant tank is dry, no actuator output is possible.
        if m_prop_L <= 0.0:
            T_L     = np.zeros_like(T_L)
        F_B_L, tau_ctrl_L = force_torque_body(T_L)

        # Rotate body-frame force to inertial frame and convert to acceleration.
        # m_total is needed here, not m_prop.
        # quat_to_CTM(q) returns C_I^B (inertial -> body); its transpose maps
        # body -> inertial:   v^I = C_I^B.T @ v^B.
        m_total_L   = self._m_total(m_prop_L, 'L')
        F_I_L       = quat_to_CTM(q_L).T @ F_B_L                       # [N]
        # a_ctrl in [km/s^2]:  (F[N] / m[kg]) * (1e-3 km/m)
        a_ctrl_L    = (F_I_L / m_total_L) * const.KM_PER_M

        # Absolute Translational Dynamics
        r_dot_L     = v_L
        v_dot_L     = self.acc_rhs_abs(r_L, m_total_L, a_ctrl_L, 'L', et)

        # Mass Derivative (Tsiolkovsky, per-thruster sum)
        m_dot_L     = self.mass_rhs(m_prop_L, T_L, 'L')

        # Rotational Kinematics + Dynamics
        q_dot_L     = self.attitude_kin_rhs(q_L, w_L)
        w_dot_L     = self.attitude_dyn_rhs(w_L, m_prop_L, m_dot_L, tau_ctrl_L, 'L')

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
            T_cmd_i     = u[self._slice_u(i, self.dim_u_sc)]

            delta_r     = x_i[0:3]
            delta_v     = x_i[3:6]
            q_i         = x_i[6:10]
            w_i         = x_i[10:13]
            m_prop_i    = x_i[13]

            T_i         = clamp_thrust(T_cmd_i, self.param.T_MAX)
            if m_prop_i <= 0.0:
                T_i     = np.zeros_like(T_i)
            F_B_i, tau_ctrl_i = force_torque_body(T_i)

            m_total_i   = self._m_total(m_prop_i, 'F')
            F_I_i       = quat_to_CTM(q_i).T @ F_B_i                   # [N]
            a_ctrl_i    = (F_I_i / m_total_i) * const.KM_PER_M

            # Relative Translational Dynamics (numerically stable per-body diff)
            r_dot_i     = delta_v
            v_dot_i     = self.acc_rhs_rel(r_L, delta_r,
                                           m_total_L, m_total_i,
                                           a_ctrl_L, a_ctrl_i, et)

            # Mass Derivative
            m_dot_i     = self.mass_rhs(m_prop_i, T_i, 'F')

            # Rotational Kinematics + Dynamics (attitude is absolute)
            q_dot_i     = self.attitude_kin_rhs(q_i, w_i)
            w_dot_i     = self.attitude_dyn_rhs(w_i, m_prop_i, m_dot_i, tau_ctrl_i, 'F')

            xdot[self._slice_x(i, self.dim_x_sc)] = np.concatenate(
                [r_dot_i, v_dot_i, q_dot_i, w_dot_i, [m_dot_i]]
            )

        return xdot

    # ── Translational dynamics ──────────────────────────────────────────────

    def acc_rhs_abs(self, r_I, m_total, a_ctrl, role, et):
        """Total absolute acceleration on a spacecraft (used by leader).
        `m_total` is the full spacecraft mass (cyl + ring_dry + prop) needed
        by mass-dependent perturbations such as SRP.  `a_ctrl` is the
        ALREADY-COMPUTED inertial-frame control acceleration produced by the
        body-frame thrusters (see Plant.x_dot)."""
        a_tot = (self.acc_grav_abs(r_I, et)
                 + self.acc_srp_abs(r_I, m_total, role, et)
                 + self.acc_grav_isc_abs()
                 + self.acc_ion_abs()
                 + self.acc_p_abs()
                 + a_ctrl)
        return a_tot

    def acc_rhs_rel(self, r_L, delta_r, m_total_L, m_total_F, a_ctrl_L, a_ctrl_F, et):
        """
        Total DIFFERENTIAL acceleration on a follower w.r.t. the leader.
        Each per-body subtraction is evaluated at well-conditioned scale to
        avoid catastrophic cancellation when differencing SSB-scale vectors.
        Both `m_total_L` and `m_total_F` are full spacecraft masses.
        """
        da_tot = (self.acc_grav_rel(r_L, delta_r, et)
                  + self.acc_srp_rel(r_L, delta_r, m_total_L, m_total_F, et)
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

    def acc_srp_abs(self, r_I, m_total, role, et):
        """
        Cannonball SRP at absolute position r_I; body of TOTAL mass `m_total`,
        projected area A(role), reflectivity c_reflect.  → [km/s^2]
        """
        if role == 'L':
            A   = self.param.SRP_area_L
        elif role == 'F':
            A   = self.param.SRP_area_F
        else: 
            raise ValueError(f"Invalid role {role!r}: expected 'L' or 'F'")

        c_R = self.param.c_reflect

        r_sun  = self.env.body_position("SUN", et)
        d      = r_I - r_sun
        d_norm = np.linalg.norm(d)
        u_hat  = d / d_norm                                            # sun → s/c (outward)
        P_at_r = const.P_SUN * (const.R_SUN_AU / d_norm) ** 2
        a_mag  = c_R * (A / m_total) * P_at_r * const.KM_PER_M
        return a_mag * u_hat

    def acc_srp_rel(self, r_L, delta_r, m_total_L, m_total_F, et):
        """
        Differential cannonball SRP: SRP(follower) − SRP(leader).
        Done as a literal difference of two SRP evaluations — safe because
        SRP itself is tiny (~1e-13 km/s^2), no large-number cancellation.
        Two physical sources of mismatch:
          (i)  position-dependent (1/r^2, sun-line) — ~|δr|/AU of nominal
          (ii) parameter-dependent (m, A may differ between leader/follower)
        """
        a_F = self.acc_srp_abs(r_L + delta_r, m_total_F, 'F', et)
        a_L = self.acc_srp_abs(r_L,           m_total_L, 'L', et)
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

    def attitude_dyn_rhs(self, omega, m_prop, m_dot, tau_ctrl, role):
        """
        Euler's rotational equation with a TIME-VARYING inertia:
            J(m_prop) ω̇  =  τ_SRP + τ_p + τ_ctrl  −  ω × (J ω)  −  J̇ ω

        J(m_prop) = J_cyl(role) + (m_ring_dry(role) + m_prop) · K_ring
        J̇         = ṁ_prop · K_ring
                    (cylinder and dry ring are constant — only the propellant
                     mass varies — so the time-derivative comes from that term)
        `m_dot` here is the time-derivative of the propellant mass (= state's
        m index), which equals the total mass time-derivative.
        """
        J     = self._inertia_now(m_prop, role)
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

    def mass_rhs(self, m_prop, T_cmd, role):
        """
        Per-thruster Tsiolkovsky mass flow, matching design-doc eq. (25):

            ṁ_prop = - (Σ_l T_l) / (Isp · g0)

        with `T_cmd` in newtons (already clamped to [0, T_max] by the caller).
        Each thruster's individual magnitude contributes additively — this
        correctly captures the propellant cost of opposing thruster firings
        used for torque generation.

        Unit handling (km / s / kg system):
            T_l       [N] = [kg·m/s²]              ← convert N → kN to match
            Isp · g0  [s · km/s²] = [km/s]
            ṁ        [kg/s]
        We do the N→kN conversion explicitly so the formula reads cleanly.

        When the propellant tank is empty, ṁ = 0 by clamping; the caller
        should also zero the thrust command (which it does in `x_dot`).
        """
        
        # Σ |T_l| in newtons → kN
        T_sum_kN = float(np.sum(T_cmd)) / const.N_PER_KN
        return - T_sum_kN / (self.param.ISP * const.G0)

    # ── Mass / inertia helpers ──────────────────────────────────────────────

    def _m_total(self, m_prop, role):
        """Total spacecraft mass = m_cyl + m_ring_dry + m_prop."""
        if role == 'L':
            return self.param.m_cylinder_L + self.param.m_ring_dry_L + max(m_prop, 0.0)
        elif role == 'F':
            return self.param.m_cylinder_F + self.param.m_ring_dry_F + max(m_prop, 0.0)
        else:
            raise ValueError(f"Invalid role {role!r}: expected 'L' or 'F'")

    def _inertia_now(self, m_prop, role):
        """
        Current J_B(m_prop) for the given role.
        J = (m_ring_dry + m_prop) · J_ring_unit + J_cyl
        Propellant occupies the same annular envelope as the dry ring, so the
        same J_ring_unit multiplies the combined ring mass.
        """
        J_now = np.zeros((3, 3))

        if role == 'L':
            m_ring_dry, J_cyl = self.param.m_ring_dry_L, self.param.J_cylinder_L
        elif role == 'F':
            m_ring_dry, J_cyl = self.param.m_ring_dry_F, self.param.J_cylinder_F
        else:
            raise ValueError(f"Invalid role {role!r}: expected 'L' or 'F'")

        m_ring = m_ring_dry + max(m_prop, 0.0)
        J_now = m_ring * J_ring_unit(self.param.r_in, self.param.r_out, self.param.h_ring) + J_cyl

        return J_now


    def _inertia_dot_now(self, m_dot):
        """
        dJ/dt. Cylinder mass and dry-ring mass are constant ⇒ only the
        propellant contributes. J_ring is linear in (m_ring_dry + m_prop),
        so dJ/dt = ṁ_prop · K_ring. Geometry (r_in, r_out, h_ring) is
        shared between leader and follower, so no role dispatch is needed.
        """
        J_dot_now = np.zeros((3, 3))
        J_dot_now = m_dot * J_ring_unit(self.param.r_in, self.param.r_out, self.param.h_ring)

        return J_dot_now

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
