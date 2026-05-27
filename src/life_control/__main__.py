"""
LIFE Mission — High-Fidelity Truth Model (Translational + Rotational Dynamics)

N-spacecraft formation: 1 LEADER + (N-1) FOLLOWERS, leader-plus-relative form.

Per spacecraft state layout (14 components):
    x[0:3]    r or δr        position           [km]    (leader: absolute,
                                                         follower: relative to leader)
    x[3:6]    v or δv        velocity           [km/s]  (same convention)
    x[6:10]   q_I^B          quaternion (inertial → body, absolute)
    x[10:13]  ω_IB^B         angular rate, body frame              [rad/s]
    x[13]     m_prop         propellant mass remaining             [kg]
                              (total mass = m_cyl + m_ring_dry + m_prop)

Per spacecraft control (20 components):
    u[0:20]   T_1 ... T_20   non-negative thrust magnitudes        [N]
                              for the 20 ring-mounted thrusters
                              (positions and directions defined in
                               config.spacecraft.thrusters, matching
                               Section 3.4 of the design document)

Spacecraft physical model (per role):
    Outer ring  (constant-mass structure + propellant tanks)  +
    inner solid cylinder (constant mass).  Both are concentric → total
    inertia adds directly with no parallel-axis term.  The propellant
    is housed inside the ring; the ring's *geometric envelope* (and
    therefore K_ring) is treated as constant, while its mass varies as
    propellant burns.  J̇ comes entirely from the changing ring mass.

Control allocation (in body frame, with T_l ≥ 0):
    F^B   = - B_F   @ T          [N]      (3 × 20)
    τ^B   = - B_TAU @ T          [N·m]    (3 × 20)
    ṁ_prop = - (Σ T_l) / (Isp · g0)         (each thruster contributes)
Force is transformed to inertial via the attitude quaternion before being
used in the translational dynamics → rotation-translation coupling.
"""

import numpy as np
from types import SimpleNamespace

# File Imports
from life_control.config.config import KERNELS, BODIES, FRAME, ABCORR, OBSERVER
from life_control.plant_model.thrusters import (
    THRUSTER_POSITIONS,
    THRUSTER_NORMALS,
    B_F,
    B_TAU,
    N_THRUSTERS,
    clamp_thrust,
    force_torque_body,
)

from life_control.plot.plotting import (plot_trajectory, plot_solar_system, plot_l2_rotating_frame_zoom)

##############################################
# Classes
##############################################

from life_control.spice.environment         import SpiceEnv
from life_control.gnc.navigation            import Sensor
from life_control.gnc.guidance              import Guidance
from life_control.gnc.control               import Controller
from life_control.plant_model.plant         import Plant
from life_control.plant_model.spacecraft    import Parameters


##############################################
# Methods
##############################################



# Initialize the stacked state vector for leader + followers.
def initialize_state(formation, baseline, att_F, m_prop_init_F, x_init_L, n_sc):

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
            x0.append(np.concatenate([dr0, dv0, q_init_F, w_init_F, [m_prop_init_F]]))
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
    dim_x_sc   = 14                  # [x,y,z, vx,vy,vz, q0..q3, wx,wy,wz, m_prop]
    dim_y_sc   = dim_x_sc            # same for now
    dim_u_sc   = N_THRUSTERS         # = 20: per-thruster thrust magnitudes [N]

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

    # Leader Initial State (r0, v0, q0, w0, m_prop0)  — Webb-like halo about L2
    r_init_L   = np.array([-9.594503991242750e7, -1.098032827423822e8, -4.778858640538428e7])   # [km]
    v_init_L   = np.array([ 2.279303677102495e1, -1.735582913273474e1, -7.640659579588683e0])   # [km/s]
    q_init_L   = np.array([1.0, 0.0, 0.0, 0.0])
    w_init_L   = np.array([0.0, 0.0, 0.0])
    x_init_L   = np.concatenate((r_init_L, v_init_L, q_init_L, w_init_L,
                                 [param.m_prop_init_L]))

    # Initial formation
    formation  = "square planar"
    baseline   = 0.1                       # 100 m
    att_F      = "same as leader"

    x_init     = initialize_state(formation, baseline, att_F,
                                  param.m_prop_init_F, x_init_L, n_sc)

    # ─── Initial-state printout ──────────────────────────────────────
    print(f"\nEpoch  : {env.et2utc(et_init)}")
    print(f"N_SC   : {n_sc} (1 leader + {n_sc - 1} followers)")
    print(f"Leader r0       : {x_init[0:3]}  km")
    print(f"Leader v0       : {x_init[3:6]}  km/s")
    print(f"Leader m_prop0  : {x_init[13]:.3f} kg  "
          f"(total mass {param.m_init_L:.1f} kg = {param.m_cylinder_L:.0f} cyl "
          f"+ {param.m_ring_dry_L:.0f} ring_dry + {param.m_prop_init_L:.0f} prop)")
    print(f"T_max / thruster: {param.T_MAX:.1f} N  "
          f"({N_THRUSTERS} thrusters → max body-frame |F| ≈ "
          f"{param.T_MAX * np.linalg.norm(B_F, ord=2):.1f} N if all parallel)")
    print("Baselines (initial, [m]):")
    for i in range(1, n_sc):
        dr_m = np.linalg.norm(x_init[dim_x_sc*i : dim_x_sc*i + 3]) * 1e3
        print(f"  follower_{i}: |δr0| = {dr_m:.3f} m")

    # ─── Time grid + history buffers ─────────────────────────────────
    dt          = 10.0
    t_tot       = 0.05 * 86400
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
        # Quick manual test: fire thruster #6 (-x face of -x cube) on the
        # leader at 1 N → continuous +x body-frame thrust of 1 N.  Uncomment
        # to exercise the new actuator path.
        u[0] = 1.0    # leader thruster index 6 (0-based: 5)
        # u[2] = 1.0
        # u[6] = 1.0
        # u[7] = 1.0

        # ------------   Act  ----------- #
        x_next  = plant.step(x, u, et, dt)

        # ------------  Data  ----------- #
        t_hist[k + 1]    = (k + 1) * dt
        X_hist[k + 1, :] = x_next

        if (k + 1) % print_every == 0 or k == n_steps - 1:
            dr1_m   = np.linalg.norm(x_next[dim_x_sc : dim_x_sc + 3]) * 1e3
            mprop_L = x_next[13]
            print(
                f"  k={k+1:5d}/{n_steps}  t={(k+1)*dt/3600:6.3f} h   "
                f"|r_L|={np.linalg.norm(x_next[0:3]):.6e} km   "
                f"|q_L|={np.linalg.norm(x_next[6:10]):.12f}   "
                f"|δr_1|={dr1_m:.6f} m   "
                f"m_prop_L={mprop_L:.4f} kg"
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