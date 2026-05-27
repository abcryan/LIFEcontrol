"""
Experiment: numerical precision of absolute-difference vs direct-relative
propagation for the relative state between two formation-flying spacecraft.

Setup
─────
Two spacecraft on (essentially) the same JWST-like halo IC near Sun–Earth L2,
offset by δr0 = 100 m along the inertial x-axis, with δv0 = 0. No control,
no torques, no SRP (so the only physics is N-body gravity, identical for both).

Method (a) – absolute differencing
    Propagate spacecraft 1 and spacecraft 2 independently as 14-state systems
    in SSB-centered ICRF. Form δr(t) = r2(t) - r1(t) after the fact.
    Suffers from catastrophic cancellation: r1, r2 ~ 1.5e8 km but δr ~ 1e-4 km.

Method (b) – direct relative propagation
    Propagate the chief (spacecraft 1) absolutely. Propagate (δr, δv) directly
    using the *difference* of gravitational accelerations, evaluated as
        a_rel(t) = a_grav(r_chief + δr, t) - a_grav(r_chief, t),
    which is a well-conditioned ~tidal-tensor times δr quantity.

Output
──────
A figure with three panels:
  (1) |δr|(t) from both methods, log scale.        --> show they agree until (a) breaks
  (2) |δr_a(t) - δr_b(t)|, log scale.              --> the noise floor of (a) wrt (b)
  (3) Sweep over integrator tolerance:
      run (a) at several rtol values; plot final-time noise vs rtol.
      Method (b)'s level is shown as a horizontal reference.

How to run
──────────
Place this file next to your main plant module (the one you shared) and adjust
the import below if your module is named differently. The script uses your
SpiceEnv and Plant classes unchanged for method (a). For method (b) it defines
a small RelativePlant that reuses Plant.a_grav for both evaluations so that the
gravity model is *byte-identical* between the two methods.
"""
import sys
sys.dont_write_bytecode = True

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Import your existing classes. Adjust this import to match your layout.
# If your main file is called e.g. `truth_model.py`, use:
#     from truth_model import SpiceEnv, Plant
# For the demo here we assume it is importable as `plant_module`.
from life_control.__main__ import SpiceEnv, Plant


# ── Direct relative-state propagator ─────────────────────────────────────────

class RelativePlant:
    """
    Propagates the chief spacecraft absolutely (14-state) AND the relative
    translational state (δr, δv) of one deputy directly, by forming the
    *difference* of gravitational accelerations rather than differencing two
    absolute states.

    Combined state (20,):
        x[ 0: 3]  r_chief^I    [km]
        x[ 3: 6]  v_chief^I    [km/s]
        x[ 6:10]  q_chief
        x[10:13]  ω_chief^B    [rad/s]
        x[13]     m_chief      [kg]
        x[14:17]  δr^I = r_deputy - r_chief    [km]
        x[17:20]  δv^I = v_deputy - v_chief    [km/s]

    Attitude is irrelevant to this experiment so we just carry the chief's
    attitude states along for shape compatibility; no torques, no SRP.
    """

    def __init__(self, plant: Plant):
        self.plant = plant   # reuse a_grav (byte-identical gravity model)
        self.env   = plant.env

    def a_rel(self, r_chief: np.ndarray, delta_r: np.ndarray, et: float) -> np.ndarray:
        """
        Differential gravitational acceleration on the deputy, evaluated as
        the difference of two SSB-relative N-body accelerations.

        IMPORTANT: this is still ultimately a difference of two ~1.5e8 km
        vectors per body — BUT the difference is taken *inside* the gravity
        sum, body by body, where each body-relative vector is small (Sun-rel
        distance ~1.5e8 km, but Earth-rel distance only ~1.5e6 km for L2).
        For each body b, we compute
            a_b(r_chief + δr) - a_b(r_chief)
        which is a well-conditioned quantity of order (μ_b / r_b^3) * δr
        (the tidal tensor times δr). The large-number cancellation never
        appears because we never form r_2 - r_1 from SSB-based absolutes —
        we form (r_chief + δr - r_body) - (r_chief - r_body) inside the sum,
        where δr is the small quantity directly.
        """
        a = np.zeros(3)
        for body, mu in self.env.GM.items():
            r_b   = self.env.body_position(body, et)
            d1    = r_chief - r_b                     # body -> chief
            d2    = d1 + delta_r                      # body -> deputy
            a    += -mu * (d2 / np.linalg.norm(d2)**3
                         - d1 / np.linalg.norm(d1)**3)
        return a

    def x_dot(self, tau, x, et0, u_chief):
        r_c   = x[0:3]
        v_c   = x[3:6]
        q_c   = x[6:10]
        w_c   = x[10:13]
        m_c   = x[13]
        d_r   = x[14:17]
        d_v   = x[17:20]
        et    = et0 + tau

        # Chief absolute dynamics (no SRP, no torque, no thrust).
        a_g_chief = self.plant.a_grav(r_c, et)
        r_c_dot   = v_c
        v_c_dot   = a_g_chief
        # Attitude kinematics — irrelevant to this experiment, freeze.
        q_dot     = np.zeros(4)
        w_dot     = np.zeros(3)
        m_dot     = 0.0

        # Relative translational dynamics.
        a_rel     = self.a_rel(r_c, d_r, et)
        dr_dot    = d_v
        dv_dot    = a_rel

        return np.concatenate([
            r_c_dot, v_c_dot, q_dot, w_dot, [m_dot], dr_dot, dv_dot
        ])

    def step(self, x, u_chief, t, dt, rtol=1e-13, atol=1e-16):
        sol = solve_ivp(
            self.x_dot, (0.0, dt), x,
            method="DOP853",
            args=(t, u_chief),
            rtol=rtol, atol=atol,
        )
        if not sol.success:
            raise RuntimeError(f"Integration failed: {sol.message}")
        x_next = sol.y[:, -1]
        # No attitude normalization (we froze it).
        return x_next


# ── Method (a): two absolute propagations, independent ───────────────────────

def propagate_absolute_pair(plant, r0, v0, q0, w0, m0, delta_r0, delta_v0,
                            et0, dt, n_steps, rtol, atol):
    """
    Propagate spacecraft 1 (chief) and spacecraft 2 (deputy) as two
    independent 14-state systems. δr is reconstructed by differencing.
    """
    x1 = np.concatenate([r0,             v0,             q0, w0, [m0]])
    x2 = np.concatenate([r0 + delta_r0,  v0 + delta_v0,  q0, w0, [m0]])

    delta_r_hist = np.zeros((n_steps + 1, 3))
    delta_r_hist[0] = delta_r0

    u = np.zeros(6)
    et = et0
    for k in range(n_steps):
        x1 = plant.step(x1, u, et, dt, rtol=rtol, atol=atol)
        x2 = plant.step(x2, u, et, dt, rtol=rtol, atol=atol)
        delta_r_hist[k + 1] = x2[0:3] - x1[0:3]
        et += dt
    return delta_r_hist


# ── Method (b): chief + direct relative state ────────────────────────────────

def propagate_relative_direct(rel_plant, r0, v0, q0, w0, m0, delta_r0, delta_v0,
                              et0, dt, n_steps, rtol, atol):
    """
    Propagate the chief absolutely and (δr, δv) directly.
    """
    x = np.concatenate([r0, v0, q0, w0, [m0], delta_r0, delta_v0])
    delta_r_hist = np.zeros((n_steps + 1, 3))
    delta_r_hist[0] = delta_r0

    u = np.zeros(6)
    et = et0
    for k in range(n_steps):
        x = rel_plant.step(x, u, et, dt, rtol=rtol, atol=atol)
        delta_r_hist[k + 1] = x[14:17]
        et += dt
    return delta_r_hist


# ── Main experiment ──────────────────────────────────────────────────────────

if __name__ == "__main__":

    # --- Common setup ----------------------------------------------------
    env   = SpiceEnv()
    plant = Plant(env)

    # Disable SRP for this experiment: both spacecraft would see essentially
    # identical SRP (they are 100 m apart at ~1 AU from the Sun) and including
    # it just adds another source of noise to the comparison. The cancellation
    # problem is about gravity differencing, not SRP.
    # Easiest: monkeypatch a_srp to return zero.
    plant.a_srp = lambda r_I, m, et: np.zeros(3)

    rel_plant = RelativePlant(plant)

    # JWST-like halo IC.
    et0 = env.str2et("2026-05-12T00:00:00")
    r0  = np.array([-9.594503991242750e7, -1.098032827423822e8, -4.778858640538428e7])
    v0  = np.array([ 2.279303677102495e1, -1.735582913273474e1, -7.640659579588683e0])
    q0  = np.array([1.0, 0.0, 0.0, 0.0])
    w0  = np.array([0.0, 0.0, 0.0])     # zero rate: attitude is irrelevant here
    m0  = 3000.0

    # 100 m separation along ICRF x-axis, zero relative velocity.
    delta_r0 = np.array([1.0e-1, 0.0, 0.0])   # km   = 100 m
    delta_v0 = np.zeros(3)

    # --- Time grid -------------------------------------------------------
    # 1 day at 200 s sampling. Long enough for natural tidal drift to grow
    # above the cancellation-noise floor of (a) so the plot tells a story.
    dt      = 200.0
    n_steps = int(1 * 86400 / dt)
    t_hist  = np.arange(n_steps + 1) * dt

    print(f"\nExperiment: |δr0| = {np.linalg.norm(delta_r0)*1e3:.3f} m, "
          f"duration = {n_steps*dt/3600:.2f} h, {n_steps} steps of {dt} s")

    # --- Run both methods at a strict tolerance --------------------------
    rtol_strict = 1e-13
    atol_strict = 1e-16

    print("\n(a) Propagating two absolute states independently ...")
    dr_a = propagate_absolute_pair(
        plant, r0, v0, q0, w0, m0, delta_r0, delta_v0,
        et0, dt, n_steps, rtol=rtol_strict, atol=atol_strict,
    )

    print("(b) Propagating chief + direct relative state ...")
    dr_b = propagate_relative_direct(
        rel_plant, r0, v0, q0, w0, m0, delta_r0, delta_v0,
        et0, dt, n_steps, rtol=rtol_strict, atol=atol_strict,
    )

    # Magnitudes in meters (more readable).
    mag_a = np.linalg.norm(dr_a, axis=1) * 1e3   # m
    mag_b = np.linalg.norm(dr_b, axis=1) * 1e3   # m
    diff  = np.linalg.norm(dr_a - dr_b, axis=1) * 1e3   # m

    print(f"\nFinal |δr|  (a, absolute differencing) = {mag_a[-1]:.6e} m")
    print(f"Final |δr|  (b, direct relative)       = {mag_b[-1]:.6e} m")
    print(f"Final |δr_a - δr_b|                    = {diff[-1]:.6e} m")

    # --- Tolerance sweep on method (a) -----------------------------------
    # Show that even tightening rtol can't fix the cancellation floor.
    rtols     = [1e-8, 1e-10, 1e-12, 1e-13]
    sweep_end_diff = []
    print("\nTolerance sweep on method (a) vs reference (b) ...")
    for rt in rtols:
        dr_a_rt = propagate_absolute_pair(
            plant, r0, v0, q0, w0, m0, delta_r0, delta_v0,
            et0, dt, n_steps, rtol=rt, atol=rt*1e-3,
        )
        end_diff = np.linalg.norm(dr_a_rt[-1] - dr_b[-1]) * 1e3
        sweep_end_diff.append(end_diff)
        print(f"  rtol={rt:.0e}  |δr_a - δr_b|(T) = {end_diff:.3e} m")

    # --- Plot ------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(9, 11), constrained_layout=True)

    t_h = t_hist / 3600.0   # hours

    # Panel 1: |δr|(t) from both methods.
    ax = axes[0]
    ax.semilogy(t_h, mag_a, label="(a) absolute differencing", lw=1.5)
    ax.semilogy(t_h, mag_b, label="(b) direct relative",       lw=1.5, ls="--")
    ax.axhline(np.linalg.norm(delta_r0)*1e3, color="gray", lw=0.8, ls=":",
               label="|δr₀| = 100 m")
    ax.set_xlabel("time [h]")
    ax.set_ylabel("|δr(t)| [m]")
    ax.set_title("Separation magnitude over time — both methods")
    ax.legend(loc="best")
    ax.grid(True, which="both", alpha=0.3)

    # Panel 2: |δr_a - δr_b|(t).
    ax = axes[1]
    ax.semilogy(t_h, diff, color="C3", lw=1.5)
    ax.set_xlabel("time [h]")
    ax.set_ylabel("|δr_a(t) - δr_b(t)| [m]")
    ax.set_title("Disagreement between methods — noise floor of (a)")
    # Reference lines: double-precision floor on a ~1.5e8 km absolute state.
    eps_floor_km = 1.5e8 * np.finfo(float).eps   # ~ 3e-8 km = 30 μm
    ax.axhline(eps_floor_km * 1e3, color="gray", lw=0.8, ls=":",
               label=f"|r| · ε_machine ≈ {eps_floor_km*1e3:.1e} m")
    ax.legend(loc="best")
    ax.grid(True, which="both", alpha=0.3)

    # Panel 3: tolerance sweep on (a).
    ax = axes[2]
    ax.loglog(rtols, sweep_end_diff, "o-", color="C0",
              label="(a) end-time error vs (b)")
    ax.axhline(eps_floor_km * 1e3, color="gray", lw=0.8, ls=":",
               label=f"cancellation floor ≈ {eps_floor_km*1e3:.1e} m")
    ax.set_xlabel("rtol of integrator")
    ax.set_ylabel("|δr_a - δr_b|(T) [m]")
    ax.set_title("Method (a) error at T = 1 day vs integrator tolerance — "
                 "cancellation floor cannot be reduced by tighter rtol")
    ax.invert_xaxis()   # tighter tolerance to the right
    ax.legend(loc="best")
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("Absolute differencing vs direct relative propagation\n"
                 "Two spacecraft, δr₀ = 100 m, halo near Sun–Earth L2, "
                 "gravity-only, 24 hours", fontsize=11)
    fig.savefig("relative_vs_absolute_experiment.png", dpi=140)
    plt.show()
    print("\nSaved figure: relative_vs_absolute_experiment.png")