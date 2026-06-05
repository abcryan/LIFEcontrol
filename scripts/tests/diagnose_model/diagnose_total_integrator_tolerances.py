"""
diagnose_tolerance.py
=====================
Tolerance-convergence diagnostic for the LIFE truth model.

Re-runs the same simulation at multiple integrator tolerances (rtol = atol),
treats the tightest-tolerance run as the reference, and plots the per-component
truncation error of every state variable for the leader and the first follower
against time.

State variables tracked (per spacecraft):
    * position / δr            [m]      (||Δr|| of error vector)
    * velocity / δv            [m/s]
    * attitude (angle error)   [arcsec] (rotation-invariant relative angle)
    * angular rate             [rad/s]
    * propellant mass          [g]

Configuration is in the constants below. Run:

    python diagnose_tolerance.py

Outputs `tolerance_convergence.png` and a printed final-time error table.
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Project imports (mirror __main__.py)
from life_control.config.config import (ABCORR, BODIES, FRAME, KERNELS,
                                        OBSERVER)
from life_control.init.initialize_state import initialize_state
from life_control.plant_model.plant import Plant
from life_control.plant_model.spacecraft import Parameters, PhysicsFlags
from life_control.plant_model.thrusters import N_THRUSTERS
from life_control.spice.environment import SpiceEnv

# ─────────────────────────────────────────────────────────────────────
# Configuration — edit here
# ─────────────────────────────────────────────────────────────────────
EPOCH        = "2026-05-12T00:00:00"
N_SC         = 5
DIM_X_SC     = 14
DIM_U_SC     = N_THRUSTERS              # 20
DT           = 100.0                    # outer step [s]
T_TOTAL      = 6.0 * 3600.0             # 6 hours; raise to 0.5*86400 to match __main__.py

# Tolerance sweep (rtol = atol = tol). Tightest is the reference.
TOLS         = [1e-6, 1e-8, 1e-10, 1e-12, 1e-13]

# Tiny IC perturbation so attitude dynamics actually evolve
W0_PERTURB   = np.array([1e-5, 1e-5, 1e-5])    # rad/s on every spacecraft

# Constant fixed thrust so all 14 state components evolve (force + torque + mass)
THRUSTER_IDX = 1                        # +y side thruster of group +x
THRUST_N     = 1e-4                     # 0.1 mN per active thruster

OUT_DIR      = Path(".")
SHOW_PLOTS   = True

ARCSEC_PER_RAD = 180.0 / np.pi * 3600.0


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────
def build_initial_state(param):
    """Same ICs as __main__.py, but with a small ω perturbation per s/c."""
    r_L = np.array([-9.594503991242750e7, -1.098032827423822e8, -4.778858640538428e7])
    v_L = np.array([ 2.279303677102495e1, -1.735582913273474e1, -7.640659579588683e0])
    q_L = np.array([1.0, 0.0, 0.0, 0.0])
    w_L = np.zeros(3)
    x_L = np.concatenate([r_L, v_L, q_L, w_L, [param.m_prop_init_L]])
    x = initialize_state("square planar", 0.1, "same as leader",
                         param.m_prop_init_F, x_L, N_SC)
    for i in range(N_SC):
        x[i * DIM_X_SC + 10:i * DIM_X_SC + 13] = W0_PERTURB
    return x


def build_control():
    """Constant thrust on one thruster per spacecraft."""
    u = np.zeros(N_SC * DIM_U_SC)
    for i in range(N_SC):
        u[i * DIM_U_SC + THRUSTER_IDX] = THRUST_N
    return u


def run_sim(plant, x0, et0, dt, n_steps, tol, u):
    x = x0.copy()
    et = et0
    X = np.zeros((n_steps + 1, x0.size))
    X[0] = x
    for k in range(n_steps):
        x = plant.step(x, u, et, dt, rtol=tol, atol=tol)
        et += dt
        X[k + 1] = x
    return X


# ─────────────────────────────────────────────────────────────────────
# Error metrics
# ─────────────────────────────────────────────────────────────────────
def quat_angle_array(Q_ref, Q_test):
    """
    Vectorised relative-rotation angle [rad] between two unit-quaternion
    timeseries (each (N,4), scalar-first). Uses atan2(||vec||, |scalar|) of
    q_ref^-1 ⊗ q_test, which is numerically robust everywhere — including
    near identity, where the arccos(|scalar|) form rounds to exactly 0
    once the difference drops below ~machine epsilon.
    """
    # q_ref^-1 (conjugate) = [q0, -q1, -q2, -q3]
    p0, p1, p2, p3 = Q_ref[:, 0], -Q_ref[:, 1], -Q_ref[:, 2], -Q_ref[:, 3]
    q0, q1, q2, q3 = Q_test[:, 0], Q_test[:, 1], Q_test[:, 2], Q_test[:, 3]
    # q_err = q_ref^-1 ⊗ q_test  (Hamilton product, scalar-first)
    w = p0*q0 - p1*q1 - p2*q2 - p3*q3
    x = p0*q1 + p1*q0 + p2*q3 - p3*q2
    y = p0*q2 - p1*q3 + p2*q0 + p3*q1
    z = p0*q3 + p1*q2 - p2*q1 + p3*q0
    vec_norm = np.sqrt(x*x + y*y + z*z)
    return 2.0 * np.arctan2(vec_norm, np.abs(w))


def sc_errors(X_test, X_ref, sc_idx):
    """Five error timeseries for spacecraft `sc_idx` (leader=0, followers ≥1)."""
    base = sc_idx * DIM_X_SC
    dr = X_test[:, base    :base + 3] - X_ref[:, base    :base + 3]
    dv = X_test[:, base + 3:base + 6] - X_ref[:, base + 3:base + 6]
    dw = X_test[:, base +10:base +13] - X_ref[:, base +10:base +13]
    pos_err   = np.linalg.norm(dr, axis=1) * 1e3                          # km  -> m
    vel_err   = np.linalg.norm(dv, axis=1) * 1e3                          # km/s-> m/s
    omega_err = np.linalg.norm(dw, axis=1)                                # rad/s
    quat_err  = quat_angle_array(X_ref [:, base+6:base+10],
                                 X_test[:, base+6:base+10]) * ARCSEC_PER_RAD
    mass_err  = np.abs(X_test[:, base + 13] - X_ref[:, base + 13]) * 1e3  # kg  -> g
    return pos_err, vel_err, quat_err, omega_err, mass_err


# ─────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────
ROW_INFO = [
    ("position",     "[m]",      "Position error"),
    ("velocity",     "[m/s]",    "Velocity error"),
    ("attitude",     "[arcsec]", "Attitude angle error"),
    ("angular_rate", "[rad/s]",  "Angular-rate error"),
    ("propellant",   "[g]",      "Propellant-mass error"),
]


def plot_errors(t_hist, errs_by_tol, ref_tol, save_path):
    tols_plot = sorted([t for t in errs_by_tol if t != ref_tol], reverse=True)
    n_rows = len(ROW_INFO)
    fig, axes = plt.subplots(n_rows, 2, figsize=(13, 14), sharex=True)

    cmap = plt.cm.viridis
    colors = [cmap(i / max(len(tols_plot) - 1, 1)) for i in range(len(tols_plot))]

    sc_pairs = [(0, "Leader (absolute state)"),
                (1, "Follower 1 (δ-state)")]

    for col, (sc_idx, sc_label) in enumerate(sc_pairs):
        for row, (name, unit, title) in enumerate(ROW_INFO):
            ax = axes[row, col]
            for tol, color in zip(tols_plot, colors):
                err = errs_by_tol[tol][sc_idx][row]
                err_plot = np.where(err > 0, err, np.nan)
                ax.semilogy(t_hist / 3600.0, err_plot,
                            color=color, label=f"tol={tol:.0e}", lw=1.3)
            if row == 0:
                ax.set_title(sc_label, fontsize=11)
            if col == 0:
                ax.set_ylabel(f"{title}\n{unit}", fontsize=10)
            if row == n_rows - 1:
                ax.set_xlabel("Time [h]")
            ax.grid(True, which="both", alpha=0.3)
            if row == 0 and col == 1:
                ax.legend(loc="best", fontsize=8,
                          title=f"vs ref={ref_tol:.0e}")

    fig.suptitle("Tolerance-convergence diagnostic — error vs tightest-tol reference",
                 fontsize=13, y=0.998)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    return fig


def convergence_summary(errs_by_tol, ref_tol):
    """Print final-time error vs tolerance for the leader and follower 1."""
    print("\n" + "=" * 90)
    print(f"  Final-time error vs reference (ref tolerance: {ref_tol:.0e})")
    print("=" * 90)
    for sc_idx, label in [(0, "Leader (absolute)"), (1, "Follower 1 (δ-state)")]:
        print(f"\n  --- {label} ---")
        header = f"  {'tol':>10}  " + "  ".join(f"{r[0]:>13}" for r in ROW_INFO)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for tol in sorted(errs_by_tol, reverse=True):
            if tol == ref_tol:
                continue
            finals = [errs_by_tol[tol][sc_idx][r][-1] for r in range(len(ROW_INFO))]
            row = f"  {tol:>10.0e}  " + "  ".join(f"{v:>13.3e}" for v in finals)
            print(row)


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 78)
    print("  LIFE Truth-Model — Tolerance-Convergence Diagnostic")
    print("=" * 78)

    env   = SpiceEnv(KERNELS, BODIES, FRAME, ABCORR, OBSERVER)
    param = Parameters()
    flags = PhysicsFlags()
    plant = Plant(env, param, flags, N_SC, DIM_X_SC, DIM_U_SC)

    x0   = build_initial_state(param)
    u    = build_control()
    et0  = env.str2et(EPOCH)
    n_steps = int(T_TOTAL / DT)
    t_hist  = np.arange(n_steps + 1) * DT

    print(f"  Epoch     : {EPOCH}")
    print(f"  N_SC      : {N_SC}")
    print(f"  T_total   : {T_TOTAL/3600:.2f} h    dt: {DT:.0f} s    steps: {n_steps}")
    print(f"  Thrust    : thruster #{THRUSTER_IDX} at {THRUST_N} N per s/c")
    print(f"  Tolerances: {TOLS}    reference: {min(TOLS):.0e}")
    print(f"  Physics   : {flags.summary()}\n")

    # ── Tolerance sweep ────────────────────────────────────────────
    results = {}
    for tol in sorted(TOLS, reverse=True):
        t0 = time.perf_counter()
        X  = run_sim(plant, x0, et0, DT, n_steps, tol, u)
        wall = time.perf_counter() - t0
        results[tol] = X
        print(f"  tol={tol:.0e}  wall={wall:7.2f} s   "
              f"|q_L final|={np.linalg.norm(X[-1, 6:10]):.15f}")

    ref_tol = min(TOLS)
    X_ref = results[ref_tol]

    # ── Errors for leader (sc 0) and first follower (sc 1) ─────────
    errs_by_tol = {
        tol: {sc_idx: sc_errors(X, X_ref, sc_idx) for sc_idx in (0, 1)}
        for tol, X in results.items()
    }

    save_path = OUT_DIR / "tolerance_convergence.png"
    plot_errors(t_hist, errs_by_tol, ref_tol, save_path)
    print(f"\n  Plot saved: {save_path.resolve()}")

    convergence_summary(errs_by_tol, ref_tol)

    if SHOW_PLOTS:
        plt.show()


if __name__ == "__main__":
    main()