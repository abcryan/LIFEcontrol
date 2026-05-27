"""
SRP-contribution diagnostic for the LIFE truth model.

Question this script answers:
    Over 24 hours, by how much does SRP change the relative-translation
    dynamics of the deputies — expressed as a percentage of the total
    relative-state drift?

Method:
    Two simulation runs are performed with IDENTICAL initial conditions,
    integrator settings, and dynamics — except that SRP is disabled in
    one of them. The difference between the two runs at the final time
    is the contribution of SRP. The ratio of that contribution to the
    total drift (final − initial) gives the percentage.

    Formally, for each deputy i:
        δr_B(t)   : run with SRP enabled  (gravity + SRP)
        δr_A(t)   : run with SRP disabled (gravity only)
        ΔSRP_i(T) = δr_B(T) − δr_A(T)         # SRP contribution to position
        Δdrift_i(T) = δr_B(T) − δr_B(0)        # total drift with SRP
        SRP_fraction_i = |ΔSRP_i(T)| / |Δdrift_i(T)| × 100 %

Scenarios:
    The script runs the comparison under two parameter scenarios so the
    user can see SRP's contribution decomposed by source:

    Scenario 1 (IDENTICAL): all spacecraft have identical (m, A, C_R).
        Here differential SRP is driven ONLY by position-dependent SRP
        (the 1/r^2 falloff and slightly different sun-pointing unit
        vectors at chief vs deputy positions). This is tiny because
        |δr| ≪ |r_sun|.

    Scenario 2 (PARAMETER MISMATCH): deputies have 10 % smaller area
        than the chief. Here differential SRP is dominated by the
        parameter mismatch. This is typically the larger contribution
        in practice (real spacecraft are never exactly identical).

Multiple baselines (10 m, 100 m, 1 km) are tested in each scenario
to show how SRP scales with formation separation.

How to run:
    Place next to your main.py and adjust the import line if needed.
    The script does not produce plots.
"""
import sys
sys.dont_write_bytecode = True

import numpy as np

# Import truth-model classes. Adjust this import to match your filename.
# If your main module is `main.py`, this will work:
from life_control.__main__ import (
    SpiceEnv,
    SpacecraftParams,
    Plant,
    NX_PER_SC,
)


# ── Helper: build a 5-spacecraft initial state vector ────────────────────────

def build_initial_state(baseline_km: float) -> np.ndarray:
    """
    Build a stacked initial state vector for 1 chief + 4 deputies at a
    given baseline. Chief at JWST-like halo IC. Deputies in a planar X
    formation around the chief.
    """
    # Chief.
    r0_chief = np.array([-9.594503991242750e7, -1.098032827423822e8, -4.778858640538428e7])
    v0_chief = np.array([ 2.279303677102495e1, -1.735582913273474e1, -7.640659579588683e0])
    q0       = np.array([1.0, 0.0, 0.0, 0.0])
    w0       = np.array([0.001, 0.01, 0.0])
    m0       = 3000.0

    x0_chief = np.concatenate([r0_chief, v0_chief, q0, w0, [m0]])

    # Deputies at ±x, ±y in chief-centered ICRF.
    delta_r0_list = [
        np.array([ baseline_km,  0.0,         0.0]),
        np.array([-baseline_km,  0.0,         0.0]),
        np.array([ 0.0,          baseline_km, 0.0]),
        np.array([ 0.0,         -baseline_km, 0.0]),
    ]

    pieces = [x0_chief]
    for dr0 in delta_r0_list:
        pieces.append(np.concatenate([dr0, np.zeros(3), q0, w0, [m0]]))
    return np.concatenate(pieces)


# ── Helper: propagate a 24-hour run with given plant ─────────────────────────

def propagate_24h(plant: Plant, x0: np.ndarray, et0: float,
                  dt: float = 1.0) -> np.ndarray:
    """
    Propagate `x0` for 24 hours under the given Plant. Returns final state.
    Zero control throughout.
    """
    n_steps = int(60 / dt)
    u       = np.zeros(plant.nu)
    x       = x0.copy()
    et      = et0
    for _ in range(n_steps):
        x   = plant.step(x, u, et, dt)
        et += dt
    return x


# ── Helper: build a Plant with optional SRP disabled ─────────────────────────

def build_plant(env: SpiceEnv, spacecraft_params: list,
                srp_enabled: bool) -> Plant:
    """
    Build a Plant with the given spacecraft parameters. If srp_enabled is
    False, monkey-patch a_srp and a_srp_relative to return zero so that
    SRP contributes nothing to either chief or deputy dynamics.
    """
    plant = Plant(env, spacecraft_params)
    if not srp_enabled:
        plant.a_srp          = lambda r_I, m, A, C_R_loc, et: np.zeros(3)
        plant.a_srp_relative = lambda r_chief, delta_r, m_chief, A_chief, CR_chief, \
                                       m_dep, A_dep, CR_dep, et: np.zeros(3)
    return plant


# ── Diagnostic core ──────────────────────────────────────────────────────────

def run_scenario(env: SpiceEnv, et0: float, scenario_name: str,
                 spacecraft_params: list, baselines_km: list) -> None:
    """
    For each baseline:
      - Run with SRP off, run with SRP on.
      - For each deputy, compute the SRP contribution and the total drift.
      - Print a per-deputy summary.
    """
    print(f"\n{'═' * 78}")
    print(f"Scenario: {scenario_name}")
    print(f"{'═' * 78}")
    print(f"Spacecraft parameters:")
    for sc in spacecraft_params:
        print(f"  {sc.label:<10s}  A = {sc.A_SRP:6.2f} m²   C_R = {sc.C_R:4.2f}")

    for baseline_km in baselines_km:
        baseline_m = baseline_km * 1e3
        print(f"\n── Baseline: {baseline_m:.1f} m ─────────────────────────"
              f"────────────────────────────────")

        x0 = build_initial_state(baseline_km)

        # Run A: SRP off (gravity only).
        plant_off = build_plant(env, spacecraft_params, srp_enabled=False)
        xA = propagate_24h(plant_off, x0, et0)

        # Run B: SRP on.
        plant_on  = build_plant(env, spacecraft_params, srp_enabled=True)
        xB = propagate_24h(plant_on, x0, et0)

        # Per-deputy comparison.
        # Columns: |Δdrift| with SRP, |Δdrift| no SRP, |ΔSRP|, fraction.
        header = (f"  {'deputy':<10s} "
                  f"{'|drift| (SRP off) [m]':>22s} "
                  f"{'|drift| (SRP on) [m]':>22s} "
                  f"{'|ΔSRP| [m]':>14s} "
                  f"{'SRP / drift [%]':>17s}")
        print(header)
        print("  " + "-" * (len(header) - 2))

        for i in range(1, 5):
            sl = slice(NX_PER_SC * i, NX_PER_SC * i + 3)
            dr0  = x0[sl]
            drA  = xA[sl]         # final δr, SRP off
            drB  = xB[sl]         # final δr, SRP on

            drift_A     = np.linalg.norm(drA - dr0) * 1e3      # m
            drift_B     = np.linalg.norm(drB - dr0) * 1e3      # m
            srp_contrib = np.linalg.norm(drB - drA) * 1e3      # m

            # Express the SRP contribution as a percentage of the total
            # (SRP-on) drift. Guard against division by zero in degenerate cases.
            if drift_B > 1e-15:
                pct = srp_contrib / drift_B * 100.0
            else:
                pct = float("nan")

            print(f"  {spacecraft_params[i].label:<10s} "
                  f"{drift_A:22.6e} "
                  f"{drift_B:22.6e} "
                  f"{srp_contrib:14.6e} "
                  f"{pct:16.4f} %")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    env = SpiceEnv()
    et0 = env.str2et("2026-05-12T00:00:00")

    baselines_km = [1.0e-2, 1.0e-1, 1.0]   # 10 m, 100 m, 1 km

    # ── Scenario 1: all spacecraft IDENTICAL ─────────────────────────────
    # Differential SRP is purely position-dependent (tiny because
    # |δr| ≪ |r_sun|).
    sc_identical = [
        SpacecraftParams(label="chief"),
        SpacecraftParams(label="deputy_1"),
        SpacecraftParams(label="deputy_2"),
        SpacecraftParams(label="deputy_3"),
        SpacecraftParams(label="deputy_4"),
    ]
    run_scenario(env, et0, "All spacecraft IDENTICAL "
                          "(differential SRP is position-only)",
                 sc_identical, baselines_km)

    # ── Scenario 2: deputies have 10% smaller area ───────────────────────
    # Differential SRP is dominated by area mismatch — realistic case where
    # spacecraft aren't perfectly identical.
    sc_mismatch = [
        SpacecraftParams(label="chief"),
        SpacecraftParams(label="deputy_1", A_SRP=36.0),   # -10%
        SpacecraftParams(label="deputy_2", A_SRP=36.0),
        SpacecraftParams(label="deputy_3", A_SRP=36.0),
        SpacecraftParams(label="deputy_4", A_SRP=36.0),
    ]
    run_scenario(env, et0, "Deputies have 10 % smaller area than chief "
                          "(realistic parameter mismatch)",
                 sc_mismatch, baselines_km)

    print()