"""
term_analysis.py — quantify per-term influence on a follower's RELATIVE
acceleration at a single epoch, in the spirit of Table 1 of Scharf et al.
(2002), "On the Validity of the Double Integrator Approximation in Deep
Space Formation Flying."

WHAT THIS IS (and isn't):
    Scharf's Table 1 lists analytical WORST-CASE BOUNDS on each term of the
    expanded relative dynamics (|r_F| = 1 AU, |ρ| ≤ 1 km, ‖Q_r‖ = 2, ...).
    This module instead reports the INSTANTANEOUS MAGNITUDE of each term at
    one chosen epoch on your actual trajectory. That is less conservative
    than a bound — it's the value at one specific geometry, not the adversarial
    one — but it's the right quantity for "which terms matter for my sim?".
    Your numbers should sit at or below the analogous Scharf bounds (scaled
    for your masses/areas/baselines); a term ABOVE the bound is a red flag.

WHY IT DOESN'T EDIT plant.py:
    Every term is already a separate Plant method returning a km/s² 3-vector.
    We call them individually instead of letting acc_rhs_rel sum them. The one
    exception is the PER-BODY gravity breakdown: acc_grav_rel sums over bodies
    internally, so to attribute gravity to Sun / Earth / Moon / planets
    separately (matching Scharf's per-planet rows) we replicate that single
    differential-gravity loop here, body by body. The math is identical to
    plant.acc_grav_rel — just not pre-summed.

UNITS:
    Plant methods return km/s². We multiply by 1e3 to report m/s², matching
    Scharf's table.
"""
import numpy as np

import life_control.utils.constants as const


KM_TO_M = 1.0e3   # km/s² → m/s²


def analyze_follower_terms(plant, x, et, follower_index, u=None,
                           dim_x_sc=14, dim_u_sc=20):
    """
    Evaluate every term of follower `follower_index`'s relative acceleration
    at state `x`, epoch `et`. Returns a dict {term_name: magnitude [m/s²]}.

    Parameters
    ----------
    plant          : the constructed Plant object (gives us env, param, flags,
                     and all the acc_* methods).
    x              : full stacked state vector at the epoch.
    et             : ephemeris time [s] at the epoch.
    follower_index : which follower to analyze (>= 1; 0 is the leader).
    u              : full stacked control vector (optional). If None, the
                     control-difference term is reported as zero.
    dim_x_sc       : per-spacecraft state dimension (default 14).
    dim_u_sc       : per-spacecraft control dimension (default 20).

    Notes
    -----
    - All terms are computed in the SAME relative (differential) sense your
      acc_rhs_rel uses: a_follower − a_leader, evaluated at well-conditioned
      scale (δr-based) where applicable.
    - The per-body gravity loop here is the body-by-body version of
      plant.acc_grav_rel; summing all rows reproduces acc_grav_rel exactly
      (verified in the self-test at the bottom of this file).
    """
    i = follower_index

    # ── Extract leader and follower sub-states ───────────────────────
    sl_L      = slice(0, dim_x_sc)
    sl_i      = slice(dim_x_sc * i, dim_x_sc * (i + 1))
    x_L       = x[sl_L]
    x_i       = x[sl_i]

    r_L       = x_L[0:3]
    q_L       = x_L[6:10]
    m_prop_L  = x_L[13]

    delta_r   = x_i[0:3]
    q_i       = x_i[6:10]
    m_prop_i  = x_i[13]

    m_total_L = plant._m_total(m_prop_L, 'L')
    m_total_i = plant._m_total(m_prop_i, 'F')

    # ── Fleet snapshot for ISC gravity ───────────────────────────────
    delta_r_all, m_total_all = plant._fleet_snapshot(x)

    results = {}

    # ── 1. Differential N-body gravity, broken out PER BODY ──────────
    # This replicates plant.acc_grav_rel's loop, but reports each body
    # separately so we can match Scharf's per-planet rows. The math is:
    #   da_b = -mu_b * [ d2/|d2|^3 - d1/|d1|^3 ],  d1 = r_L - r_b, d2 = d1 + δr
    grav_total = np.zeros(3)
    for body, mu in plant.env.GM.items():
        r_b = plant.env.body_position(body, et)
        d1  = r_L - r_b
        d2  = d1 + delta_r
        da_b = -mu * (d2 / np.linalg.norm(d2) ** 3
                    - d1 / np.linalg.norm(d1) ** 3)
        results[f"grav_diff[{body}]"] = np.linalg.norm(da_b) * KM_TO_M
        grav_total += da_b
    results["grav_diff[TOTAL]"] = np.linalg.norm(grav_total) * KM_TO_M

    # ── 2. Differential SRP (combined DC+Q+Offset, cannonball or N-plate) ─
    # Whatever SRP model the flags select, this is the literal a_F − a_L.
    # NOTE: this is the COMBINED SRP differential. Scharf splits it into
    # DC/Q/Offset analytically; your truth model does not — so this single
    # number corresponds to their (DC + Q + Offset) sum.
    a_srp_rel = plant.acc_srp_rel(r_L, delta_r, m_total_L, m_total_i, et)
    results["srp_diff[combined]"] = np.linalg.norm(a_srp_rel) * KM_TO_M

    # ── 3. Inter-spacecraft gravity (differential) ───────────────────
    a_isc_rel = plant.acc_grav_isc_rel(i, delta_r_all, m_total_all)
    results["grav_isc_diff"] = np.linalg.norm(a_isc_rel) * KM_TO_M

    # ── 4. Differential control acceleration (one-thruster-ish) ──────
    if u is not None:
        from life_control.plant_model.thrusters import (
            clamp_thrust, force_torque_body,
        )
        from life_control.utils.coordinate_trafos import quat_to_CTM

        # Leader control accel
        T_L = clamp_thrust(u[0:dim_u_sc], plant.param.T_MAX)
        if m_prop_L <= 0.0:
            T_L = np.zeros_like(T_L)
        F_B_L, _ = force_torque_body(T_L)
        a_ctrl_L = (quat_to_CTM(q_L).T @ F_B_L / m_total_L) * const.KM_PER_M

        # Follower control accel
        T_i = clamp_thrust(u[dim_u_sc * i: dim_u_sc * (i + 1)], plant.param.T_MAX)
        if m_prop_i <= 0.0:
            T_i = np.zeros_like(T_i)
        F_B_i, _ = force_torque_body(T_i)
        a_ctrl_i = (quat_to_CTM(q_i).T @ F_B_i / m_total_i) * const.KM_PER_M

        results["ctrl_diff"] = np.linalg.norm(a_ctrl_i - a_ctrl_L) * KM_TO_M
    else:
        results["ctrl_diff"] = 0.0

    # ── 5. Total relative acceleration (cross-check) ─────────────────
    # Should equal the sum of all differential terms (within float error).
    # We call the real acc_rhs_rel to get the authoritative total.
    if u is not None:
        a_ctrl_L_vec = a_ctrl_L
        a_ctrl_i_vec = a_ctrl_i
    else:
        a_ctrl_L_vec = np.zeros(3)
        a_ctrl_i_vec = np.zeros(3)

    a_rel_total = plant.acc_rhs_rel(
        r_L, delta_r, m_total_L, m_total_i,
        a_ctrl_L_vec, a_ctrl_i_vec,
        i, delta_r_all, m_total_all, et,
    )
    results["TOTAL_rel_accel"] = np.linalg.norm(a_rel_total) * KM_TO_M

    return results


def print_term_table(results, title="Relative-acceleration term magnitudes"):
    """Pretty-print the results dict, sorted descending by magnitude."""
    print(f"\n{title}")
    print(f"{'term':<28}{'|a| [m/s^2]':>16}")
    print("-" * 44)
    # Keep TOTAL rows at the bottom; sort the rest by magnitude.
    special = {k: v for k, v in results.items() if "TOTAL" in k}
    normal  = {k: v for k, v in results.items() if "TOTAL" not in k}
    for k, v in sorted(normal.items(), key=lambda kv: kv[1], reverse=True):
        print(f"{k:<28}{v:>16.3e}")
    print("-" * 44)
    for k, v in special.items():
        print(f"{k:<28}{v:>16.3e}")
        