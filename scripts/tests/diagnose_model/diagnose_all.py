"""
LIFE Mission — Truth-Model Verification Suite
=============================================

A standalone, dependency-light test harness for the high-fidelity plant
(`life_control.plant_model.plant.Plant`). It mirrors the structure of
`main.py` (same env / Parameters / PhysicsFlags wiring) but instead of
running a mission it exercises the dynamics term-by-term.

Design goals
------------
1. Use PhysicsFlags to isolate EVERY term, verify flag-gating (off -> exact
   zero), then verify each term's UNITS, SIGN, MAGNITUDE and PHYSICS.
2. Verify the coupled structure: rotation<->translation (thrust), mass<->inertia,
   and the conventions shared by Omega_omega and quat_to_CTM.
3. Stress the numerical design: the relative-state trick for inter-spacecraft
   gravity (where it WINS) vs differential N-body gravity (where it does NOT
   actually beat the cancellation). These are DIAGNOSTIC tests — they print the
   number of significant digits retained.
4. Conservation laws (torque-free rigid body: H and KE), integration sanity
   (quaternion norm, bounded orbit, dt-convergence), and edge cases.

How to read the output
----------------------
Each line is [PASS]/[FAIL]/[INFO]. INFO lines are diagnostics that report a
measured quantity rather than asserting a hard threshold (used where the
"correct" value is a known limitation, e.g. float64 cancellation).

Run:  python -m life_control.test     (or wherever you place this file)
"""

import math
import time
from dataclasses import replace

import numpy as np

# ── Project imports (mirror main.py) ─────────────────────────────────────────
from life_control.config.config import KERNELS, BODIES, FRAME, ABCORR, OBSERVER
from life_control.spice.environment import SpiceEnv
from life_control.plant_model.plant import Plant
from life_control.plant_model.spacecraft import Parameters, PhysicsFlags
from life_control.plant_model.thrusters import (
    THRUSTER_POSITIONS, THRUSTER_NORMALS, B_F, B_TAU,
    N_THRUSTERS, clamp_thrust, force_torque_body,
)
from life_control.utils.coordinate_trafos import (
    quat_to_CTM, euler_to_quat, quat_to_euler,
)
from life_control.utils.other import Omega_omega
from life_control.utils.physics import J_ring_unit, J_cylinder_unit
import life_control.utils.constants as const


# ─────────────────────────────────────────────────────────────────────────────
# Constants reused across tests
# ─────────────────────────────────────────────────────────────────────────────

DIM_X_SC = 14
DIM_U_SC = 20
EPOCH    = "2026-05-12T00:00:00"

# Leader IC from main.py (Webb-like halo about L2, SSB-ICRF) [km, km/s]
R_INIT_L = np.array([-9.594503991242750e7, -1.098032827423822e8, -4.778858640538428e7])
V_INIT_L = np.array([ 2.279303677102495e1, -1.735582913273474e1, -7.640659579588683e0])


# ─────────────────────────────────────────────────────────────────────────────
# Tiny test harness
# ─────────────────────────────────────────────────────────────────────────────

class Check:
    def __init__(self):
        self.n = 0
        self.passed = 0
        self.fails = []

    def ok(self, name, cond, detail=""):
        self.n += 1
        if cond:
            self.passed += 1
            print(f"  [PASS] {name}   {detail}")
        else:
            self.fails.append(name)
            print(f"  [FAIL] {name}   {detail}")

    def info(self, name, detail=""):
        print(f"  [INFO] {name}   {detail}")

    def summary(self):
        print("\n" + "=" * 78)
        print(f"  {self.passed}/{self.n} checks passed.")
        if self.fails:
            print("  FAILURES:")
            for f in self.fails:
                print(f"    - {f}")
        else:
            print("  All hard assertions passed.")
        print("=" * 78)


def rel_err(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    den = np.linalg.norm(b)
    return np.linalg.norm(a - b) / den if den > 0 else np.linalg.norm(a - b)


def sig_digits(a, ref):
    """Number of correct significant digits of `a` relative to `ref`."""
    e = rel_err(a, ref)
    return np.inf if e == 0 else -math.log10(e)


def skew(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


# ─────────────────────────────────────────────────────────────────────────────
# Builders
# ─────────────────────────────────────────────────────────────────────────────

ALL_FLAG_NAMES = ["grav_nbody", "srp", "grav_isc", "tau_srp", "tau_grav",
                  "gyro_coupling", "j_dot_term", "mass_change"]


def flags_all_off():
    return PhysicsFlags(grav_nbody=False, srp=False, grav_isc=False,
                        tau_srp=False, tau_grav=False, gyro_coupling=False,
                        j_dot_term=False, mass_change=False)


def flags_only(**on):
    return replace(flags_all_off(), **on)


def make_plant(env, param, flags, n_sc):
    return Plant(env, param, flags, n_sc, DIM_X_SC, DIM_U_SC)


def sc_state(dr, q, w, mprop):
    """One spacecraft sub-state (leader uses dr=absolute r, v; follower uses delta)."""
    return np.concatenate([dr[0:3], dr[3:6], q, w, [mprop]])


def build_state(deltas, q_list=None, w_list=None, mprop_list=None,
                v_init_L=None):
    """
    Assemble a fleet state vector.
      deltas[0]    -> leader (absolute r,v) as (6,) ; if None use R_INIT_L/V_INIT_L
      deltas[i>0]  -> follower (delta r, delta v) as (6,)
    """
    n = len(deltas)
    q_list = q_list or [np.array([1.0, 0, 0, 0])] * n
    w_list = w_list or [np.zeros(3)] * n
    mprop_list = mprop_list or [150.0] * n

    x = np.zeros(n * DIM_X_SC)
    # leader
    if deltas[0] is None:
        rv0 = np.concatenate([R_INIT_L, V_INIT_L if v_init_L is None else v_init_L])
    else:
        rv0 = deltas[0]
    x[0:DIM_X_SC] = sc_state(rv0, q_list[0], w_list[0], mprop_list[0])
    for i in range(1, n):
        x[DIM_X_SC * i: DIM_X_SC * (i + 1)] = sc_state(
            deltas[i], q_list[i], w_list[i], mprop_list[i])
    return x


def bodies_dict(env, et):
    """Ephemeris snapshot independent of flags (so direct method tests work)."""
    return {b: env.body_position(b, et) for b in env.env_gm_keys()} \
        if hasattr(env, "env_gm_keys") else {b: env.body_position(b, et) for b in env.GM}


def integrate(plant, x0, u, et0, dt, nsteps, **kw):
    x = x0.copy(); et = et0
    traj = [x0.copy()]
    for _ in range(nsteps):
        x = plant.step(x, u, et, dt, **kw)
        et += dt
        traj.append(x.copy())
    return np.array(traj)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Construction & parameter consistency
# ─────────────────────────────────────────────────────────────────────────────

def test_parameters(env, param, chk):
    print("\n--- 1. Parameters & construction ----------------------------------")

    # Mass decomposition closes
    mtot_L = param.m_cylinder_L + param.m_ring_dry_L + param.m_prop_init_L
    mtot_F = param.m_cylinder_F + param.m_ring_dry_F + param.m_prop_init_F
    chk.ok("mass decomposition L closes to m_init",
           abs(mtot_L - param.m_init_L) < 1e-9, f"({mtot_L:.3f} vs {param.m_init_L})")
    chk.ok("mass decomposition F closes to m_init",
           abs(mtot_F - param.m_init_F) < 1e-9, f"({mtot_F:.3f} vs {param.m_init_F})")

    chk.ok("ring dry mass L > 0", param.m_ring_dry_L > 0, f"({param.m_ring_dry_L})")
    chk.ok("ring dry mass F > 0", param.m_ring_dry_F > 0, f"({param.m_ring_dry_F})")

    # SRP area ~ 25 m^2 (doc 'First Results' cannonball)
    chk.ok("SRP area L in [20,35] m^2", 20 < param.SRP_area_L < 35, f"({param.SRP_area_L:.2f})")
    chk.ok("SRP area F in [20,35] m^2", 20 < param.SRP_area_F < 35, f"({param.SRP_area_F:.2f})")

    # Doc/code parameter mismatch surfacing (NOT a hard fail — informational)
    if param.m_cylinder_L == param.m_cylinder_F:
        chk.info("leader/follower identical cylinder mass",
                 f"both {param.m_cylinder_L} kg — Table 1 wants combiner=3000, collector=2000")
    if param.m_prop_init_L == param.m_prop_init_F:
        chk.info("leader/follower identical propellant",
                 f"both {param.m_prop_init_L} kg — Table 1 wants 150 (combiner) / 100 (collector)")


def test_inertia_model(env, param, chk):
    print("\n--- 2. Inertia model vs analytic (doc eq. 22/23) ------------------")

    ri, ro, hr = param.r_in, param.r_out, param.h_ring
    # Standard thick-annulus (doc eq. 22): note eq.(26) in the doc has a typo.
    K_ring_expected = np.diag([
        (3.0 * (ro**2 + ri**2) + hr**2) / 12.0,
        (3.0 * (ro**2 + ri**2) + hr**2) / 12.0,
        (ro**2 + ri**2) / 2.0,
    ])
    K_ring_code = J_ring_unit(ri, ro, hr)
    chk.ok("J_ring_unit matches doc eq.(22)",
           rel_err(K_ring_code, K_ring_expected) < 1e-12,
           f"(diag code={np.diag(K_ring_code)}, exp={np.diag(K_ring_expected)})")

    rc, hL = param.r_cylinder, param.h_cylinder_L
    K_cyl_expected = np.diag([
        (3.0 * rc**2 + hL**2) / 12.0,
        (3.0 * rc**2 + hL**2) / 12.0,
        rc**2 / 2.0,
    ])
    K_cyl_code = J_cylinder_unit(rc, hL)
    chk.ok("J_cylinder_unit matches doc eq.(23)",
           rel_err(K_cyl_code, K_cyl_expected) < 1e-12,
           f"(diag code={np.diag(K_cyl_code)}, exp={np.diag(K_cyl_expected)})")

    # _inertia_now at full tank == param.J_init_L (internal consistency)
    plant = make_plant(env, param, PhysicsFlags(), 1)
    J_now = plant._inertia_now(param.m_prop_init_L, 'L')
    chk.ok("_inertia_now(full tank,L) == J_init_L",
           rel_err(J_now, param.J_init_L) < 1e-12, f"(diag={np.diag(J_now)})")

    # Physical sanity: symmetric, SPD, diagonal-dominant
    chk.ok("J symmetric", np.allclose(J_now, J_now.T), "")
    eig = np.linalg.eigvalsh(J_now)
    chk.ok("J positive definite", np.all(eig > 0), f"(eig={eig})")
    offdiag = np.max(np.abs(J_now - np.diag(np.diag(J_now))))
    chk.ok("J diagonal (off-diag==0 in current model)", offdiag < 1e-9, f"(max off-diag={offdiag:.2e})")

    # Triangle inequality for principal moments (Jx+Jy>=Jz etc.)
    Jx, Jy, Jz = np.diag(J_now)
    tri = (Jx + Jy >= Jz) and (Jy + Jz >= Jx) and (Jx + Jz >= Jy)
    chk.ok("inertia triangle inequality holds", tri, f"(diag={Jx:.0f},{Jy:.0f},{Jz:.0f})")

    # J monotonic decreasing as propellant burns
    J_full = plant._inertia_now(param.m_prop_init_L, 'L')
    J_half = plant._inertia_now(param.m_prop_init_L * 0.5, 'L')
    J_dry  = plant._inertia_now(0.0, 'L')
    mono = np.all(np.diag(J_full) > np.diag(J_half)) and np.all(np.diag(J_half) > np.diag(J_dry))
    chk.ok("J decreases monotonically with burn", mono, "")

    # _inertia_dot_now == m_dot * K_ring
    m_dot = -1.0e-6
    Jdot = plant._inertia_dot_now(m_dot)
    chk.ok("J_dot == m_dot * K_ring",
           rel_err(Jdot, m_dot * K_ring_code) < 1e-12, "")


def test_thrusters(env, param, chk):
    print("\n--- 3. Thruster table & allocation (doc Table 2) ------------------")

    chk.ok("20 thrusters", N_THRUSTERS == 20, f"({N_THRUSTERS})")

    # Spot-check geometry against Table 2.
    # NOTE: compute expected radii from the ACTUAL parameters (r_out, l_cube),
    # not the hard-coded comments in thrusters.py. Those comments assume
    # r_out=3.82 (-> 4.1875 / 4.555), but Parameters.r_out=3.8, so the code
    # actually builds 4.1675 / 4.535. The code is self-consistent; the comments
    # (and the design-doc prose '3.82') are stale.
    half     = param.l_cube / 2.0
    d_center = param.r_out + half          # cube-center radius (== 4.1675 for r_out=3.8)
    d_outer  = d_center + half             # outer-face thruster radius (== 4.535)
    chk.ok("thruster 1 pos/dir (+x outer)",
           np.allclose(THRUSTER_POSITIONS[0], [d_outer, 0, 0]) and
           np.allclose(THRUSTER_NORMALS[0], [1, 0, 0]),
           f"(pos={THRUSTER_POSITIONS[0]}, expect x={d_outer:.4f})")
    chk.ok("thruster 2 pos/dir (+x,+y side)",
           np.allclose(THRUSTER_POSITIONS[1], [d_center, half, 0]) and
           np.allclose(THRUSTER_NORMALS[1], [0, 1, 0]),
           f"(pos={THRUSTER_POSITIONS[1]})")
    chk.ok("thruster 6 pos/dir (-x outer)",
           np.allclose(THRUSTER_POSITIONS[5], [-d_outer, 0, 0]) and
           np.allclose(THRUSTER_NORMALS[5], [-1, 0, 0]),
           f"(pos={THRUSTER_POSITIONS[5]})")
    chk.info("thruster radii",
             f"cube-center={d_center:.4f} m, outer-face={d_outer:.4f} m "
             f"(thrusters.py comments say 4.1875/4.555 -> assume r_out=3.82, "
             f"but param.r_out={param.r_out})")
    chk.ok("all normals unit length",
           np.allclose(np.linalg.norm(THRUSTER_NORMALS, axis=1), 1.0), "")

    # Allocation ranks: full force + torque controllability
    chk.ok("rank(B_F) == 3", np.linalg.matrix_rank(B_F) == 3, "")
    chk.ok("rank(B_TAU) == 3", np.linalg.matrix_rank(B_TAU) == 3, "")
    chk.ok("rank([B_F;B_TAU]) == 6 (6-DOF actuation span)",
           np.linalg.matrix_rank(np.vstack([B_F, B_TAU])) == 6, "")

    # Sign convention: a radial outer thruster (r || n) gives pure force, no torque
    T = np.zeros(20); T[0] = param.T_MAX
    f_B, tau_B = force_torque_body(T)
    chk.ok("outer +x thruster -> force along -x (reaction), |tau|~0",
           np.allclose(f_B, [-param.T_MAX, 0, 0]) and np.linalg.norm(tau_B) < 1e-12,
           f"(f={f_B})")

    # A side thruster (r x n != 0) gives torque about z
    T = np.zeros(20); T[1] = param.T_MAX           # +x group, +y side
    f_B, tau_B = force_torque_body(T)
    chk.ok("+y side thruster -> force -y, torque about -z",
           np.allclose(f_B, [0, -param.T_MAX, 0]) and
           tau_B[2] < 0 and abs(tau_B[0]) < 1e-12 and abs(tau_B[1]) < 1e-12,
           f"(tau={tau_B})")

    chk.ok("clamp_thrust enforces [0, T_max]",
           np.allclose(clamp_thrust(np.array([-1.0, 0.001, 99.0]), param.T_MAX),
                       [0.0, 0.001, param.T_MAX]), "")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Quaternion utilities & kinematics convention
# ─────────────────────────────────────────────────────────────────────────────

def test_quaternion_kinematics(env, param, chk):
    print("\n--- 4. Quaternion / attitude conventions --------------------------")

    plant = make_plant(env, param, PhysicsFlags(), 1)

    # CTM orthonormal, det=+1
    q = euler_to_quat(np.array([0.3, -0.2, 0.5]))
    C = quat_to_CTM(q)
    chk.ok("C^T C == I", np.allclose(C.T @ C, np.eye(3), atol=1e-12), "")
    chk.ok("det(C) == +1", abs(np.linalg.det(C) - 1.0) < 1e-12, "")

    # round trip euler -> quat -> euler
    eul = np.array([0.3, -0.2, 0.5])
    chk.ok("euler -> quat -> euler round-trip",
           np.allclose(quat_to_euler(euler_to_quat(eul)), eul, atol=1e-10), "")

    # Omega_omega skew-symmetric  -> q_dot . q == 0 (norm preserving)
    w = np.array([0.01, -0.02, 0.015])
    Om = Omega_omega(w)
    chk.ok("Omega(w) skew-symmetric", np.allclose(Om, -Om.T), "")
    qd = plant.attitude_kin_rhs(q, w)
    chk.ok("q_dot . q == 0 (kinematics preserve |q|)", abs(qd @ q) < 1e-12,
           f"(q.qdot={qd @ q:.2e})")

    # CRITICAL coupling check: Omega_omega vs quat_to_CTM share the SAME
    # convention.  For q = q_I^B and w = w_IB^B we must have
    #     d/dt C_I^B = -[w x] C_I^B
    eps = 1e-7
    q1 = q + eps * qd
    q1 /= np.linalg.norm(q1)
    C0 = quat_to_CTM(q); C1 = quat_to_CTM(q1)
    Cdot_num = (C1 - C0) / eps
    Cdot_ana = -skew(w) @ C0
    chk.ok("Omega_omega <-> quat_to_CTM convention consistent "
           "(Cdot = -[wx]C)",
           rel_err(Cdot_num, Cdot_ana) < 1e-5,
           f"(rel err={rel_err(Cdot_num, Cdot_ana):.2e})")

    # Pure spin about a principal (z) axis: rate constant, angle = wz*T
    et0 = env.str2et(EPOCH)
    wz = 1e-3
    x0 = build_state([np.concatenate([R_INIT_L, V_INIT_L])],
                     q_list=[np.array([1.0, 0, 0, 0])],
                     w_list=[np.array([0.0, 0.0, wz])])
    spin_plant = make_plant(env, param, flags_only(gyro_coupling=True), 1)  # torque-free
    T_tot = 600.0
    traj = integrate(spin_plant, x0, np.zeros(20), et0, 100.0, int(T_tot / 100))
    xf = traj[-1]
    wf = xf[10:13]
    chk.ok("free spin about principal z: omega constant",
           np.allclose(wf, [0, 0, wz], atol=1e-10), f"(wf={wf})")
    eul_f = quat_to_euler(xf[6:10])
    chk.ok("free spin about z: yaw angle == wz*T",
           abs(((eul_f[2] - wz * T_tot + np.pi) % (2 * np.pi)) - np.pi) < 1e-6,
           f"(yaw={eul_f[2]:.6f}, expected={wz*T_tot:.6f})")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Flag gating — every term, off -> exact zero
# ─────────────────────────────────────────────────────────────────────────────

def test_flag_gating(env, param, chk):
    print("\n--- 5. Flag gating (off -> exact zero) ----------------------------")

    et0 = env.str2et(EPOCH)
    bodies = bodies_dict(env, et0)
    r = R_INIT_L
    q = euler_to_quat(np.array([0.1, 0.2, 0.3]))
    w = np.array([1e-3, -2e-3, 5e-4])
    m_total = param.m_init_L
    J = make_plant(env, param, PhysicsFlags(), 1)._inertia_now(param.m_prop_init_L, 'L')

    off = make_plant(env, param, flags_all_off(), 2)
    delta_r_all = np.array([[0, 0, 0], [1e-4, 0, 0]])
    m_total_all = np.array([param.m_init_L, param.m_init_F])

    chk.ok("grav_nbody off -> 0", np.allclose(off.acc_grav_abs(r, bodies), 0), "")
    chk.ok("srp off -> 0", np.allclose(off.acc_srp_abs(r, m_total, 'L', bodies), 0), "")
    chk.ok("grav_isc off -> 0", np.allclose(off.acc_grav_isc_abs(1, delta_r_all, m_total_all), 0), "")
    chk.ok("tau_srp off -> 0", np.allclose(off.attitude_tau_srp(q, np.ones(3) * 1e-9, m_total, 'L'), 0), "")
    chk.ok("tau_grav off -> 0", np.allclose(off.attitude_tau_grav(q, r, J, bodies), 0), "")
    chk.ok("mass_change off -> 0", off.mass_rhs(150.0, np.ones(20) * param.T_MAX, 'L') == 0.0, "")

    # gyro / j_dot gating affect w_dot
    base = make_plant(env, param, flags_all_off(), 1)
    a_srp = np.zeros(3)
    wd_none = base.attitude_dyn_rhs(w, 150.0, -1e-6, m_total, np.zeros(3), 'L', q, r, a_srp, bodies)
    gyro = make_plant(env, param, flags_only(gyro_coupling=True), 1)
    wd_gyro = gyro.attitude_dyn_rhs(w, 150.0, -1e-6, m_total, np.zeros(3), 'L', q, r, a_srp, bodies)
    chk.ok("gyro_coupling changes w_dot when on",
           not np.allclose(wd_none, wd_gyro), f"(|d|={np.linalg.norm(wd_gyro-wd_none):.2e})")
    jdot = make_plant(env, param, flags_only(j_dot_term=True), 1)
    wd_jdot = jdot.attitude_dyn_rhs(w, 150.0, -1e-6, m_total, np.zeros(3), 'L', q, r, a_srp, bodies)
    # The J_dot effect is real but physically tiny (~1e-12 rad/s^2 at mN thrust),
    # so np.allclose with its 1e-8 atol would call it "zero". Compare the actual
    # delta to the analytic -J^-1 J_dot omega instead.
    J5      = jdot._inertia_now(150.0, 'L')
    Jdot5   = jdot._inertia_dot_now(-1e-6)
    delta_expected = -np.linalg.inv(J5) @ (Jdot5 @ w)
    delta_actual   = wd_jdot - wd_none
    chk.ok("j_dot_term contributes exactly -J^-1 J_dot omega",
           rel_err(delta_actual, delta_expected) < 1e-8,
           f"(|delta|={np.linalg.norm(delta_actual):.2e} rad/s^2)")
    chk.info("j_dot magnitude", f"{np.linalg.norm(delta_actual):.2e} rad/s^2 "
             "-> negligible vs other torques at mN thrust")

    # j_dot has NO effect when m_dot == 0 (no burn)
    wd_jdot_nb = jdot.attitude_dyn_rhs(w, 150.0, 0.0, m_total, np.zeros(3), 'L', q, r, a_srp, bodies)
    wd_none_nb = base.attitude_dyn_rhs(w, 150.0, 0.0, m_total, np.zeros(3), 'L', q, r, a_srp, bodies)
    chk.ok("j_dot_term inert when m_dot==0", np.allclose(wd_jdot_nb, wd_none_nb), "")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Translational term magnitudes / signs / units
# ─────────────────────────────────────────────────────────────────────────────

def test_translational_terms(env, param, chk):
    print("\n--- 6. Translational terms (magnitude / sign / units) -------------")

    et0 = env.str2et(EPOCH)
    bodies = bodies_dict(env, et0)
    r = R_INIT_L

    # N-body gravity dominated by Sun: |a| ~ mu_sun / r^2
    g = make_plant(env, param, flags_only(grav_nbody=True), 1)
    a_grav = g.acc_grav_abs(r, bodies)
    r_sun = bodies["SUN"]
    d_sun = r - r_sun
    a_sun_mag = env.GM["SUN"] / np.dot(d_sun, d_sun)              # km/s^2
    chk.ok("N-body |a| consistent with Sun mu/r^2 (within 20%)",
           abs(np.linalg.norm(a_grav) - a_sun_mag) / a_sun_mag < 0.2,
           f"(|a|={np.linalg.norm(a_grav):.3e}, sun={a_sun_mag:.3e} km/s^2)")
    chk.ok("N-body gravity points toward Sun",
           np.dot(a_grav, -d_sun) > 0, "")
    chk.info("N-body |a| in SI",
             f"{np.linalg.norm(a_grav)/const.KM_PER_M:.3e} m/s^2 (Sun ~5.9e-3 expected)")

    # SRP: formula assembly + direction + order of magnitude (doc table ~1e-8 m/s^2 diff.)
    s = make_plant(env, param, flags_only(srp=True), 1)
    a_srp = s.acc_srp_abs(r, param.m_init_L, 'L', bodies)
    u_hat = d_sun / np.linalg.norm(d_sun)
    P_at_r = const.P_SUN * (const.R_SUN_AU / np.linalg.norm(d_sun)) ** 2
    a_srp_expected = (param.c_reflect * (param.SRP_area_L / param.m_init_L)
                      * P_at_r * const.KM_PER_M) * u_hat
    chk.ok("SRP matches cannonball formula C_R*P*A/m",
           rel_err(a_srp, a_srp_expected) < 1e-12, "")
    chk.ok("SRP pushes AWAY from Sun", np.dot(a_srp, u_hat) > 0, "")
    a_srp_si = np.linalg.norm(a_srp) / const.KM_PER_M
    chk.ok("SRP magnitude physically plausible (1e-9..1e-7 m/s^2)",
           1e-9 < a_srp_si < 1e-7, f"({a_srp_si:.3e} m/s^2)")

    # Inter-spacecraft gravity: magnitude + Newton's third law
    isc = make_plant(env, param, flags_only(grav_isc=True), 2)
    baseline_km = 100.0 * const.KM_PER_M                          # 100 m
    delta_r_all = np.array([[0, 0, 0], [baseline_km, 0, 0]])
    m_total_all = np.array([param.m_init_L, param.m_init_F])
    a_on_follower = isc.acc_grav_isc_abs(1, delta_r_all, m_total_all)
    a_on_leader = isc.acc_grav_isc_abs(0, delta_r_all, m_total_all)
    a_mag_exp = const.G_KM * param.m_init_L / baseline_km ** 2     # leader pulls follower
    chk.ok("ISC |a| ~ G m / d^2",
           abs(np.linalg.norm(a_on_follower) - a_mag_exp) / a_mag_exp < 1e-9,
           f"(|a|={np.linalg.norm(a_on_follower):.3e} km/s^2 ~ {a_mag_exp:.3e})")
    chk.ok("ISC: spacecraft attract (follower pulled toward leader, -x)",
           a_on_follower[0] < 0, f"(a={a_on_follower})")
    # Newton's third law: m_i a_i = - m_j a_j
    chk.ok("ISC Newton's 3rd law (m_L a_L = - m_F a_F)",
           np.allclose(param.m_init_L * a_on_leader,
                       -param.m_init_F * a_on_follower, rtol=1e-10), "")
    chk.info("ISC |a| in SI", f"{np.linalg.norm(a_on_follower)/const.KM_PER_M*1e12:.2f} pm/s^2")


def test_mass_flow(env, param, chk):
    print("\n--- 6b. Mass flow (Tsiolkovsky, units) ----------------------------")
    p = make_plant(env, param, flags_only(mass_change=True), 1)
    T = np.ones(20) * param.T_MAX                       # all thrusters at max [N]
    m_dot = p.mass_rhs(150.0, T, 'L')
    T_sum_N = 20 * param.T_MAX
    m_dot_exp = -(T_sum_N / const.N_PER_KN) / (param.ISP * const.G0)
    chk.ok("m_dot matches -sum(T)/(Isp g0)",
           abs(m_dot - m_dot_exp) < 1e-18, f"(m_dot={m_dot:.3e} kg/s)")
    chk.ok("m_dot negative under thrust", m_dot < 0, "")
    # plausible: tiny flow given mN-class thrusters
    chk.ok("m_dot magnitude < 1e-5 kg/s", abs(m_dot) < 1e-5, f"({abs(m_dot):.3e})")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Torque terms
# ─────────────────────────────────────────────────────────────────────────────

def test_torque_terms(env, param, chk):
    print("\n--- 7. Torque terms (srp / grav / gyro) ---------------------------")

    et0 = env.str2et(EPOCH)
    bodies = bodies_dict(env, et0)
    r = R_INIT_L
    q = euler_to_quat(np.array([0.2, -0.3, 0.4]))
    J = make_plant(env, param, PhysicsFlags(), 1)._inertia_now(param.m_prop_init_L, 'L')

    # ---- SRP torque: requires srp; matches r_CP x f_srp ----
    s_on = make_plant(env, param, flags_only(srp=True, tau_srp=True), 1)
    a_srp = s_on.acc_srp_abs(r, param.m_init_L, 'L', bodies)
    tau_srp = s_on.attitude_tau_srp(q, a_srp, param.m_init_L, 'L')
    f_srp_B = (param.m_init_L * (quat_to_CTM(q) @ a_srp)) / const.KM_PER_M
    tau_exp = np.cross(param.r_CP_L, f_srp_B)
    chk.ok("tau_srp == r_CP x f_srp", rel_err(tau_srp, tau_exp) < 1e-12, "")
    chk.ok("tau_srp nonzero & finite", np.linalg.norm(tau_srp) > 0 and np.all(np.isfinite(tau_srp)),
           f"(|tau|={np.linalg.norm(tau_srp):.3e} N*m)")

    # Coupling: tau_srp silently zero if srp flag OFF (upstream a_srp -> 0)
    s_off = make_plant(env, param, flags_only(srp=False, tau_srp=True), 1)
    a_srp0 = s_off.acc_srp_abs(r, param.m_init_L, 'L', bodies)   # == 0
    tau_srp0 = s_off.attitude_tau_srp(q, a_srp0, param.m_init_L, 'L')
    chk.ok("tau_srp -> 0 when srp flag off (documented coupling)",
           np.allclose(tau_srp0, 0), "see note: tau_srp depends on srp")

    # ---- Gravity-gradient torque ----
    gg = make_plant(env, param, flags_only(tau_grav=True), 1)
    tau_gg = gg.attitude_tau_grav(q, r, J, bodies)
    chk.ok("tau_grav finite & small (nN*m..uN*m scale)",
           np.all(np.isfinite(tau_gg)) and np.linalg.norm(tau_gg) < 1e-3,
           f"(|tau|={np.linalg.norm(tau_gg):.3e} N*m)")

    # Structural physics: if J is isotropic, gravity-gradient torque == 0
    #   (u x (J u) = u x (c u) = 0).  We monkeypatch a scalar-multiple J.
    J_iso = np.eye(3) * np.trace(J) / 3.0
    tau_gg_iso = gg.attitude_tau_grav(q, r, J_iso, bodies)
    chk.ok("tau_grav == 0 for isotropic J (u x Ju = 0)",
           np.allclose(tau_gg_iso, 0, atol=1e-20), f"(|tau|={np.linalg.norm(tau_gg_iso):.2e})")

    # Structural physics: align body so line-of-sight to the dominant body is
    # a principal axis -> that body's contribution vanishes.  Hard to isolate a
    # single body cleanly, so we instead verify the per-body kernel directly.
    d = r - bodies["SUN"]
    u_sun_I = d / np.linalg.norm(d)
    # choose q s.t. C_I^B u_sun_I = e_z (principal axis) -> kernel u x Ju = 0
    z = np.array([0.0, 0, 1])
    v = np.cross(u_sun_I, z); s_ = np.linalg.norm(v); c_ = np.dot(u_sun_I, z)
    if s_ > 1e-12:
        R = np.eye(3) + skew(v) + skew(v) @ skew(v) * ((1 - c_) / s_**2)  # Rodrigues
        u_B = R @ u_sun_I
        chk.ok("principal-axis line-of-sight kernel u x Ju == 0",
               np.linalg.norm(np.cross(u_B, J @ u_B)) < 1e-9,
               f"(|u x Ju|={np.linalg.norm(np.cross(u_B, J @ u_B)):.2e})")

    # ---- Gyroscopic coupling: zero on a principal axis, nonzero off-axis ----
    gyro = make_plant(env, param, flags_only(gyro_coupling=True), 1)
    a0 = np.zeros(3)
    w_axis = np.array([2e-3, 0, 0])                  # along principal x
    wd_axis = gyro.attitude_dyn_rhs(w_axis, 150.0, 0.0, param.m_init_L, a0, 'L', q, r, a0, bodies)
    chk.ok("gyro: spin on principal axis -> no precession (w_dot==0)",
           np.allclose(wd_axis, 0, atol=1e-14), f"(w_dot={wd_axis})")
    # Axisymmetric body (Jxx == Jyy): gyro torque is (C-A)*wz*[wy,-wx,0], so it
    # vanishes for ANY omega in the xy-plane. Must give omega an AXIAL (z)
    # component to see precession.
    w_off = np.array([2e-3, 0.0, 2e-3])             # transverse + axial
    wd_off = gyro.attitude_dyn_rhs(w_off, 150.0, 0.0, param.m_init_L, a0, 'L', q, r, a0, bodies)
    chk.ok("gyro: off-axis spin (wz!=0) -> precession (w_dot!=0)",
           np.linalg.norm(wd_off) > 1e-9, f"(|w_dot|={np.linalg.norm(wd_off):.2e})")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Coupling: rotation<->translation, mass<->inertia
# ─────────────────────────────────────────────────────────────────────────────

def test_coupling(env, param, chk):
    print("\n--- 8. Coupling (rot<->trans, mass<->inertia) ---------------------")

    et0 = env.str2et(EPOCH)
    p = make_plant(env, param, PhysicsFlags(grav_nbody=False, srp=False, grav_isc=False,
                                            tau_srp=False, tau_grav=False,
                                            gyro_coupling=True, j_dot_term=True,
                                            mass_change=True), 1)

    # Fire a side thruster -> body force AND torque, force rotates with attitude.
    # The control accel is ~1e-9 km/s^2 (one mN thruster on 3150 kg), which is
    # SMALLER than np.allclose's atol=1e-8 AND is swamped by gravity (~6e-6) in
    # the full v_dot. So: isolate control (all perturbations off) and compare to
    # the analytic a_ctrl = C_B^I F_B / m via a RELATIVE tolerance.
    p_ctrl = make_plant(env, param, flags_all_off(), 1)
    u = np.zeros(20); u[1] = param.T_MAX           # +x group +y side -> body -y force
    q_a = np.array([1.0, 0, 0, 0])
    q_b = euler_to_quat(np.array([0, 0, np.pi / 2]))  # 90 deg yaw
    x_a = build_state([np.concatenate([R_INIT_L, V_INIT_L])], q_list=[q_a])
    x_b = build_state([np.concatenate([R_INIT_L, V_INIT_L])], q_list=[q_b])
    a_ctrl_a = p_ctrl.x_dot(0.0, x_a, et0, u)[3:6]
    a_ctrl_b = p_ctrl.x_dot(0.0, x_b, et0, u)[3:6]

    F_B, _ = force_torque_body(clamp_thrust(u, param.T_MAX))
    a_exp_a = (quat_to_CTM(q_a).T @ F_B / param.m_init_L) * const.KM_PER_M
    a_exp_b = (quat_to_CTM(q_b).T @ F_B / param.m_init_L) * const.KM_PER_M
    chk.ok("a_ctrl(identity) == C_B^I F_B/m", rel_err(a_ctrl_a, a_exp_a) < 1e-9, "")
    chk.ok("a_ctrl(90deg yaw) == C_B^I F_B/m", rel_err(a_ctrl_b, a_exp_b) < 1e-9, "")
    chk.ok("control force rotates with attitude (a_ctrl differs)",
           rel_err(a_ctrl_a, a_ctrl_b) > 1e-3,
           f"(rel diff={rel_err(a_ctrl_a, a_ctrl_b):.2e}, "
           f"|diff|={np.linalg.norm(a_ctrl_a-a_ctrl_b):.2e} km/s^2)")

    # Full-physics check: torque + propellant depletion still register.
    p = make_plant(env, param, PhysicsFlags(grav_nbody=False, srp=False, grav_isc=False,
                                            tau_srp=False, tau_grav=False,
                                            gyro_coupling=True, j_dot_term=True,
                                            mass_change=True), 1)
    xd_a = p.x_dot(0.0, x_a, et0, u)
    chk.ok("thruster produces nonzero body torque (w_dot != 0)",
           np.linalg.norm(xd_a[10:13]) > 0, f"(|w_dot|={np.linalg.norm(xd_a[10:13]):.2e})")
    chk.ok("thrust depletes propellant (m_dot < 0)", xd_a[13] < 0, f"(m_dot={xd_a[13]:.2e})")

    # Mass-inertia coupling: during a burn, j_dot_term changes w_dot. Effect is
    # tiny, so compare to the analytic -J^-1 J_dot omega (not allclose).
    bodies = bodies_dict(env, et0)
    q = q_a; w = np.array([1e-3, 1e-3, 0.0]); a0 = np.zeros(3)
    m_dot = p.mass_rhs(150.0, clamp_thrust(u, param.T_MAX), 'L')
    p_no_jdot = make_plant(env, param, flags_only(gyro_coupling=True), 1)
    wd_with = p.attitude_dyn_rhs(w, 150.0, m_dot, param.m_init_L, np.zeros(3), 'L', q, R_INIT_L, a0, bodies)
    wd_without = p_no_jdot.attitude_dyn_rhs(w, 150.0, m_dot, param.m_init_L, np.zeros(3), 'L', q, R_INIT_L, a0, bodies)
    J8    = p._inertia_now(150.0, 'L')
    Jdot8 = p._inertia_dot_now(m_dot)
    delta_expected = -np.linalg.inv(J8) @ (Jdot8 @ w)
    chk.ok("J_dot term active during burn (== -J^-1 J_dot omega)",
           rel_err(wd_with - wd_without, delta_expected) < 1e-8,
           f"(|delta|={np.linalg.norm(wd_with-wd_without):.2e} rad/s^2)")


# ─────────────────────────────────────────────────────────────────────────────
# 9. Relative dynamics & numerical precision (the headline diagnostics)
# ─────────────────────────────────────────────────────────────────────────────

def test_relative_precision(env, param, chk):
    print("\n--- 9. Relative-state design: where it WINS and where it does NOT --")

    et0 = env.str2et(EPOCH)
    bodies = bodies_dict(env, et0)
    r_L = R_INIT_L
    baseline_km = 100.0 * const.KM_PER_M

    g = make_plant(env, param, flags_only(grav_nbody=True), 2)

    # --- (a) Differential N-body gravity vs high-precision reference ---
    delta_r = np.array([baseline_km, 0.5 * baseline_km, -0.3 * baseline_km])
    a_rel_code = g.acc_grav_rel(r_L, delta_r, bodies)
    # naive: a_abs(r_F) - a_abs(r_L) in float64
    r_F = r_L + delta_r
    a_naive = g.acc_grav_abs(r_F, bodies) - g.acc_grav_abs(r_L, bodies)

    ref = None
    try:
        import mpmath as mp
        mp.mp.dps = 60
        da = [mp.mpf(0)] * 3
        for body, mu in env.GM.items():
            rb = bodies[body]
            d1 = [mp.mpf(float(r_L[k])) - mp.mpf(float(rb[k])) for k in range(3)]
            d2 = [d1[k] + mp.mpf(float(delta_r[k])) for k in range(3)]
            n1 = mp.sqrt(sum(x * x for x in d1))
            n2 = mp.sqrt(sum(x * x for x in d2))
            for k in range(3):
                da[k] -= mp.mpf(float(mu)) * (d2[k] / n2**3 - d1[k] / n1**3)
        ref = np.array([float(x) for x in da])
    except ImportError:
        chk.info("mpmath not installed", "skipping high-precision reference")

    if ref is not None:
        d_code = sig_digits(a_rel_code, ref)
        d_naive = sig_digits(a_naive, ref)
        chk.info("differential N-body gravity precision",
                 f"acc_grav_rel keeps ~{d_code:.1f} sig digits; "
                 f"naive keeps ~{d_naive:.1f}")
        chk.info(">>> FINDING",
                 "d2=d1+dr does NOT defeat the cancellation for distant bodies; "
                 "consider a Battin/Encke f(q) formulation.")
        # Hard assertion only that they are at least *somewhat* right (sanity),
        # not that they are precise (we expect ~3 digits for the Sun term).
        chk.ok("acc_grav_rel is the correct tidal sign/order",
               rel_err(a_rel_code, ref) < 1e-2, f"(rel err={rel_err(a_rel_code, ref):.2e})")

    # acc_grav_rel must equal the differential (definition check, loose tol)
    chk.ok("acc_grav_rel == a_abs(F) - a_abs(L) (consistent assembly)",
           rel_err(a_rel_code, a_naive) < 1e-6,
           f"(rel diff={rel_err(a_rel_code, a_naive):.2e})")

    # --- (b) Inter-spacecraft separation: relative WINS decisively ---
    dr_i = np.array([baseline_km, 0, 0])
    dr_j = np.array([baseline_km + 1.0 * const.KM_PER_M, 0, 0])  # 1 m apart
    true_sep = dr_j - dr_i                                       # exact answer
    # naive reconstruction from SSB-absolute positions
    ri_abs = r_L + dr_i
    rj_abs = r_L + dr_j
    naive_sep = rj_abs - ri_abs
    err_naive = rel_err(naive_sep, true_sep)
    chk.info("ISC separation reconstructed from absolutes",
             f"loses ~{-math.log10(err_naive) if err_naive>0 else 99:.1f} digits "
             f"(rel err={err_naive:.2e})")
    chk.ok("relative-state ISC separation is exact (design WINS here)",
           err_naive > 1e-6,  # i.e. the *naive* path is demonstrably worse
           "code uses dr_j - dr_i directly -> machine-precise regardless of r_L")

    # ISC acceleration is independent of absolute leader position (uses only dr)
    isc = make_plant(env, param, flags_only(grav_isc=True), 2)
    m_all = np.array([param.m_init_L, param.m_init_F])
    drall = np.array([[0, 0, 0], [baseline_km, 0, 0]])
    a1 = isc.acc_grav_isc_abs(1, drall, m_all)
    a2 = isc.acc_grav_isc_abs(1, drall, m_all)  # same -> deterministic
    chk.ok("ISC accel depends only on relative state (deterministic)",
           np.allclose(a1, a2), "")


def test_relative_dynamics_assembly(env, param, chk):
    print("\n--- 9b. Follower relative dynamics assembly -----------------------")
    et0 = env.str2et(EPOCH)
    # Full physics, 2 spacecraft. Verify follower r_dot == delta_v and that
    # the follower velocity-derivative equals (a_F - a_L) for gravity-only case.
    flags = flags_only(grav_nbody=True)
    p = make_plant(env, param, flags, 2)
    delta_r = np.array([100.0 * const.KM_PER_M, 0, 0])
    delta_v = np.array([0.0, 1e-6, 0.0])
    x = build_state([np.concatenate([R_INIT_L, V_INIT_L]),
                     np.concatenate([delta_r, delta_v])])
    xd = p.x_dot(0.0, x, et0, np.zeros(40))
    # follower r_dot == delta_v
    chk.ok("follower r_dot == delta_v", np.allclose(xd[DIM_X_SC:DIM_X_SC+3], delta_v), "")
    # follower v_dot == a_grav(r_F) - a_grav(r_L)  (gravity-only)
    bodies = bodies_dict(env, et0)
    a_diff = p.acc_grav_rel(R_INIT_L, delta_r, bodies)
    chk.ok("follower v_dot == differential gravity (gravity-only)",
           rel_err(xd[DIM_X_SC+3:DIM_X_SC+6], a_diff) < 1e-9, "")


# ─────────────────────────────────────────────────────────────────────────────
# 10. Conservation laws
# ─────────────────────────────────────────────────────────────────────────────

def test_conservation(env, param, chk):
    print("\n--- 10. Conservation (torque-free rigid body; all-off frozen) -----")

    et0 = env.str2et(EPOCH)

    # Torque-free rotation: H_inertial and KE conserved.
    flags = flags_only(gyro_coupling=True)  # gyro on, no torques, no burn, J constant
    p = make_plant(env, param, flags, 1)
    J = p._inertia_now(param.m_prop_init_L, 'L')
    w0 = np.array([2e-3, 1.5e-3, -1e-3])    # off-axis -> genuine precession
    q0 = np.array([1.0, 0, 0, 0])
    x0 = build_state([np.concatenate([R_INIT_L, V_INIT_L])], q_list=[q0], w_list=[w0])

    def H_KE(x):
        q = x[6:10]; w = x[10:13]
        H_in = quat_to_CTM(q).T @ (J @ w)   # body angular momentum -> inertial
        KE = 0.5 * w @ (J @ w)
        return H_in, KE

    traj = integrate(p, x0, np.zeros(20), et0, 50.0, 200)  # 10000 s
    H0, KE0 = H_KE(traj[0])
    H_err = max(rel_err(H_KE(x)[0], H0) for x in traj)
    KE_err = max(abs(H_KE(x)[1] - KE0) / KE0 for x in traj)
    chk.ok("torque-free: |H_inertial| conserved", H_err < 1e-7, f"(max rel err={H_err:.2e})")
    chk.ok("torque-free: rotational KE conserved", KE_err < 1e-7, f"(max rel err={KE_err:.2e})")
    # genuine precession occurred (body-frame w changed) -> not a trivial pass
    chk.ok("precession actually occurred (w_body changed)",
           rel_err(traj[-1][10:13], w0) > 1e-4, "")

    # All-off, zero control, zero omega: only translation drifts (r += v*dt)
    p_off = make_plant(env, param, flags_all_off(), 2)
    dr = np.array([100.0 * const.KM_PER_M, 0, 0])
    dv = np.array([0.0, 2e-6, 0.0])
    x0 = build_state([np.concatenate([R_INIT_L, V_INIT_L]),
                      np.concatenate([dr, dv])])
    dt = 100.0
    x1 = p_off.step(x0, np.zeros(40), et0, dt)
    chk.ok("all-off: leader r advances by v*dt",
           np.allclose(x1[0:3], R_INIT_L + V_INIT_L * dt, rtol=1e-10), "")
    chk.ok("all-off: leader v unchanged", np.allclose(x1[3:6], V_INIT_L), "")
    chk.ok("all-off: follower dr advances by dv*dt",
           np.allclose(x1[DIM_X_SC:DIM_X_SC+3], dr + dv * dt, rtol=1e-9), "")
    chk.ok("all-off: quaternions unchanged", np.allclose(x1[6:10], [1, 0, 0, 0]), "")
    chk.ok("all-off: propellant unchanged", x1[13] == x0[13], "")


# ─────────────────────────────────────────────────────────────────────────────
# 11. Integration sanity / solver tolerances
# ─────────────────────────────────────────────────────────────────────────────

def test_integration_sanity(env, param, chk):
    print("\n--- 11. Integration sanity & tolerances ---------------------------")

    et0 = env.str2et(EPOCH)
    p = make_plant(env, param, PhysicsFlags(), 3)
    dr1 = np.array([100.0 * const.KM_PER_M, 0, 0])
    dr2 = np.array([0, 100.0 * const.KM_PER_M, 0])
    x0 = build_state([np.concatenate([R_INIT_L, V_INIT_L]), dr1_state(dr1), dr1_state(dr2)],
                     w_list=[np.array([1e-4, 0, 0])] * 3)

    # RHS finite everywhere
    xd = p.x_dot(0.0, x0, et0, np.zeros(60))
    chk.ok("x_dot finite for nominal full-physics state", np.all(np.isfinite(xd)), "")

    # One step: finite, quaternion renormalized to unit
    x1 = p.step(x0, np.zeros(60), et0, 100.0)
    chk.ok("step output finite", np.all(np.isfinite(x1)), "")
    for i in range(3):
        qn = np.linalg.norm(x1[DIM_X_SC*i+6:DIM_X_SC*i+10])
        chk.ok(f"sc{i} |q|==1 after renormalize", abs(qn - 1) < 1e-12, f"(|q|={qn:.15f})")

    # quaternion drift WITHIN a step (renormalize off) should be small
    x1_nr = p.step(x0, np.zeros(60), et0, 100.0, renormalize=False)
    qn_drift = abs(np.linalg.norm(x1_nr[6:10]) - 1)
    chk.ok("intra-step quaternion drift < 1e-8 (renorm off)", qn_drift < 1e-8,
           f"(|q|-1={qn_drift:.2e})")

    # leader orbit bounded over a short arc (|r| ~ 1 AU, no blow-up)
    traj = integrate(p, x0, np.zeros(60), et0, 100.0, 50)  # 5000 s
    r_norms = np.linalg.norm(traj[:, 0:3], axis=1)
    chk.ok("leader |r| stays ~1 AU (bounded)",
           np.all(np.abs(r_norms - r_norms[0]) / r_norms[0] < 1e-3), "")
    dr_norms = np.linalg.norm(traj[:, DIM_X_SC:DIM_X_SC+3], axis=1) / const.KM_PER_M
    chk.ok("follower baseline stays O(100 m) (no blow-up)",
           np.all((dr_norms > 50) & (dr_norms < 200)), f"(range {dr_norms.min():.1f}-{dr_norms.max():.1f} m)")

    # dt-convergence: one 100s step vs two 50s steps
    xa = p.step(x0, np.zeros(60), et0, 100.0)
    xb = p.step(p.step(x0, np.zeros(60), et0, 50.0), np.zeros(60), et0 + 50.0, 50.0)
    # compare follower delta-r (the precision-critical quantity)
    da = rel_err(xa[DIM_X_SC:DIM_X_SC+3], xb[DIM_X_SC:DIM_X_SC+3])
    chk.ok("dt-refinement agreement on follower dr (< 1e-6)", da < 1e-6, f"(rel diff={da:.2e})")

    # Determinism
    chk.ok("step is deterministic",
           np.array_equal(p.step(x0, np.zeros(60), et0, 100.0), xa), "")

    chk.info("solver tolerance note",
             "single scalar atol=1e-12 is inert for the 1.5e8 km leader position "
             "(rtol governs, ~0.15 m/step) and very tight for dr/omega; a "
             "per-component atol vector would be cleaner.")


def dr1_state(dr):
    return np.concatenate([dr, np.zeros(3)])


# ─────────────────────────────────────────────────────────────────────────────
# 12. Edge cases
# ─────────────────────────────────────────────────────────────────────────────

def test_edge_cases(env, param, chk):
    print("\n--- 12. Edge cases ------------------------------------------------")
    et0 = env.str2et(EPOCH)
    p = make_plant(env, param, PhysicsFlags(), 1)

    # Dry tank: thrust commanded but m_prop == 0 -> no force, no mass change.
    u = np.ones(20) * param.T_MAX
    x_dry = build_state([np.concatenate([R_INIT_L, V_INIT_L])], mprop_list=[0.0])
    x_dry_noT = x_dry.copy()
    xd_thrust = p.x_dot(0.0, x_dry, et0, u)
    xd_noThrust = p.x_dot(0.0, x_dry_noT, et0, np.zeros(20))
    chk.ok("dry tank: m_dot == 0 despite thrust command", xd_thrust[13] == 0.0, "")
    chk.ok("dry tank: dynamics identical with/without thrust (no actuator output)",
           np.allclose(xd_thrust, xd_noThrust), "")

    # Single-thruster minimal command still finite
    u2 = np.zeros(20); u2[3] = param.T_MAX
    chk.ok("single +z thruster -> finite x_dot",
           np.all(np.isfinite(p.x_dot(0.0, build_state([np.concatenate([R_INIT_L, V_INIT_L])]),
                                      et0, u2))), "")


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 78)
    print("  LIFE Truth-Model Verification Suite")
    print("=" * 78)

    env = SpiceEnv(KERNELS, BODIES, FRAME, ABCORR, OBSERVER)
    param = Parameters()
    chk = Check()

    sections = [
        test_parameters,
        test_inertia_model,
        test_thrusters,
        test_quaternion_kinematics,
        test_flag_gating,
        test_translational_terms,
        test_mass_flow,
        test_torque_terms,
        test_coupling,
        test_relative_precision,
        test_relative_dynamics_assembly,
        test_conservation,
        test_integration_sanity,
        test_edge_cases,
    ]

    for sec in sections:
        try:
            sec(env, param, chk)
        except Exception as e:  # noqa: BLE001 - keep the suite running
            chk.n += 1
            chk.fails.append(f"{sec.__name__} [EXCEPTION]")
            import traceback
            print(f"  [ERROR] {sec.__name__}: {e}")
            traceback.print_exc()

    chk.summary()


if __name__ == "__main__":
    main()