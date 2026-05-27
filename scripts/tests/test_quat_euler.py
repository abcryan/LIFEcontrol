"""Thorough correctness tests for Quat_to_Euler / Euler_to_Quat.

All conventions follow Groves, "Principles of GNSS, Inertial, and Multisensor
Integrated Navigation Systems," 2nd ed.:
  - Scalar-first quaternion q = [q0, q1, q2, q3], representing q_alpha^beta.
  - Euler angles (phi, theta, psi) = (roll, pitch, yaw), ZYX sequence,
    representing the rotation alpha -> beta.
  - CTM C_alpha^beta transforms a column vector from alpha to beta.
"""

import time
import numpy as np

from life_control.utils.coordinate_trafos.Quat_to_Euler import quat_to_euler
from life_control.utils.coordinate_trafos.Euler_to_Quat import euler_to_quat
from life_control.utils.coordinate_trafos.Quat_to_CTM import quat_to_CTM
from life_control.utils.coordinate_trafos.CTM_to_Quat import CTM_to_quat
from life_control.utils.coordinate_trafos.Euler_to_CTM    import euler_to_CTM
from life_control.utils.coordinate_trafos.CTM_to_Euler    import CTM_to_euler

# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------
ATOL_TIGHT = 1e-12   # exact algebraic checks
ATOL_LOOSE = 1e-10   # round-trip through trig
ATOL_RAND  = 1e-9    # random fuzzing
SING_EPS   = 1e-6    # gap from the +/- pi/2 gimbal-lock singularity


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _quat_equal_up_to_sign(q1, q2, atol):
    """Quaternions q and -q represent the same rotation."""
    return np.allclose(q1, q2, atol=atol) or np.allclose(q1, -q2, atol=atol)


def _wrap_pi(a):
    """Wrap angle to (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def _euler_equal(e1, e2, atol):
    """Equality of two Euler triples modulo 2pi on phi and psi."""
    d = np.array([_wrap_pi(e1[0] - e2[0]),
                  e1[1] - e2[1],
                  _wrap_pi(e1[2] - e2[2])])
    return np.max(np.abs(d)) <= atol


def _report(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    line = f"  [{status}] {name}"
    if detail and not ok:
        line += f"\n         {detail}"
    print(line)
    return ok


# ---------------------------------------------------------------------------
# Test 1: Identity
# ---------------------------------------------------------------------------
def test_identity():
    print("\nTest 1: identity quaternion / zero Euler angles")
    all_ok = True

    e = Quat_to_Euler(np.array([1.0, 0.0, 0.0, 0.0]))
    all_ok &= _report("Quat_to_Euler([1,0,0,0]) == [0,0,0]",
                      np.allclose(e, 0.0, atol=ATOL_TIGHT),
                      f"got {e}")

    q = Euler_to_Quat(np.array([0.0, 0.0, 0.0]))
    all_ok &= _report("Euler_to_Quat([0,0,0]) == [1,0,0,0]",
                      _quat_equal_up_to_sign(q, np.array([1, 0, 0, 0]), ATOL_TIGHT),
                      f"got {q}")
    return all_ok


# ---------------------------------------------------------------------------
# Test 2: Single-axis rotations
# ---------------------------------------------------------------------------
def test_single_axis():
    print("\nTest 2: pure single-axis rotations")
    all_ok = True
    angle = 0.37  # arbitrary mid-range angle

    # Roll only: phi = angle -> q = [cos(a/2), sin(a/2), 0, 0]
    q_expected = np.array([np.cos(angle/2), np.sin(angle/2), 0.0, 0.0])
    q = Euler_to_Quat(np.array([angle, 0.0, 0.0]))
    all_ok &= _report("Euler_to_Quat pure roll", 
                      _quat_equal_up_to_sign(q, q_expected, ATOL_TIGHT),
                      f"got {q}, expected {q_expected}")
    e = Quat_to_Euler(q_expected)
    all_ok &= _report("Quat_to_Euler pure roll",
                      np.allclose(e, [angle, 0, 0], atol=ATOL_LOOSE),
                      f"got {e}")

    # Pitch only: theta = angle -> q = [cos(a/2), 0, sin(a/2), 0]
    q_expected = np.array([np.cos(angle/2), 0.0, np.sin(angle/2), 0.0])
    q = Euler_to_Quat(np.array([0.0, angle, 0.0]))
    all_ok &= _report("Euler_to_Quat pure pitch",
                      _quat_equal_up_to_sign(q, q_expected, ATOL_TIGHT),
                      f"got {q}, expected {q_expected}")
    e = Quat_to_Euler(q_expected)
    all_ok &= _report("Quat_to_Euler pure pitch",
                      np.allclose(e, [0, angle, 0], atol=ATOL_LOOSE),
                      f"got {e}")

    # Yaw only: psi = angle -> q = [cos(a/2), 0, 0, sin(a/2)]
    q_expected = np.array([np.cos(angle/2), 0.0, 0.0, np.sin(angle/2)])
    q = Euler_to_Quat(np.array([0.0, 0.0, angle]))
    all_ok &= _report("Euler_to_Quat pure yaw",
                      _quat_equal_up_to_sign(q, q_expected, ATOL_TIGHT),
                      f"got {q}, expected {q_expected}")
    e = Quat_to_Euler(q_expected)
    all_ok &= _report("Quat_to_Euler pure yaw",
                      np.allclose(e, [0, 0, angle], atol=ATOL_LOOSE),
                      f"got {e}")

    return all_ok


# ---------------------------------------------------------------------------
# Test 3: Round-trip Euler -> Quat -> Euler (off singularity)
# ---------------------------------------------------------------------------
def test_roundtrip_euler():
    print("\nTest 3: round-trip Euler -> Quat -> Euler")
    rng = np.random.default_rng(0xC0FFEE)
    N = 1000
    phis   = rng.uniform(-np.pi, np.pi, N)
    thetas = rng.uniform(-np.pi/2 + SING_EPS, np.pi/2 - SING_EPS, N)
    psis   = rng.uniform(-np.pi, np.pi, N)

    max_err = 0.0
    worst   = None
    for phi, th, ps in zip(phis, thetas, psis):
        e_in  = np.array([phi, th, ps])
        e_out = Quat_to_Euler(Euler_to_Quat(e_in))
        d = np.array([_wrap_pi(e_in[0] - e_out[0]),
                      e_in[1] - e_out[1],
                      _wrap_pi(e_in[2] - e_out[2])])
        err = np.max(np.abs(d))
        if err > max_err:
            max_err, worst = err, (e_in, e_out)

    ok = max_err <= ATOL_RAND
    return _report(f"{N} random round-trips, max angular error {max_err:.2e}",
                   ok,
                   f"worst: in={worst[0]}, out={worst[1]}" if not ok else "")


# ---------------------------------------------------------------------------
# Test 4: Round-trip Quat -> Euler -> Quat (off singularity)
# ---------------------------------------------------------------------------
def test_roundtrip_quat():
    print("\nTest 4: round-trip Quat -> Euler -> Quat")
    rng = np.random.default_rng(0xBEEF)
    N = 1000

    max_err = 0.0
    worst = None
    accepted = 0
    for _ in range(N):
        # Sample a uniform unit quaternion (Marsaglia / Shoemake)
        u1, u2, u3 = rng.uniform(0, 1, 3)
        q = np.array([
            np.sqrt(1 - u1) * np.sin(2*np.pi*u2),
            np.sqrt(1 - u1) * np.cos(2*np.pi*u2),
            np.sqrt(u1)     * np.sin(2*np.pi*u3),
            np.sqrt(u1)     * np.cos(2*np.pi*u3),
        ])
        # Skip samples near gimbal lock where theta is close to +/- pi/2
        if abs(2*(q[0]*q[2] - q[1]*q[3])) > 1 - 1e-4:
            continue
        accepted += 1
        q2 = Euler_to_Quat(Quat_to_Euler(q))
        err = min(np.linalg.norm(q - q2), np.linalg.norm(q + q2))
        if err > max_err:
            max_err, worst = err, (q, q2)

    ok = max_err <= ATOL_RAND
    return _report(f"{accepted} random round-trips, max quaternion error {max_err:.2e}",
                   ok,
                   f"worst: q={worst[0]}, q2={worst[1]}" if not ok else "")


# ---------------------------------------------------------------------------
# Test 5: Cross-check via CTM (Euler -> Quat -> CTM == Euler -> CTM)
# ---------------------------------------------------------------------------
def test_cross_ctm_from_euler():
    print("\nTest 5: Euler -> Quat -> CTM matches direct Euler -> CTM")
    rng = np.random.default_rng(0xDEAD)
    N = 500

    max_err = 0.0
    worst = None
    for _ in range(N):
        e = np.array([rng.uniform(-np.pi, np.pi),
                      rng.uniform(-np.pi/2 + SING_EPS, np.pi/2 - SING_EPS),
                      rng.uniform(-np.pi, np.pi)])
        C_via_q = Quat_to_CTM(Euler_to_Quat(e))
        C_direct = Euler_to_CTM(e)
        err = np.max(np.abs(C_via_q - C_direct))
        if err > max_err:
            max_err, worst = err, (e, C_via_q, C_direct)

    ok = max_err <= ATOL_LOOSE
    return _report(f"{N} cases, max |dC| = {max_err:.2e}",
                   ok,
                   f"worst Euler {worst[0]}" if not ok else "")


# ---------------------------------------------------------------------------
# Test 6: Cross-check via CTM (Quat -> Euler -> CTM == Quat -> CTM)
# ---------------------------------------------------------------------------
def test_cross_ctm_from_quat():
    print("\nTest 6: Quat -> Euler -> CTM matches direct Quat -> CTM")
    rng = np.random.default_rng(0xFADE)
    N = 500

    max_err = 0.0
    accepted = 0
    worst = None
    for _ in range(N):
        u1, u2, u3 = rng.uniform(0, 1, 3)
        q = np.array([
            np.sqrt(1 - u1) * np.sin(2*np.pi*u2),
            np.sqrt(1 - u1) * np.cos(2*np.pi*u2),
            np.sqrt(u1)     * np.sin(2*np.pi*u3),
            np.sqrt(u1)     * np.cos(2*np.pi*u3),
        ])
        if abs(2*(q[0]*q[2] - q[1]*q[3])) > 1 - 1e-4:
            continue
        accepted += 1

        C_direct = Quat_to_CTM(q)
        C_via_e  = Euler_to_CTM(Quat_to_Euler(q))
        err = np.max(np.abs(C_via_e - C_direct))
        if err > max_err:
            max_err, worst = err, (q, C_direct, C_via_e)

    ok = max_err <= ATOL_LOOSE
    return _report(f"{accepted} cases, max |dC| = {max_err:.2e}",
                   ok,
                   f"worst quat {worst[0]}" if not ok else "")


# ---------------------------------------------------------------------------
# Test 7: Composed cross-check (Euler -> Quat then Quat -> Euler via CTM)
# ---------------------------------------------------------------------------
def test_quat_consistency_with_ctm_path():
    print("\nTest 7: Euler_to_Quat agrees with CTM_to_Quat o Euler_to_CTM")
    rng = np.random.default_rng(0x1234)
    N = 500

    max_err = 0.0
    worst = None
    for _ in range(N):
        e = np.array([rng.uniform(-np.pi, np.pi),
                      rng.uniform(-np.pi/2 + SING_EPS, np.pi/2 - SING_EPS),
                      rng.uniform(-np.pi, np.pi)])
        q_direct = Euler_to_Quat(e)
        q_via_C  = CTM_to_Quat(Euler_to_CTM(e))
        err = min(np.linalg.norm(q_direct - q_via_C),
                  np.linalg.norm(q_direct + q_via_C))
        if err > max_err:
            max_err, worst = err, (e, q_direct, q_via_C)

    ok = max_err <= ATOL_RAND
    return _report(f"{N} cases, max |dq| = {max_err:.2e}",
                   ok,
                   f"worst Euler {worst[0]}" if not ok else "")


# ---------------------------------------------------------------------------
# Test 8: Gimbal-lock behavior (theta = +/- pi/2)
# ---------------------------------------------------------------------------
def test_gimbal_lock():
    """At theta = +/- pi/2 the Groves arctan2 formulas in eq. (2.37) reduce
    to atan2(0, 0) for both phi and psi (verifiable algebraically: with the
    half-angle quaternion expansion the numerators and denominators all
    collapse to zero). In exact arithmetic phi and psi are therefore
    undefined; in floating-point they are whatever round-off produces.
    Recovering the determined combination (phi -/+ psi) would require a
    separate gimbal-lock branch, which Groves' published formulas do not
    include.

    So at the singularity we only test the invariants the formulas DO
    provide:
      - the function does not blow up (no NaN/Inf),
      - theta is recovered correctly,
      - the output is in the documented range.

    Behavior strictly *between* the singularities (incl. very close to
    them) is already covered by Tests 3-7.
    """
    print("\nTest 8: gimbal-lock graceful behavior (theta = +/- pi/2)")
    all_ok = True

    for sign in (+1, -1):
        e_in = np.array([0.3, sign * np.pi / 2, 0.7])
        q     = Euler_to_Quat(e_in)
        e_out = Quat_to_Euler(q)

        ok_finite = np.all(np.isfinite(e_out))
        all_ok &= _report(f"theta={sign}pi/2: output is finite (no NaN/Inf)",
                          ok_finite, f"got {e_out}")

        ok_theta = abs(e_out[1] - sign * np.pi / 2) <= ATOL_LOOSE
        all_ok &= _report(f"theta={sign}pi/2: theta recovered exactly",
                          ok_theta, f"got theta={e_out[1]}")

        ok_range = (-np.pi - 1e-12 <= e_out[0] <= np.pi + 1e-12 and
                    -np.pi - 1e-12 <= e_out[2] <= np.pi + 1e-12)
        all_ok &= _report(f"theta={sign}pi/2: phi, psi within (-pi, pi]",
                          ok_range, f"got phi={e_out[0]}, psi={e_out[2]}")

    # Just-off-singularity behavior: at theta = pi/2 - 1e-5 everything
    # should still round-trip cleanly.
    for sign in (+1, -1):
        eps = 1e-5
        e_in = np.array([0.3, sign * (np.pi / 2 - eps), 0.7])
        e_out = Quat_to_Euler(Euler_to_Quat(e_in))
        d = np.array([_wrap_pi(e_in[0] - e_out[0]),
                      e_in[1] - e_out[1],
                      _wrap_pi(e_in[2] - e_out[2])])
        err = np.max(np.abs(d))
        # Near (but not at) the singularity, ill-conditioning amplifies
        # round-off by ~1/eps, so we use a looser tolerance here.
        tol = 1e-7
        all_ok &= _report(f"theta=sign*(pi/2 - {eps:g}): round-trip "
                          f"max err {err:.2e}",
                          err <= tol)

    return all_ok


# ---------------------------------------------------------------------------
# Test 9: Speed (single-call latency, since both fns are scalar-valued)
# ---------------------------------------------------------------------------
def test_speed():
    print("\nTest 9: timing")
    rng = np.random.default_rng(0x5A5A)
    N = 20000

    eulers = np.stack([
        rng.uniform(-np.pi, np.pi, N),
        rng.uniform(-np.pi/2 + SING_EPS, np.pi/2 - SING_EPS, N),
        rng.uniform(-np.pi, np.pi, N),
    ], axis=1)

    t0 = time.perf_counter()
    quats = np.empty((N, 4))
    for i in range(N):
        quats[i] = Euler_to_Quat(eulers[i])
    t1 = time.perf_counter()
    print(f"  Euler_to_Quat: {N} calls in {t1-t0:.3f}s  ({1e6*(t1-t0)/N:.1f} us/call)")

    t0 = time.perf_counter()
    e_out = np.empty((N, 3))
    for i in range(N):
        e_out[i] = Quat_to_Euler(quats[i])
    t1 = time.perf_counter()
    print(f"  Quat_to_Euler: {N} calls in {t1-t0:.3f}s  ({1e6*(t1-t0)/N:.1f} us/call)")

    # Sanity: round-trip should still match within tolerance
    diffs = np.abs(np.stack([
        _wrap_pi(eulers[:, 0] - e_out[:, 0]),
        eulers[:, 1] - e_out[:, 1],
        _wrap_pi(eulers[:, 2] - e_out[:, 2]),
    ], axis=1))
    return _report(f"bulk round-trip max error {diffs.max():.2e}",
                   diffs.max() <= ATOL_RAND)


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print(" Tests: Quat_to_Euler / Euler_to_Quat (Groves convention)")
    print("=" * 70)

    results = [
        ("identity",                test_identity()),
        ("single-axis",             test_single_axis()),
        ("roundtrip Euler",         test_roundtrip_euler()),
        ("roundtrip Quat",          test_roundtrip_quat()),
        ("cross-check via CTM (E)", test_cross_ctm_from_euler()),
        ("cross-check via CTM (Q)", test_cross_ctm_from_quat()),
        ("CTM path consistency",    test_quat_consistency_with_ctm_path()),
        ("gimbal lock",             test_gimbal_lock()),
        ("speed",                   test_speed()),
    ]

    print("\n" + "=" * 70)
    print(" Summary")
    print("=" * 70)
    n_pass = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n  {n_pass}/{len(results)} test groups passed.")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())