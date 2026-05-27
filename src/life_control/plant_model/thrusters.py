from life_control.plant_model.spacecraft import Parameters
import numpy as np

"""
LIFE Mission — Thruster geometry and control allocation.

Per Section 3.4 of the design document, each spacecraft has 20 thrusters,
arranged as 4 cubic clusters of 5 thrusters located on the outer ring at the
±x and ±y body-frame directions.  Each cube has an inward (centerward) face
with no thruster; the 5 remaining faces each carry one thruster firing
outward along the face normal.

Cube geometry (from the document):
    - cube center on the outer face of the ring, at radial distance
        d_cube_center = r_out + 0.5 * L_CUBE  ≈ 3.82 + 0.3675 = 4.1875  m
      where L_CUBE = 0.735 m is the cube side length.
    - the outer-face thruster sits at radial distance 4.555 m  ( = 4.1875 + 0.3675 )
    - the side and top/bottom thrusters sit at 4.1875 m radial, with an
      additional ±0.3675 m offset along the perpendicular face normal.

Sign convention:
    Each thruster fires its exhaust along its outward-pointing nozzle normal
    n_hat.  The reaction force on the spacecraft is therefore
        f_l^B = -T_l * n_hat_l ,        T_l ≥ 0
    so the body-frame net force and torque from a thrust command vector
    T = (T_1, ..., T_20) are
        F^B = - B_F  @ T ,     B_F[:, l] = n_hat_l
        τ^B = - B_τ @ T ,     B_τ[:, l] = r_l × n_hat_l
    Both B_F and B_τ are 3×20 constants for a given spacecraft.

The geometry is identical for the leader and follower (per the document — the
ring dimensions are shared), so we keep a single allocation that both can
reuse.
"""

# Import necessary parameters: 
param = Parameters()

# ── Geometry constants from Section 3.4 / Figure 4 of the design doc ────────

R_OUT         = param.r_out          # ring outer radius                  [m]
L_CUBE        = param.l_cube         # cube side length                   [m]
HALF_CUBE     = L_CUBE / 2.0         

# Cube center radius — cube is mounted with its inner face flush against the
# ring's outer face, so the cube center sits HALF_CUBE further out.
D_CUBE_CENTER = R_OUT + HALF_CUBE        # = 4.1875 m
# Outer-face thruster — sits on the cube's outermost face, one more
# HALF_CUBE beyond the cube center.
D_OUTER       = D_CUBE_CENTER + HALF_CUBE # = 4.555  m


# ── Build the 20-thruster table in the body frame ───────────────────────────

def _build_thruster_table():
    """
    Build per-thruster (position, outward normal) in the body frame.

    Returns
    -------
    positions : (20, 3) ndarray   — r_l^B   [m]
    normals   : (20, 3) ndarray   — n_hat_l (unit, outward exhaust direction)

    Layout (matches Table 2 of the design document):
        Group  +x:   thrusters  1– 5
        Group  -x:   thrusters  6–10
        Group  +y:   thrusters 11–15
        Group  -y:   thrusters 16–20
    Within each group the 5 thrusters are ordered as
        outer-face, +tangent-face, -tangent-face, +z-face, -z-face
    where the tangent direction is +y for the ±x groups and +x for the ±y
    groups, exactly as printed in Table 2 (so the signs of the side-thruster
    normals are identical for the +x and -x groups — both fire ±y on the
    sides — and likewise the ±y groups both fire ±x on the sides).
    """
    z_hat = np.array([0.0, 0.0, 1.0])

    # (group_outward, tangent_direction)
    groups = [
        ( np.array([+1.0, 0.0, 0.0]),  np.array([0.0, +1.0, 0.0]) ),   # +x, tang = +y
        ( np.array([-1.0, 0.0, 0.0]),  np.array([0.0, +1.0, 0.0]) ),   # -x, tang = +y
        ( np.array([ 0.0,+1.0, 0.0]),  np.array([+1.0, 0.0, 0.0]) ),   # +y, tang = +x
        ( np.array([ 0.0,-1.0, 0.0]),  np.array([+1.0, 0.0, 0.0]) ),   # -y, tang = +x
    ]

    positions = np.zeros((20, 3))
    normals   = np.zeros((20, 3))

    for g_idx, (g_hat, t_hat) in enumerate(groups):
        cube_center = D_CUBE_CENTER * g_hat        # 4.1875 m along group axis
        base        = 5 * g_idx

        # Thruster 1 of group: outer face — sits further out along g_hat
        positions[base + 0] = D_OUTER * g_hat
        normals  [base + 0] = g_hat

        # Thruster 2: +tangent side
        positions[base + 1] = cube_center + HALF_CUBE * t_hat
        normals  [base + 1] = t_hat

        # Thruster 3: -tangent side
        positions[base + 2] = cube_center - HALF_CUBE * t_hat
        normals  [base + 2] = -t_hat

        # Thruster 4: +z top
        positions[base + 3] = cube_center + HALF_CUBE * z_hat
        normals  [base + 3] = z_hat

        # Thruster 5: -z bottom
        positions[base + 4] = cube_center - HALF_CUBE * z_hat
        normals  [base + 4] = -z_hat

    return positions, normals


THRUSTER_POSITIONS, THRUSTER_NORMALS = _build_thruster_table()
N_THRUSTERS = THRUSTER_POSITIONS.shape[0]


# ── Control allocation matrices (constants in the body frame) ───────────────
#
#   F^B = - B_F @ T            B_F[:, l] = n_hat_l        (3 × N_THRUSTERS)
#   τ^B = - B_TAU @ T          B_TAU[:, l] = r_l × n_hat_l (3 × N_THRUSTERS)
#
# Both are precomputed once and reused by every spacecraft, every step.

B_F   = THRUSTER_NORMALS.T                                        # (3, 20)
B_TAU = np.cross(THRUSTER_POSITIONS, THRUSTER_NORMALS).T          # (3, 20)


# ── Allocation API ──────────────────────────────────────────────────────────

def force_torque_body(T_cmd: np.ndarray):
    """
    Map a 20-vector of (non-negative) thrust magnitudes to body-frame force
    and torque using the precomputed allocation matrices.

        F^B = - B_F   @ T          [N]          --> summed over all thrusters
        τ^B = - B_TAU @ T          [N·m]        --> summed over all thrusters

    The caller is responsible for clamping `T_cmd` to [0, T_max] beforehand.
    """

    f_B   = - B_F   @ T_cmd                                       # (3,)
    tau_B = - B_TAU @ T_cmd                                       # (3,)
    return f_B, tau_B


def clamp_thrust(T_cmd: np.ndarray, T_max: float) -> np.ndarray:
    """Project a thrust command into [0, T_max]^N (the physical actuator set)."""
    return np.clip(T_cmd, 0.0, T_max)


def total_propellant_flow(T_cmd: np.ndarray, Isp: float, g0_km: float) -> float:
    """
    Tsiolkovsky propellant flow for a 20-thruster cluster, matching the
    design document eq. (25):

        ṁ = - Σ_l T_l / (Isp · g0)

    Units (km / s / kg system, matching the rest of the code):
        T_l       [kN]   = [kg·km/s²]   (same as |F| since 1 N = 1e-3 kN)
        Isp · g0  [km/s] = [s · km/s²]
        ṁ         [kg/s]
    so we MUST receive T_cmd in kN here (or N converted to kN by the caller).
    """
    return - float(np.sum(T_cmd)) / (Isp * g0_km)


# ── Debug / inspection helpers ──────────────────────────────────────────────

def summarize():
    """Print the thruster table — handy for sanity-checking against Table 2."""
    print(f"Thruster table — {N_THRUSTERS} thrusters")
    print(f"{'idx':>3}  {'group':>5}  {'position [m]':<28} {'direction':<10}")
    group_labels = ['+x', '-x', '+y', '-y']
    for l in range(N_THRUSTERS):
        g = group_labels[l // 5]
        p = THRUSTER_POSITIONS[l]
        n = THRUSTER_NORMALS[l]
        dir_label = ''.join([
            f"{'+' if n[k] > 0.5 else ('-' if n[k] < -0.5 else '')}"
            f"{'xyz'[k] if abs(n[k]) > 0.5 else ''}"
            for k in range(3)
        ])
        print(f"{l+1:>3}  {g:>5}  [{p[0]:+7.4f}, {p[1]:+7.4f}, {p[2]:+7.4f}]   {dir_label}")
    print()
    print("Allocation matrices:")
    print(f"  B_F   shape {B_F.shape},   rank {np.linalg.matrix_rank(B_F)}")
    print(f"  B_TAU shape {B_TAU.shape}, rank {np.linalg.matrix_rank(B_TAU)}")
    print(f"  full stack rank (force+torque controllability): "
          f"{np.linalg.matrix_rank(np.vstack([B_F, B_TAU]))}")


if __name__ == "__main__":
    summarize()