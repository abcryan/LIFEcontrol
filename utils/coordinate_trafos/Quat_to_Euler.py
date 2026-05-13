import numpy as np


def quat_to_euler(q: np.ndarray) -> np.ndarray:
    """Convert Groves attitude quaternion q_alpha^beta to Euler angles.

    Implements eq. (2.37) from Groves, "Principles of GNSS, Inertial, and
    Multisensor Integrated Navigation Systems," 2nd ed.

    Euler angles follow the Groves convention: (phi, theta, psi) = (roll,
    pitch, yaw), representing a ZYX (yaw-pitch-roll) sequence from frame
    alpha to frame beta.

    Args:
        q^beta_alpha: (4,) quaternion [q0, q1, q2, q3] (scalar-first,
            alpha -> beta)

    Returns:
        euler: (3,) array [phi, theta, psi] in radians (alpha -> beta).
            phi   in (-pi, pi]
            theta in [-pi/2, pi/2]
            psi   in (-pi, pi]

    Note:
        At the gimbal-lock singularity theta = +/- pi/2, the (phi, psi)
        decomposition is not unique (only phi -/+ psi is determined). The
        arctan2 formulas in Groves eq. (2.37) return a valid but arbitrary
        split in that case; the rotation itself remains correct via that
        combined angle, but individual phi and psi values should not be
        relied upon near the singularity.
    """

    # Ensure input is a 4-element array of floats
    q = np.asarray(q, dtype=float).reshape(4)

    # Normalize quaternion to ensure unit norm (important for numerical stability)
    q = q / np.linalg.norm(q)

    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]

    # Precompute squared terms (reused below)
    q1_sq = q1 * q1
    q2_sq = q2 * q2
    q3_sq = q3 * q3

    # Argument of arcsin clipped to [-1, 1] to guard against round-off
    # when the true value sits on the singularity (theta = +/- pi/2).
    sin_theta = 2.0 * (q0 * q2 - q1 * q3)
    sin_theta = np.clip(sin_theta, -1.0, 1.0)

    phi   = np.arctan2(2.0 * (q0 * q1 + q2 * q3), 1.0 - 2.0 * q1_sq - 2.0 * q2_sq)
    theta = np.arcsin(sin_theta)
    psi   = np.arctan2(2.0 * (q0 * q3 + q1 * q2), 1.0 - 2.0 * q2_sq - 2.0 * q3_sq)

    return np.array([phi, theta, psi])
