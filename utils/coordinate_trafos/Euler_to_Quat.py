import numpy as np


def euler_to_quat(euler: np.ndarray) -> np.ndarray:
    """Convert Groves Euler angles to attitude quaternion q_alpha^beta.

    Implements eq. (2.38) from Groves, "Principles of GNSS, Inertial, and
    Multisensor Integrated Navigation Systems," 2nd ed.

    Euler angles follow the Groves convention: (phi, theta, psi) = (roll,
    pitch, yaw), representing a ZYX (yaw-pitch-roll) sequence from frame
    alpha to frame beta.

    Args:
        euler: (3,) array [phi, theta, psi] in radians (alpha -> beta)

    Returns:
        q^beta_alpha: (4,) quaternion [q0, q1, q2, q3] (scalar-first,
            alpha -> beta), normalized to unit norm.
    """

    # Ensure input is a 3-element array of floats
    euler = np.asarray(euler, dtype=float).reshape(3)
    phi, theta, psi = euler[0], euler[1], euler[2]

    # Half-angle sines and cosines (computed once each)
    cphi,   sphi   = np.cos(0.5 * phi),   np.sin(0.5 * phi)
    cth,    sth    = np.cos(0.5 * theta), np.sin(0.5 * theta)
    cpsi,   spsi   = np.cos(0.5 * psi),   np.sin(0.5 * psi)

    q0 = cphi * cth * cpsi + sphi * sth * spsi
    q1 = sphi * cth * cpsi - cphi * sth * spsi
    q2 = cphi * sth * cpsi + sphi * cth * spsi
    q3 = cphi * cth * spsi - sphi * sth * cpsi

    q = np.array([q0, q1, q2, q3])

    # Normalize quaternion to ensure unit norm (important for numerical stability)
    q = q / np.linalg.norm(q)

    return q