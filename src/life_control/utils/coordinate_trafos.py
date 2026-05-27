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



def quat_to_CTM(q: np.ndarray) -> np.ndarray:
    """Convert Groves attitude quaternion q_alpha^beta to CTM C_alpha^beta.

    Implements eq. (2.34) from Groves, "Principles of GNSS, Inertial, and
    Multisensor Integrated Navigation Systems," 2nd ed.

    Args:
        q^beta_alpha : (4,) quaternion [q0, q1, q2, q3] (scalar-first, alpha -> beta)

    Returns:
        C^beta_alpha: 3x3 coordinate transformation matrix (alpha -> beta)
    """

    # Ensure input is a 4-element array of floats
    q = np.asarray(q, dtype=float).reshape(4)

    # Normalize quaternion to ensure unit norm (important for numerical stability)
    q = q / np.linalg.norm(q)

    # Optional: enforce unique sign convention q0 >= 0
    # if q[0] < 0:
    #     q = -q 

    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]

    C = np.array([
        [q0**2 + q1**2 - q2**2 - q3**2,    2*(q1*q2 + q3*q0),               2*(q1*q3 - q2*q0)            ],
        [2*(q1*q2 - q3*q0),                q0**2 - q1**2 + q2**2 - q3**2,   2*(q2*q3 + q1*q0)            ],
        [2*(q1*q3 + q2*q0),                2*(q2*q3 - q1*q0),               q0**2 - q1**2 - q2**2 + q3**2],
    ])
    return C


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

def euler_to_CTM(eul: np.ndarray) -> np.ndarray:
    """Convert Euler angles to a coordinate transformation matrix.

    Converts Euler angles describing the rotation from frame beta to frame
    alpha into the corresponding CTM, using eq. (2.22) from Groves,
    "Principles of GNSS, Inertial, and Multisensor Integrated Navigation
    Systems," 2nd ed.

    Args:
        eul: (3,) array of Euler angles [roll, pitch, yaw] in radians

    Returns:
        C: 3x3 coordinate transformation matrix (beta -> alpha)
    """
    sin_phi   = np.sin(eul[0])
    cos_phi   = np.cos(eul[0])
    sin_theta = np.sin(eul[1])
    cos_theta = np.cos(eul[1])
    sin_psi   = np.sin(eul[2])
    cos_psi   = np.cos(eul[2])

    C = np.array([
        [ cos_theta * cos_psi,
          cos_theta * sin_psi,
         -sin_theta],
        [-cos_phi * sin_psi + sin_phi * sin_theta * cos_psi,
          cos_phi * cos_psi + sin_phi * sin_theta * sin_psi,
          sin_phi * cos_theta],
        [ sin_phi * sin_psi + cos_phi * sin_theta * cos_psi,
         -sin_phi * cos_psi + cos_phi * sin_theta * sin_psi,
          cos_phi * cos_theta],
    ])
    return C


def CTM_to_quat(C: np.ndarray) -> np.ndarray:
    """Convert a coordinate transformation matrix to a quaternion.

    Converts a CTM describing the transformation from frame alpha to frame
    beta into the corresponding attitude quaternion q_alpha^beta, using
    eqs. (2.34)-(2.36) from Groves, "Principles of GNSS, Inertial, and
    Multisensor Integrated Navigation Systems," 2nd ed.

    Convention: Groves, scalar-first. q = [q0, q1, q2, q3] with q0 the
    scalar part.

    Args:
        C: 3x3 coordinate transformation matrix (alpha -> beta)

    Returns:
        q: (4,) array [q0, q1, q2, q3]
    """

    # Ensure input is a 3x3 array of floats
    C = np.asarray(C, dtype=float).reshape(3, 3)

    t = np.trace(C)
    q = np.zeros(4, dtype=float)

    if t > 0:
        t = np.sqrt(t + 1.0)
        q[0] = 0.5 * t
        t = 0.5 / t
        q[1] = (C[1, 2] - C[2, 1]) * t   
        q[2] = (C[2, 0] - C[0, 2]) * t   
        q[3] = (C[0, 1] - C[1, 0]) * t   

    else:
        i = 0
        if C[1, 1] > C[0, 0]:
            i = 1
        if C[2, 2] > C[i, i]:
            i = 2
        j = (i + 1) % 3
        k = (j + 1) % 3

        t = np.sqrt(C[i, i] - C[j, j] - C[k, k] + 1.0)
        q[i + 1] = 0.5 * t
        t = 0.5 / t
        q[0]     = (C[j, k] - C[k, j]) * t   
        q[j + 1] = (C[j, i] + C[i, j]) * t   
        q[k + 1] = (C[k, i] + C[i, k]) * t   

    # Normalize quaternion to ensure unit norm (important for numerical stability)
    q = q / np.linalg.norm(q)

    # Optional: enforce unique sign convention q0 >= 0
    # if q[0] < 0:
    #     q = -q

    return q

def CTM_to_euler(C: np.ndarray) -> np.ndarray:
    """Convert a coordinate transformation matrix to Euler angles.

    Converts a CTM describing the transformation from frame beta to frame
    alpha into the corresponding Euler angles, using eq. (2.23) from
    Groves, "Principles of GNSS, Inertial, and Multisensor Integrated
    Navigation Systems," 2nd ed.

    Args:
        C: 3x3 coordinate transformation matrix (beta -> alpha)

    Returns:
        eul: (3,) array of Euler angles [roll, pitch, yaw] in radians
    """
    eul = np.zeros(3)
    eul[0] = np.arctan2(C[1, 2], C[2, 2])   # roll
    eul[1] = -np.arcsin(C[0, 2])            # pitch
    eul[2] = np.arctan2(C[0, 1], C[0, 0])   # yaw
    return eul

