import numpy as np


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