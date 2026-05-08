import numpy as np


def Quat_to_CTM(q: np.ndarray) -> np.ndarray:
    """Convert Groves attitude quaternion q_alpha^beta to CTM C_alpha^beta.

    Implements eq. (2.34) from Groves, "Principles of GNSS, Inertial, and
    Multisensor Integrated Navigation Systems," 2nd ed.

    Args:
        q : (4,) quaternion [q0, q1, q2, q3] (scalar-first, alpha -> beta)

    Returns:
        C: 3x3 coordinate transformation matrix (alpha -> beta)
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
