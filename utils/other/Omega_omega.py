import numpy as np

def Omega_omega(omega: np.ndarray) -> np.ndarray:
    """Construct the Omega matrix for quaternion kinematics.

    The Omega matrix is used in the quaternion kinematic equation
    q_dot = 0.5 * Omega(omega) * q, where omega is the angular velocity
    vector.

    Args:
        omega: (3,) array of angular velocity components [omega_x, omega_y, omega_z]

    Returns:
        Omega: 4x4 matrix used in quaternion kinematics
    """
    Omega = np.array([
        [0,         -omega[0], -omega[1], -omega[2]],
        [omega[0],   0,         omega[2], -omega[1]],
        [omega[1],  -omega[2],  0,         omega[0]],
        [omega[2],   omega[1], -omega[0],  0        ]
    ])
    return Omega