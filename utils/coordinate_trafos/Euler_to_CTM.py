import numpy as np


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
