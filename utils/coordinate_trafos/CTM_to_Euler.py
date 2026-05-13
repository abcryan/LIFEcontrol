import numpy as np


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
