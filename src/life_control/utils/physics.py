import numpy as np


# Per Unit Mass Inertia Tensor of a homogenous cylinder (z = symmetry axis).
def J_cylinder_unit(r_cylinder, h_cylinder) -> np.ndarray:
    unit_J_cylinder         = np.zeros((3, 3))
    unit_J_cylinder[0, 0]   = (1/12) * (3 * r_cylinder**2 + h_cylinder**2)
    unit_J_cylinder[1, 1]   = unit_J_cylinder[0, 0]
    unit_J_cylinder[2, 2]   = (1/2) * r_cylinder**2
    return unit_J_cylinder


# Per Unit Mass Inertia Tensor of a homogenous ring cylinder (z = symmetry axis).
def J_ring_unit(r_in, r_out, h_ring) -> np.ndarray:
    unit_J_ring         = np.zeros((3, 3))
    unit_J_ring[0, 0]   = (1/12) * (3 * (r_in**2 + r_out**2) + h_ring**2)
    unit_J_ring[1, 1]   = unit_J_ring[0, 0]
    unit_J_ring[2, 2]   = (1/2)  * (r_in**2 + r_out**2)
    return unit_J_ring

