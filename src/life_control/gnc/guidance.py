import numpy as np

# SKELETON CLASS
# class Guidance:
#     """Guidance Class"""

#     def __init__(self, dim_y):
#         # TODO
#         self.dim_y = dim_y

#     def reference(self, et):
#         # TODO
#         return np.zeros(self.dim_y)




class Guidance:
    """Guidance: builds the reference state vector y_ref for the controller.

    The only active reference for now is a circular trajectory for the FIRST
    follower spacecraft, given as relative position/velocity (delta_r, delta_v)
    w.r.t. the leader. The circle lies in the ICRF XY-plane. Every other entry
    of y_ref is left at zero.
    """

    SEC_PER_DAY = 86400.0

    def __init__(self, dim_y, dim_f1=14, radius=0.1, omega_rot_per_day=1.0):
        """
        Parameters
        ----------
        dim_y : int
            Dimension of the full measurement/state vector (= 14 * n).
        dim_f1 : int
            Dimension of the state space for 1 spacecraft.
        radius : float
            Radius of the follower's circular orbit about the leader [km].
        omega_rot_per_day : float
            Orbit angular rate [rotations / day], e.g. 1.0 = one loop per day.
        """
        self.dim_y  = dim_y
        self.radius = radius

        # rotations/day -> rad/s
        self.omega = omega_rot_per_day * 2.0 * np.pi / self.SEC_PER_DAY

        # start index of the first follower's block (spacecraft #2)
        self.f1 = dim_f1

        # accumulated guidance time [s]
        self.t = 0.0

    def reference(self, et, dt):
        """Return the reference state vector y_ref for the current step.

        Parameters
        ----------
        et : float
            Ephemeris time (unused for now, kept for interface compatibility).
        dt : float
            Time increment to advance the reference by [s].
        """
        y_ref = np.zeros(self.dim_y)

        theta = self.omega * self.t
        R, w  = self.radius, self.omega

        i = self.f1
        # relative position delta_r  (circle in XY-plane)
        y_ref[i + 0] = R * np.cos(theta)
        y_ref[i + 1] = R * np.sin(theta)
        y_ref[i + 2] = 0.0
        # relative velocity delta_v = d(delta_r)/dt
        y_ref[i + 3] = -R * w * np.sin(theta)
        y_ref[i + 4] =  R * w * np.cos(theta)
        y_ref[i + 5] = 0.0

        # advance internal clock
        self.t += dt
        return y_ref