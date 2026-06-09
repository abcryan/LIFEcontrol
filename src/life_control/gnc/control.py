import numpy as np

# Skeleton Class
# class Controller:
#     """Controller Class"""

#     def __init__(self, u_dim):
#         self.u_dim = u_dim

#     def compute(self, y_hat, y_ref, t):
#         # 1. Build OCP over [t, t + N*dt] using y_hat as x0 and y_ref as target.
#         # 2. Solve (e.g. CasADi / acados / cvxpy).
#         # 3. Cache the tail as warm-start for next step.
#         # 4. Return u_0.
#         return np.zeros(self.u_dim)



from life_control.plant_model.thrusters       import B_F, N_THRUSTERS
from life_control.utils.coordinate_trafos     import quat_to_CTM
# ^^ adjust path if coordinate_trafos lives elsewhere in your tree

class Controller:
    """Translation PID for follower-1.

    Pipeline inside compute():
      1. PID on (δr, δv) → desired force in ICRF.
      2. Rotate ICRF → body using follower-1's quaternion q_I^B.
      3. Allocate body-frame force to 20 non-negative thruster magnitudes
         via pseudoinverse of the thruster geometry (F^B = -B_F @ T).
      4. Assemble a u_dim vector: zeros for the leader and all other
         followers, allocated thrusts in follower-1's slot.
    """

    def __init__(self, kp, ki, kd, n_sc, T_max, dim_x_sc=14, dim_u_sc=20):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.n_sc, self.dim_x_sc, self.dim_u_sc = n_sc, dim_x_sc, dim_u_sc
        self.T_max    = T_max
        self.f1_x     = dim_x_sc                  # follower-1 state block start
        self.f1_u     = dim_u_sc                  # follower-1 control block start
        self.u_dim    = n_sc * dim_u_sc
        self.integral = np.zeros(3)

        # Pre-compute allocation pseudoinverse:
        #   F^B = -B_F @ T   →   T = -pinv(B_F) @ F^B
        self._alloc_pinv = -np.linalg.pinv(B_F)   # (20, 3)

    def compute(self, y_hat, y_ref, et, dt):
        ix = self.f1_x

        # 1. PID on δr, δv (ICRF)
        e_r = y_ref[ix:ix+3]   - y_hat[ix:ix+3]
        e_v = y_ref[ix+3:ix+6] - y_hat[ix+3:ix+6]
        self.integral += e_r * dt
        F_icrf = self.kp * e_r + self.ki * self.integral + self.kd * e_v   # [N]

        # 2. Rotate ICRF → body using q_I^B
        F_body = self._rotate_to_body(F_icrf, y_hat[ix+6:ix+10])

        # 3. Allocate body-frame force to 20 thrusters
        T = self._allocate(F_body)

        # 4. Pack into full u vector (only follower-1 actuated)
        u = np.zeros(self.u_dim)
        u[self.f1_u : self.f1_u + self.dim_u_sc] = T
        return u

    # ── helpers ────────────────────────────────────────────────────────
    def _rotate_to_body(self, F_icrf, q):
        """ICRF force → body frame using q_I^B (scalar-first, ICRF→body)."""
        return quat_to_CTM(q) @ F_icrf

    def _allocate(self, F_body):
        """Body force → 20 non-negative thruster magnitudes [N].

        Minimum-norm solution to F^B = -B_F @ T, then clipped to [0, T_max].
        Clipping introduces parasitic torques and a small force residual;
        refine later with a constrained LP/QP if needed.
        """
        T = self._alloc_pinv @ F_body
        return np.clip(T, 0.0, self.T_max)