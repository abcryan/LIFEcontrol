import numpy as np

class Controller:
    """Controller Class"""

    def __init__(self, u_dim):
        self.u_dim = u_dim

    def compute(self, y_hat, y_ref, t):
        # 1. Build OCP over [t, t + N*dt] using y_hat as x0 and y_ref as target.
        # 2. Solve (e.g. CasADi / acados / cvxpy).
        # 3. Cache the tail as warm-start for next step.
        # 4. Return u_0.
        return np.zeros(self.u_dim)