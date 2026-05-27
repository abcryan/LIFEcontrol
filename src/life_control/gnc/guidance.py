import numpy as np

class Guidance:
    """Guidance Class"""

    def __init__(self, dim_y):
        # TODO
        self.dim_y = dim_y

    def reference(self, et):
        # TODO
        return np.zeros(self.dim_y)