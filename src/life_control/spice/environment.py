import spiceypy as spice
import numpy as np
from pathlib import Path


class SpiceEnv:
    """Kernel set + GM cache + ephemeris lookups."""

    def __init__(self, kernels, bodies, frame, abcorr, observer):
        self.kernels  = kernels
        self.bodies   = bodies
        self.frame    = frame
        self.abcorr   = abcorr
        self.observer = observer

        self._furnish()

        self.GM = {}
        for body in self.bodies:
            _, gm_values = spice.bodvrd(body, "GM", 1)
            self.GM[body] = float(gm_values[0])

        self._print_gm()

    def _furnish(self):
        if not all(Path(k).exists() for k in self.kernels):
            raise FileNotFoundError("SPICE kernels not found!")
        for k in self.kernels:
            spice.furnsh(k)

    def _print_gm(self):
        print("Gravitational Parameters (GM) [km^3/s^2]:")
        for body, gm in self.GM.items():
            print(f"  {body:<20s} {gm: .10e}")

    def body_position(self, body: str, et: float) -> np.ndarray:
        return spice.spkpos(body, et, self.frame, self.abcorr, self.observer)[0]

    def str2et(self, s: str) -> float:
        return spice.str2et(s)

    def et2utc(self, et: float, fmt: str = "ISOC", prec: int = 3) -> str:
        return spice.et2utc(et, fmt, prec)