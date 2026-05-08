"""
LIFE Mission – High-Fidelity Translational Dynamics (Truth Model)

State:  x = [r; v] ∈ R^6 – position (km) and velocity (km/s) in ICRF J2000,
        origin at solar system barycenter (SSB).
Time:   SPICE Ephemeris Time (ET) = seconds past J2000 TDB.
Units:  km, km/s, s  (native SPICE units).

Precision note: float64 at L2 scale (~1.5e8 km) gives ~33 nm resolution,
sufficient for mm-level relative dynamics. Catastrophic cancellation when
computing inter-spacecraft relative positions from absolute ICRF coordinates
is not an issue until separations fall below ~1 μm – addressed when
multi-spacecraft relative dynamics are added.

Required SPICE kernels (download to ./data/spice_kernels/):
  naif0012.tls   – leapseconds        naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/
  de440.bsp      – planetary ephem (~114 MB)  .../generic_kernels/spk/planets/
  gm_de440.tpc   – GM values matched to DE440  .../generic_kernels/pck/

Library rationale:
  SpiceyPy  – industry-standard ephemeris (ICRF body positions, time conversion,
              and GM values via the gm_de440 PCK – single source of truth).
  DOP853    – 8th-order Dormand-Prince RK; well-tested, deterministic, tight
              error control without the overhead of a full framework like TudatPy.
  CasADi    – reserved for MPC controller (symbolic differentiation, NLP);
              cannot easily host SPICE calls inside its symbolic graph.
"""
import sys
import numpy as np
import spiceypy as spice
from scipy.integrate import solve_ivp
from pathlib import Path

sys.dont_write_bytecode = True

# ── Kernels ───────────────────────────────────────────────────────────────────

KERNELS = [
    "data/spice_kernels/naif0012.tls",
    "data/spice_kernels/de440.bsp",
    "data/spice_kernels/gm_de440.tpc",
]


def _check_kernels_exist() -> bool:
    """Check if all required kernels exist."""
    for kernel_path in KERNELS:
        if not Path(kernel_path).exists():
            return False
    return True

# ── Gravitational parameters [km³/s²] ────────────────────────────────────────
# Values are read once from gm_de440.tpc (consistent with the DE440 ephemeris).
# Planet barycenters are used for the multi-moon outer systems – their
# BODY<id>_GM entries already give the *system* GM. Earth and Moon are kept
# separate because their separation matters at L2.

BODIES: tuple[str, ...] = (
    "SUN",
    "MERCURY BARYCENTER",       # same as MERCURY (no moons)
    "VENUS BARYCENTER",         # same as VENUS (no moons)
    "EARTH",                    
    "MOON",
    "MARS BARYCENTER",
    "JUPITER BARYCENTER",
    "SATURN BARYCENTER",
    "URANUS BARYCENTER",
    "NEPTUNE BARYCENTER",
)

GM: dict[str, float] = {}   # populated by load_kernels() from the PCK


def load_kernels() -> None:
    """Furnish all SPICE kernels and read GM values from the PCK."""
    if not _check_kernels_exist():
        raise FileNotFoundError(
            "\n" + "="*70 + "\n"
            "✗ SPICE kernels not found!\n\n"
            "Required kernels are missing from data/spice_kernels/\n"
            "Download them automatically using:\n\n"
            "  pip install -e .\n"
            "  lifecontrol-setup-kernels\n\n"
            "Or manually download from NAIF servers:\n"
            "  https://naif.jpl.nasa.gov/pub/naif/generic_kernels/\n"
            "="*70
        )
    for k in KERNELS:
        spice.furnsh(k)
    GM.clear()
    for body in BODIES:
        GM[body] = float(spice.bodvrd(body, "GM", 1)[1][0])

# ── Dynamics ──────────────────────────────────────────────────────────────────

def a_grav(r: np.ndarray, et: float) -> np.ndarray:
    """
    N-body gravitational acceleration [km/s²] at ICRF position r [km].

    Implements Eq. (5):  a_grav = Σ_b  –μ_b · (r – r_b) / |r – r_b|³

    Body positions queried from SPICE (SSB-centred, J2000 frame, no aberration).
    """
    a = np.zeros(3)
    for body, mu in GM.items():
        r_b = spice.spkezr(body, et, "J2000", "NONE", "SSB")[0][:3]
        dr  = r - r_b
        a  -= mu * dr / np.linalg.norm(dr) ** 3
    return a


def x_dot(t: float, x: np.ndarray, et0: float, u: np.ndarray) -> np.ndarray:
    """ODE right-hand side: dot x = [v; 
                                     a_grav(r, et0 + t) + a_ctrl  ]."""
    a_ctrl = u  # control acceleration [km/s²]
    v = x[3:]
    a = a_grav(x[:3], et0 + t) + a_ctrl
    return np.concatenate([
        v, 
        a,
    ])

# ── Propagator ────────────────────────────────────────────────────────────────

def propagate(
    r0:     np.ndarray,
    v0:     np.ndarray,
    t_span: tuple[float, float],
    et0:    float,
    u:      np.ndarray | None = None,
    t_eval: np.ndarray | None = None,
    rtol:   float = 1e-12,
    atol:   float = 1e-12,
) -> dict[str, np.ndarray]:
    """
    Propagate spacecraft from (r0 [km], v0 [km/s]) over t_span [s].

    et0     : start epoch as SPICE ET (seconds past J2000 TDB).
    u       : constant control acceleration [km/s²] (default: zeros).
    t_eval  : optional output times within t_span; if None, uses adaptive steps.
    rtol/atol: integrator tolerances (defaults give sub-μm accuracy).

    Returns {'t': (N,), 'r': (N,3) km, 'v': (N,3) km/s}.
    """
    if u is None:
        u = np.zeros(3)
    sol = solve_ivp(
        x_dot, t_span,
        np.concatenate([r0, v0]),
        method       = "DOP853",
        t_eval       = t_eval,
        args         = (et0, u),
        rtol         = rtol,
        atol         = atol,
        dense_output = False,
    )
    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")
    return {"t": sol.t, "r": sol.y[:3].T, "v": sol.y[3:].T}


def step(
    r:    np.ndarray,
    v:    np.ndarray,
    dt:   float,
    et:   float,
    u:    np.ndarray | None = None,
    rtol: float = 1e-12,
    atol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Advance state one control sample of dt [s] from epoch et.

    u       : control acceleration [km/s²], zero-order hold over dt (default: zeros).
    rtol/atol: integrator tolerances (defaults give sub-μm accuracy).

    Returns (r_next [km], v_next [km/s]).
    """
    if u is None:
        u = np.zeros(3)
    sol = solve_ivp(
        x_dot, (0.0, dt),
        np.concatenate([r, v]),
        method       = "DOP853",
        args         = (et, u),
        rtol         = rtol,
        atol         = atol,
        dense_output = False,
    )
    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")
    return sol.y[:3, -1], sol.y[3:, -1]

# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    load_kernels()

    print("GM values loaded from gm_de440.tpc [km³/s²]:")
    for body, mu in GM.items():
        print(f"  {body:<20s} {mu: .10e}")

    # Epoch: 2027-01-01 00:00:00 UTC
    et0 = spice.str2et("2027-01-01T00:00:00")

    # Initial conditions: spacecraft placed at approximate Sun–Earth L2
    # (1.5e6 km beyond Earth along the Sun–Earth line, inheriting Earth's velocity)
    r_earth = spice.spkezr("EARTH", et0, "J2000", "NONE", "SSB")[0]
    r_sun   = spice.spkezr("SUN",   et0, "J2000", "NONE", "SSB")[0]
    earth_hat = (r_earth[:3] - r_sun[:3]) / np.linalg.norm(r_earth[:3] - r_sun[:3])
    r0 = r_earth[:3] + 1.5e6 * earth_hat   # km
    v0 = r_earth[3:].copy()                 # km/s

    print(f"\nEpoch  : {spice.et2utc(et0, 'ISOC', 3)}")
    print(f"r0     : {r0}  km")
    print(f"v0     : {v0}  km/s")

    u = np.zeros(3)   # control acceleration [km/s²] – zero thrust

    # # Full trajectory: 1 day at 60 s output cadence
    # t_eval = np.arange(0.0, 86401.0, 60.0)
    # result = propagate(r0, v0, (0.0, t_eval[-1]), et0, u=u, t_eval=t_eval)

    # dr = np.linalg.norm(result["r"][-1] - result["r"][0])
    # print(f"\nPropagated {len(result['t'])} samples over 1 day")
    # print(f"r(tf)  : {result['r'][-1]}  km")
    # print(f"|Δr|   : {dr:.3f} km")

    # Single control step (example: 1 s sample period)
    r_next, v_next = step(r0, v0, dt=1.0, et=et0, u=u)
    print(f"\n1 s step  Δr = {np.linalg.norm(r_next - r0)*1e6:.3f} mm")
