"""
LIFE mission - Delta-v to sustain the rotating collector formation
==================================================================

The LIFE (Large Interferometer For Exoplanets) array is a free-flying
formation of 1 central beam-combiner spacecraft + 4 collector spacecraft at
the corners of a rectangle. The rectangle has a FIXED ASPECT RATIO: the long
baseline is `baseline_ratio` times the short one (default 1:6). The whole
formation rotates about the line-of-sight (LOS) axis through the rectangle
centre to modulate the exoplanet signal.

PHYSICS
-------
To follow a circular path of radius r at angular rate omega, each collector
must be pushed continuously toward the centre with a centripetal acceleration
a_c = omega**2 * r. In space this is supplied by thrust, so the propulsive
velocity increment accumulated over a continuous-rotation mission of length T
is the time integral of that (constant) acceleration:

        delta_v = a_c * T = omega**2 * r * T          [m/s]   (per collector)

All four corners are equidistant from the centre, so r is the half-diagonal
and ALL FOUR COLLECTORS NEED THE SAME delta-v. For a short baseline s and a
long baseline = ratio * s:

        r = 0.5 * sqrt(s**2 + (ratio * s)**2) = 0.5 * s * sqrt(1 + ratio**2)

The combiner is on the rotation axis (r = 0) -> no formation delta-v, so only
the collectors are simulated.

MASS & THRUST FORCE
-------------------
delta-v is INDEPENDENT of mass (it cancels: F = m*a, dv = F*T/m = a*T). The
mass only sets the *thrust force* the collectors must exert,
F = m*omega**2*r = (m / T) * delta_v. Because that is a simple linear rescale
of delta-v (independent of baseline), the required force is shown on a shared
secondary right-hand axis valid for every curve at once.
"""

from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt

# --- rotation-rate unit -> rad/s ------------------------------------------
_OMEGA_TO_RADS = {
    "rad/s":   1.0,
    "deg/s":   np.pi / 180.0,
    "rpm":     2.0 * np.pi / 60.0,
    "rev/hr":  2.0 * np.pi / 3600.0,
    "rev/day": 2.0 * np.pi / 86400.0,
}
SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0


# ============================== INPUTS ====================================
@dataclass
class LifeRotationConfig:
    collector_mass_kg: float                 # mass of ONE collector [kg]
    baselines_short_m: list                  # LIST of short baselines [m]
    baseline_ratio: float = 6.0              # long = ratio * short  (1:6)
    mission_time_years: float = 5.0          # continuous-rotation duration [yr]
    omega_min: float = 0.5                   # min rotation rate (omega_unit)
    omega_max: float = 5.0                   # max rotation rate (omega_unit)
    n_points: int = 300                      # number of rates to probe
    omega_unit: str = "rev/day"              # unit for the rates & x-axis

    def radius_m(self, short_baseline: float) -> float:
        """Half-diagonal (= collector orbit radius) for a given short side."""
        long = self.baseline_ratio * short_baseline
        return 0.5 * np.sqrt(short_baseline**2 + long**2)


# ============================ SIMULATION ==================================
def simulate(cfg: LifeRotationConfig):
    """Return (omega_user, results) where results is a list of per-baseline
    dicts: {short, long, radius, delta_v[array]}."""
    if cfg.omega_unit not in _OMEGA_TO_RADS:
        raise ValueError(f"omega_unit must be one of {list(_OMEGA_TO_RADS)}")

    factor = _OMEGA_TO_RADS[cfg.omega_unit]
    T = cfg.mission_time_years * SECONDS_PER_YEAR
    omega_user = np.linspace(cfg.omega_min, cfg.omega_max, cfg.n_points)
    omega_rads = omega_user * factor

    results = []
    for s in sorted(cfg.baselines_short_m):
        r = cfg.radius_m(s)
        delta_v = omega_rads**2 * r * T               # per collector [m/s]
        results.append(dict(short=s, long=cfg.baseline_ratio * s,
                            radius=r, delta_v=delta_v))
    return omega_user, results


# ============================== PLOT ======================================
def plot(cfg: LifeRotationConfig, save_path: str | None = None):
    omega_user, results = simulate(cfg)
    factor = _OMEGA_TO_RADS[cfg.omega_unit]
    T_sec = cfg.mission_time_years * SECONDS_PER_YEAR

    fig, ax = plt.subplots(figsize=(10.8, 6.8))
    fig.patch.set_facecolor("white")

    # one colour per baseline
    n = len(results)
    cmap = plt.get_cmap("plasma")
    colours = [cmap(x) for x in np.linspace(0.10, 0.80, n)]

    for res, c in zip(results, colours):
        ax.plot(omega_user, res["delta_v"], color=c, lw=2.4, zorder=3,
                label=f"{res['short']:g} x {res['long']:g} m   "
                      f"(r = {res['radius']:.0f} m)")

    ax.set_xlabel(f"Formation rotation rate about LOS axis  [{cfg.omega_unit}]",
                  fontsize=12)
    ax.set_ylabel(r"Propulsive $\Delta v$ per collector  [m/s]", fontsize=12)
    ax.set_title("LIFE collector formation: $\\Delta v$ to sustain rotation\n"
                 "(continuous thrust providing centripetal acceleration "
                 "$\\omega^2 r$ over the mission)",
                 fontsize=13, fontweight="bold", pad=14)
    ax.grid(True, which="major", ls="-", lw=0.6, alpha=0.35)
    ax.grid(True, which="minor", ls=":", lw=0.4, alpha=0.25)
    ax.minorticks_on()
    ax.set_xlim(omega_user.min(), omega_user.max())
    ax.set_ylim(bottom=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # secondary TOP axis: corresponding rotation PERIOD in hours
    def to_period(o):
        o = np.asarray(o, dtype=float)
        return np.divide(2 * np.pi, o * factor,
                         out=np.full_like(o, np.inf), where=o > 0) / 3600.0

    def to_omega(p):
        p = np.asarray(p, dtype=float)
        return np.divide(2 * np.pi, p * 3600.0 * factor,
                         out=np.full_like(p, np.inf), where=p > 0)

    secax = ax.secondary_xaxis("top", functions=(to_period, to_omega))
    secax.set_xlabel("Rotation period  [hours]", fontsize=11, color="#555")
    secax.tick_params(colors="#555")
    p_lo, p_hi = to_period(omega_user.max()), to_period(omega_user.min())
    nice = np.array([4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 36, 48, 72])
    secax.set_xticks(nice[(nice >= p_lo) & (nice <= p_hi)])

    # secondary RIGHT axis: required thrust FORCE per collector [N] (black).
    # F = m * a_c = (m / T) * delta_v -> linear rescale, same for every curve.
    k = cfg.collector_mass_kg / T_sec        # N per (m/s) of delta-v

    def dv_to_force(dv):
        return np.asarray(dv, dtype=float) * k

    def force_to_dv(F):
        return np.asarray(F, dtype=float) / k

    secay = ax.secondary_yaxis("right", functions=(dv_to_force, force_to_dv))
    secay.set_ylabel("Required thrust force per collector  [N]",
                     fontsize=11, color="black")
    secay.tick_params(colors="black")

    # parameter box (baseline-specific numbers live in the legend)
    info = (
        "INPUTS\n"
        f"  Collector mass   : {cfg.collector_mass_kg:,.0f} kg (each)\n"
        f"  Baseline ratio   : 1 : {cfg.baseline_ratio:g}\n"
        f"  Mission time     : {cfg.mission_time_years:g} yr (continuous)\n"
        f"  Rate range       : {cfg.omega_min:g}-{cfg.omega_max:g} {cfg.omega_unit}\n"
        f"  Points evaluated : {cfg.n_points}\n"
        "MODEL\n"
        "  dv = omega^2 * r * T   (per collector; all 4 identical)\n"
        "  F  = (m / T) * dv      (right axis; uses mass)"
    )
    ax.text(0.025, 0.975, info, transform=ax.transAxes, va="top", ha="left",
            fontsize=8.6, family="monospace",
            bbox=dict(boxstyle="round,pad=0.6", fc="#f4f7fb",
                      ec="#1f4e79", alpha=0.95))

    ax.legend(title=f"Rectangle baseline (short x long, ratio 1:{cfg.baseline_ratio:g})",
              loc="center left", fontsize=9, title_fontsize=9.5,
              framealpha=0.95, edgecolor="#888")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"saved -> {save_path}")
    return fig


# ============================== MAIN ======================================
if __name__ == "__main__":
    cfg = LifeRotationConfig(
        collector_mass_kg=3000.0,                 # mass of each collector [kg]
        baselines_short_m=[10, 50, 100],    # SHORT baselines to probe [m]
        baseline_ratio=6.0,                       # long = 6 x short  (1:6)
        mission_time_years=5.0,                   # years of continuous rotation
        omega_min=0.5,                            # min rotation rate
        omega_max=4.0,                            # max rotation rate
        n_points=300,                             # rates evaluated (linear)
        omega_unit="rev/day",                     # rad/s|deg/s|rpm|rev/hr|rev/day
    )

    omega_user, results = simulate(cfg)
    for res in results:
        print(f"short={res['short']:>5g} m  long={res['long']:>5g} m  "
              f"r={res['radius']:7.1f} m  |  "
              f"dv(min)={res['delta_v'][0]:8.1f} m/s  "
              f"dv(max)={res['delta_v'][-1]:9.1f} m/s")

    plot(cfg, save_path="life_rotation_deltav.png")
    plt.show()