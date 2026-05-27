from dataclasses import dataclass, field
import numpy as np

from life_control.utils.physics import J_cylinder_unit, J_ring_unit

@dataclass(frozen=True)
class Parameters:

    # ── Spacecraft masses [kg] ──────────────────────────────────
    # Mass decomposition per spacecraft:
    #     m_total(t) = m_cyl + m_ring_dry + m_prop(t)
    #     m_total(0) = m_init
    # m_cyl       : inner solid cylinder, structural, constant
    # m_ring_dry  : outer ring structure (walls, tanks, plumbing), constant
    # m_prop      : propellant inside the ring, variable, 0 ≤ m_prop ≤ m_prop_init
    m_init_L        = 4150.0            # leader   total initial mass
    m_init_F        = 3150.0            # follower total initial mass
    m_cylinder_L    = 3000.0            # leader   inner cylinder (constant)
    m_cylinder_F    = 2000.0            # follower inner cylinder (constant)
    m_prop_init_L   = 150.0             # leader   usable propellant
    m_prop_init_F   = 150.0             # follower usable propellant

    # ── Spacecraft off-diagonal inertia contributions ───────────
    # In case you want to include some off-diagonal perturbation via:    eps = off_diag_frac * K[2, 2]
    # Currently applies to L and F equally AND only to cylinder inertia
    # TODO

    # ── Spacecraft geometry [m] ─────────────────────────────────
    h_cylinder_L    = 5.0               # leader   cylinder height
    h_cylinder_F    = 4.8               # follower cylinder height
    h_ring          = 1.2               # ring height (shared)
    r_cylinder      = 2.57              # cylinder radius (shared)
    r_in            = 2.7               # ring inner radius (shared)
    r_out           = 3.8               # ring outer radius (shared)

    # ── Propulsion ──────────────────────────────────────────────
    ISP_L           = 220.0             # leader   specific impulse [s]
    ISP_F           = 220.0             # follower specific impulse [s]
    T_MAX           = 20.0              # max thrust PER THRUSTER  [N]
                                         # (used to saturate the control input
                                         #  in the truth model; matches the
                                         #  20 N upper bound in Malladi et al.
                                         #  Sec. IV; user may tune this freely)

    # ── SRP ─────────────────────────────────────────────────────
    c_reflect       = 1.8               # SRP reflectivity coefficient (Webb-like)

    # ── Derived (computed in __post_init__) ─────────────────────
    J_cylinder_L:   np.ndarray = field(init=False)
    J_cylinder_F:   np.ndarray = field(init=False)
    J_init_L:       np.ndarray = field(init=False)
    J_init_F:       np.ndarray = field(init=False)
    SRP_area_L:     float      = field(init=False)
    SRP_area_F:     float      = field(init=False)
    m_ring_dry_L:   float      = field(init=False)
    m_ring_dry_F:   float      = field(init=False)

    def __post_init__(self):
        # frozen=True forbids normal assignment — bypass via object.__setattr__
        s = lambda k, v: object.__setattr__(self, k, v)

        # ── Inner-cylinder inertias (constant) ──
        J_cyl_L = self.m_cylinder_L * J_cylinder_unit(self.r_cylinder, self.h_cylinder_L)
        J_cyl_F = self.m_cylinder_F * J_cylinder_unit(self.r_cylinder, self.h_cylinder_F)
        s("J_cylinder_L", J_cyl_L)
        s("J_cylinder_F", J_cyl_F)

        # ── Dry ring mass = what's left of m_init after subtracting cylinder
        # and propellant. Must be non-negative.
        m_ring_dry_L = self.m_init_L - self.m_cylinder_L - self.m_prop_init_L
        m_ring_dry_F = self.m_init_F - self.m_cylinder_F - self.m_prop_init_F
        if m_ring_dry_L < 0 or m_ring_dry_F < 0:
            raise ValueError(
                "Negative ring dry mass — check m_init, m_cylinder, m_prop_init."
            )
        s("m_ring_dry_L", m_ring_dry_L)
        s("m_ring_dry_F", m_ring_dry_F)

        # ── Initial inertias (full ring mass = dry + propellant) ──
        m_ring_init_L = m_ring_dry_L + self.m_prop_init_L
        m_ring_init_F = m_ring_dry_F + self.m_prop_init_F
        s("J_init_L", inertia(m_ring_init_L, self.r_in, self.r_out, self.h_ring, J_cyl_L))
        s("J_init_F", inertia(m_ring_init_F, self.r_in, self.r_out, self.h_ring, J_cyl_F))

        # Cylindrical "cannonball" projected area (rectangle 2r × h)
        s("SRP_area_L", 2 * self.r_cylinder * self.h_cylinder_L)
        s("SRP_area_F", 2 * self.r_cylinder * self.h_cylinder_F)



def inertia(m_ring, r_in, r_out, h_ring, J_cylinder) -> np.ndarray:
    """Total J = m_ring * K_ring + J_cylinder."""
    return m_ring * J_ring_unit(r_in, r_out, h_ring) + J_cylinder
