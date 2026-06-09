from dataclasses import dataclass, field
import numpy as np

from life_control.utils.physics import J_cylinder_unit, J_ring_unit


@dataclass(frozen=True)
class PhysicsFlags:
    """
    Toggleable physics terms for the truth model.

    Each flag corresponds to one acceleration or torque term in the dynamics.
    When False, the corresponding `acc_*` / `attitude_tau_*` method returns
    np.zeros(3) regardless of state. This lets you:

      - Compare contributions of different perturbations (run with one term
        on, all others off, and integrate to see the effect on the orbit).
      - Profile the computational cost of each term cleanly: the architecture
        and dispatch cost is the same in every run, so timing differences
        between flag configurations reflect the cost of the physics only.
      - Reproduce results: the flag configuration is part of the run config
        and can be stamped into outputs.

    Defaults:
        Everything that is currently *implemented* defaults to True.
        Terms that are stubs (ion, process noise) default to False.
        ISC gravity defaults to True since we just implemented it.

    Note on `a_ctrl` / `tau_ctrl`:
        Control terms are NOT flag-gated. The actuator output is part of
        the input `u`, not the dynamics. To "disable" control, zero the
        control vector in your main loop. Flag-gating control would conflate
        "we don't model thrusters" with "we don't fire thrusters this step",
        which are different things.
    """

    # ── Translational acceleration terms ─────────────────────────────
    grav_nbody:    bool = True    # N-body gravity (Sun, planets, Moon)
    srp:           bool = True    # solar radiation pressure (cannonball)
    grav_isc:      bool = True    # inter-spacecraft gravity

    # ── Rotational torque terms ──────────────────────────────────────
    tau_srp:       bool = True    # SRP-induced torque 
    tau_grav:      bool = True    # Torque induced by differential gravity

    # ── Rotational dynamics structural terms ─────────────────────────
    # These are not "perturbations" — they're parts of Euler's equation.
    # Off by default = False would give nonsensical rotational dynamics,
    # so they default True. Exposing them lets you test e.g. "what happens
    # if I treat J as constant?" or "what if I drop gyroscopic coupling?"
    gyro_coupling: bool = True    # -ω × (Jω) term
    j_dot_term:    bool = True    # -J̇ω term (only nonzero during burn)

    # ── Mass dynamics  term ─────────────────────────────────────────
    mass_change: bool = True

    def summary(self) -> str:
        """One-line printable summary, useful for run headers and logs."""
        on  = [name for name, val in self.__dict__.items() if val]
        off = [name for name, val in self.__dict__.items() if not val]
        return f"PhysicsFlags(ON: {on};  OFF: {off})"
    



@dataclass(frozen=True)
class Parameters:

    # ── Spacecraft masses [kg] ──────────────────────────────────
    # Mass decomposition per spacecraft:
    #     m_total(t) = m_cyl + m_ring_dry + m_prop(t)
    #     m_total(0) = m_init
    # m_cyl       : inner solid cylinder, structural, constant
    # m_ring_dry  : outer ring structure (walls, tanks, plumbing), constant
    # m_prop      : propellant inside the ring, variable, 0 ≤ m_prop ≤ m_prop_init
    m_init_L        = 3150.0            # leader   total initial mass
    m_init_F        = 3150.0            # follower total initial mass
    m_cylinder_L    = 2000.0            # leader   inner cylinder (constant)
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

    # ── SRP centre-of-pressure offset (body frame, COM -> COP) [m] ──
    # Small COM/COP asymmetry, ~2-3% of the ~5 m body length. z = cylinder axis;
    # axial term dominates (fore/aft mass asymmetry), lateral terms from build
    # tolerance. Replace with mass-properties analysis when available.
    r_CP_L = np.array([ 0.045, -0.030,  0.120])   # |r| ≈ 0.132 m
    r_CP_F = np.array([-0.035,  0.050,  0.105])   # |r| ≈ 0.122 m

    # ── Thrusters ──────────────────────────────────────────────
    l_cube          = 0.735             # side length of the thruster cube sitting at r_out [m]
    ISP             = 3000.0            # Xenon Thruster specific impulse [s]
    T_MAX           = 0.003             # max thrust PER THRUSTER  [N] 
                                        # based on TPF-E 2008 paper --> max. 3.0 mN thrust.

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
        J_i_L   = m_ring_init_L * J_ring_unit(self.r_in, self.r_out, self.h_ring) + J_cyl_L
        J_i_F   = m_ring_init_F * J_ring_unit(self.r_in, self.r_out, self.h_ring) + J_cyl_F
        s("J_init_L", J_i_L)
        s("J_init_F", J_i_F)

        # Cylindrical "cannonball" projected area (rectangle 2r × h)
        s("SRP_area_L", 2 * self.r_cylinder * self.h_cylinder_L)
        s("SRP_area_F", 2 * self.r_cylinder * self.h_cylinder_F)
