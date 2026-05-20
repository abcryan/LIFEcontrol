
import numpy as np
import spiceypy as spice
from pathlib import Path

# File Imports
from config.mission.config import KERNELS, BODIES, FRAME, ABCORR, OBSERVER


##############################################
# Classes
##############################################

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

##############################################
# Methods
##############################################


# Calculates inertia tensor of the spacecraft
def inertia(m_ring, r_in, r_out, h_ring, J_cylinder) -> np.ndarray:

    J_ring      = np.zeros((3,3))

    J_ring[0,0] = (1/12) * m_ring * (3*(r_in**2 + r_out**2) + h_ring**2)
    J_ring[1,1] = J_ring[0,0]
    J_ring[2,2] = (1/2) * m_ring * (r_in**2 + r_out**2)

    # Total inertia is the sum of the ring and cylinder contributions
    J_tot       = J_ring + J_cylinder   

    return J_tot


# Calculates inertia tensor of the cylindrical part of the spacecraft (constant)
def inertia_cylinder(m_cylinder, r_cylinder, h_cylinder) -> np.ndarray:

    J_cylinder = np.zeros((3,3))

    J_cylinder[0,0] = (1/12) * m_cylinder * (3*r_cylinder**2 + h_cylinder**2)
    J_cylinder[1,1] = J_cylinder[0,0]
    J_cylinder[2,2] = (1/2) * m_cylinder * r_cylinder**2

    return J_cylinder

# Calculate ring mass of spacecraft
def mass_ring(m_init, m_cylinder):
    return m_init - m_cylinder


# Initialize the state vector for the leader and followers based on formation geometry and initial conditions
def initialize_state(formation, baseline, att_F, m_init_F, x_init_L, n_sc):
    # NOTE: Relative Dynamics for Followers!

    # Set follower initial states based on formation geometry
    if formation == "square planar" and att_F == "same as leader" and n_sc == 5:

        delta_r0_list = [
            np.array([ baseline,  0.0,         0.0]),   # follower_1: +x
            np.array([-baseline,  0.0,         0.0]),   # follower_2: -x
            np.array([ 0.0,          baseline, 0.0]),   # follower_3: +y
            np.array([ 0.0,         -baseline, 0.0]),   # follower_4: -y
        ]

        q_init_F = np.array([1.0, 0.0, 0.0, 0.0])
        w_init_F = np.array([0.001, 0.01, 0.0])

        x0 = [x_init_L]
        for i, dr0 in enumerate(delta_r0_list):
            dv0 = np.zeros(3)
            x0.append(np.concatenate([dr0, dv0, q_init_F, w_init_F, [m_init_F]]))
        x = np.concatenate(x0)

    else: 
        raise NotImplementedError("Formation geometry or attitude initialization not implemented for the given parameters.")

    return x

##############################################
# Main Function
##############################################

def main():

    # Load SPICE kernels into environment
    env = SpiceEnv(KERNELS, BODIES, FRAME, ABCORR, OBSERVER)

    # Constants
    P_SUN           = 4.53e-6           # solar pressure at 1 AU [N/m^2]
    R_SUN_AU        = 1.495978707e8     # 1 AU in km

    # ---------------------------------------
    # SPACECRAFT Parameters: 
    m_init_L        = 4000.0  # [kg]
    m_init_F        = 3000.0  # [kg]

    m_cylinder_L    = 3000.0  # [kg]
    m_cylinder_F    = 2000.0  # [kg]

    h_cylinder_L    = 5.0     # [m]
    h_cylinder_F    = 4.8     # [m]

    h_ring          = 1.2     # [m]

    r_cylinder      = 2.57    # [m]

    r_in            = 2.7     # [m]
    r_out           = 3.8     # [m]

    J_cylinder_L    = inertia_cylinder(m_cylinder_L, r_cylinder, h_cylinder_L)      # [kg*m^2]
    J_cylinder_F    = inertia_cylinder(m_cylinder_F, r_cylinder, h_cylinder_F)      # [kg*m^2]

    J_init_L        = inertia(mass_ring(m_init_L, m_cylinder_L), r_in, r_out, h_ring, J_cylinder_L)       # [kg*m^2]
    J_init_F        = inertia(mass_ring(m_init_F, m_cylinder_F), r_in, r_out, h_ring, J_cylinder_F)       # [kg*m^2]

    c_reflect       = 1.8    # SRP reflectivity coefficient (~ Webb)

    SRP_area_L      = 2 * r_cylinder * h_cylinder_L     # [m^2] (cylindrical area facing the Sun)
    SRP_area_F      = 2 * r_cylinder * h_cylinder_F     # [m^2] (cylindrical area facing the Sun)


    # ---------------------------------------
    # MODEL Parameters:

    # DEFINE: Number of spacecrafts ( n_sc = (1 Leader) + (n_sc - 1 Follower) )
    n_sc = 5

    # State Space Dimensions: 
    dim_x_sc = 14      # [x, y, z, vx, vy, vz, q1, q2, q3, q4, wx, wy, wz, m]
    dim_u_sc = 6       # [fx, fy, yz, taux, tauy, tauz]

    dim_x = n_sc * dim_x_sc
    dim_u = n_sc * dim_u_sc

    # Initialize the Plant
    # TODO

    # ---------------------------------------
    # SIMULATION Parameters:

    # Initial Epoch (ET)
    et0         = env.str2et("2026-05-12T00:00:00")

    # Leader Initial State (r0, v0, q0, w0, m0)  (Webb-like orbit around L2)
    r_init_L    = np.array([-9.594503991242750e7, -1.098032827423822e8, -4.778858640538428e7])     # [km]
    v_init_L    = np.array([ 2.279303677102495e1, -1.735582913273474e1, -7.640659579588683e0])     # [km/s]
    q_init_L    = np.array([1.0, 0.0, 0.0, 0.0])    # [quaternion] (initially aligned with inertial frame)
    w_init_L    = np.array([0.001, 0.01, 0.0])      # [rad/s] (small initial angular velocity)
    
    x_init_L    = np.concatenate((r_init_L, v_init_L, q_init_L, w_init_L, [m_init_L]))   # [km, km/s, quat, rad/s, kg]

    # Initial Formation 
    formation   = "square planar"
    baseline    = 0.1  # [km] (initial separation between leader and followers)
    att_F       = "same as leader"  # Follower attitude initialization (same as leader for simplicity)

    # Initialize state vector for all spacecraft
    x_init = initialize_state(formation, baseline, att_F, m_init_F, x_init_L, n_sc)

    # Initial Sate Printout
    print("Dimension of x_init:", x_init.shape)
    print(x_init)

if __name__ == "__main__":
    main()








