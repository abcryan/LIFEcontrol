#!/usr/bin/env encoding=utf-8
"""
LIFE Mission — Dynamics Vulnerability & Edge-Case Diagnostic Suite
Tests: Catastrophic Cancellation, SRP Paradox, COM Migration, and Quaternion Ambiguity.
"""

import numpy as np
import sys

# ── Ensure local package resolution ──────────────────────────────────────────
from life_control.plant_model.plant import Plant
from life_control.plant_model.spacecraft import Parameters, PhysicsFlags
from life_control.utils.coordinate_trafos import quat_to_CTM
import life_control.utils.constants as const

# ── Mock Environment class to satisfy Plant's ephemeris interface ────────────
class MockEnv:
    def __init__(self):
        # Earth/Sun standard gravitational parameter reference
        self.GM = {"SUN": 1.32712440018e11}  # km^3/s^2
        
    def body_position(self, body, et):
        # Place the Sun exactly at the origin of the coordinate system
        if body == "SUN":
            return np.zeros(3)
        return np.zeros(3)


def run_diagnostics():
    print("=" * 70)
    print("       LIFE MISSION DYNAMICS SYSTEM VULNERABILITY DIAGNOSTICS       ")
    print("=" * 70)

    # Initialize standard test objects
    env = MockEnv()
    flags = PhysicsFlags(
        grav_nbody=True,
        srp=True,
        grav_isc=True,
        tau_srp=True,
        tau_grav=True,
        gyro_coupling=True,
        j_dot_term=True,
        mass_change=True
    )
    param = Parameters()
    
    # 1 Leader, 1 Follower system
    plant = Plant(env=env, param=param, flags=flags, n_sc=2, dim_x_sc=14, dim_u_sc=20)
    bodies = env.body_position("SUN", 0.0) # cached position map

    # ──────────────────────────────────────────────────────────────────────────
    # TEST 1: Catastrophic Cancellation in Relative Gravity
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[TEST 1] Catastrophic Cancellation in 'acc_grav_rel'")
    print("-" * 70)
    
    # Leader positioned at 1 AU along the X axis
    r_L = np.array([1.5e8, 0.0, 0.0]) 
    snapshot_bodies = {"SUN": np.zeros(3)}
    
    # Define baselines: 100 meters, 10 meters, 1 meter, and 10 centimeters (in km)
    baselines_km = [0.1, 0.01, 0.001, 0.0001]
    
    for b in baselines_km:
        delta_r = np.array([b, 0.0, 0.0]) # Separation along Sun-spacecraft line
        
        # 1. Compute via the floating-point subtraction in the plant
        da_computed = plant.acc_grav_rel(r_L, delta_r, snapshot_bodies)
        
        # 2. Compute the exact analytical linear gravity gradient (Reference)
        # da = - (mu/r^3) * [ delta_r - 3*(u_hat . delta_r)*u_hat ]
        mu = env.GM["SUN"]
        r_norm = np.linalg.norm(r_L)
        u_hat = r_L / r_norm
        da_analytical = -(mu / (r_norm**3)) * (delta_r - 3.0 * np.dot(u_hat, delta_r) * u_hat)
        
        # Evaluate difference
        abs_err = np.linalg.norm(da_computed - da_analytical)
        rel_err = abs_err / np.linalg.norm(da_analytical) if np.linalg.norm(da_analytical) > 0 else 0
        
        print(f"Baseline: {b*1000:6.1f} m | Computed: {da_computed[0]:.12e} km/s²")
        print(f"               | Analytical: {da_analytical[0]:.12e} km/s²")
        print(f"               | Absolute Error: {abs_err:.6e} km/s² | Relative Error: {rel_err:.4e}")
        
    # ──────────────────────────────────────────────────────────────────────────
    # TEST 2: Attitude-Force Paradox in SRP Cannonball Model
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[TEST 2] Attitude-Force Paradox in SRP Model")
    print("-" * 70)
    
    m_total = param.m_init_L
    snapshot_bodies = {"SUN": np.zeros(3)}
    
    # Compute base inertial SRP acceleration vector
    a_srp_I = plant.acc_srp_abs(r_L, m_total, 'L', snapshot_bodies)
    
    # Attitude A: Identity alignment
    q_A = np.array([1.0, 0.0, 0.0, 0.0])
    tau_srp_A = plant.attitude_tau_srp(q_A, a_srp_I, m_total, 'L')
    
    # Attitude B: 90 degree pitch/yaw rotation (swaps projected profile completely)
    # q = [cos(pi/4), 0, sin(pi/4), 0] -> 90 deg rotation about Y axis
    q_B = np.array([0.70710678118, 0.0, 0.70710678118, 0.0])
    tau_srp_B = plant.attitude_tau_srp(q_B, a_srp_I, m_total, 'L')
    
    print(f"Inertial SRP Accel Vector (Attitude A): {a_srp_I} km/s²")
    print(f"Inertial SRP Accel Vector (Attitude B): {a_srp_I} km/s² <--- Unchanged!")
    print(f"Resulting SRP Torque (Attitude A)      : {tau_srp_A} N*m")
    print(f"Resulting SRP Torque (Attitude B)      : {tau_srp_B} N*m")
    print("Observation: The translational force ignores spacecraft cross-sectional rotation,")
    print("             but the attitude system extracts highly specific local torques from it.")

    # ──────────────────────────────────────────────────────────────────────────
    # TEST 3: Center of Mass (COM) Asymmetry vs. Static Moment Arms
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[TEST 3] Rigid COM vs. Actuator Moment Arms")
    print("-" * 70)
    
    J_full = plant._inertia_now(param.m_prop_init_L, 'L')
    J_dry  = plant._inertia_now(0.0, 'L')
    
    print("Inertia Matrix Tensor (Full Fuel 150kg):")
    print(J_full)
    print("Inertia Matrix Tensor (Dry Tanks  0kg):")
    print(J_dry)
    
    # Implied Physical Shift Scenario:
    # If 150 kg of fuel burns out unevenly or is offset, causing a modest 1.5 cm (0.015 m) 
    # shift in the physical COM relative to the original body geometric axis:
    com_shift = np.array([0.015, 0.0, 0.0]) # 1.5 cm shift along X-axis
    
    # Simulating a maximum thruster force command (e.g., translation maneuver)
    # Suppose a subset of thrusters generates a net force of 10 mN (10e-3 N)
    net_force_B = np.array([0.0, 0.010, 0.0]) 
    
    # Parasitic cross-coupling torque generated physically but omitted by simulator:
    parasitic_torque = np.cross(com_shift, net_force_B)
    print(f"Assumed Center of Pressure Offset (Static): {param.r_CP_L} m")
    print(f"Unmodeled parasitic torque from a 1.5cm COM shift under 10mN thrust: {parasitic_torque} N*m")
    print("Observation: Control systems might encounter unmodeled disturbances during long burns.")

    # ──────────────────────────────────────────────────────────────────────────
    # TEST 4: Quaternion Sign Ambiguity Check (+q vs -q)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[TEST 4] Quaternion Tracking Error Ambiguity (+q vs -q)")
    print("-" * 70)
    
    # Define a clean helper for standard Hamilton quaternion multiplication
    def quat_conjugate(q):
        return np.array([q[0], -q[1], -q[2], -q[3]])
        
    def quat_multiply(q, p):
        w1, x1, y1, z1 = q
        w2, x2, y2, z2 = p
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    # Target Attitude (Commanded Orientation)
    q_cmd = np.array([1.0, 0.0, 0.0, 0.0])
    
    # Actual Attitude A (+q)
    q_actual_pos = np.array([0.9961947, 0.0871557, 0.0, 0.0]) # ~10 degree rotation
    # Actual Attitude B (-q) represents the exact same spatial pointing rotation matrix
    q_actual_neg = -q_actual_pos
    
    # Standard controller error calculation: q_err = q_cmd^-1 * q_actual
    q_err_pos = quat_multiply(quat_conjugate(q_cmd), q_actual_pos)
    q_err_neg = quat_multiply(quat_conjugate(q_cmd), q_actual_neg)
    
    print(f"Commanded Quaternion (q_cmd)  : {q_cmd}")
    print(f"Positive State Vector (+q)    : {q_actual_pos}")
    print(f"Negative State Vector (-q)    : {q_actual_neg}")
    print(f"Error Quaternion from (+q)    : {q_err_pos} -> Vector part: {q_err_pos[1:4]}")
    print(f"Error Quaternion from (-q)    : {q_err_neg} -> Vector part: {q_err_neg[1:4]}")
    
    # Compute angle error metric often used by attitude control feedback loops: theta_err = 2 * acos(q_err[0])
    # Protect bounds against floating point errors
    val_pos = np.clip(q_err_pos[0], -1.0, 1.0)
    val_neg = np.clip(q_err_neg[0], -1.0, 1.0)
    angle_err_pos = 2.0 * np.arccos(val_pos) * (180.0 / np.pi)
    angle_err_neg = 2.0 * np.arccos(val_neg) * (180.0 / np.pi)
    
    print(f"Calculated Loop Angular Error (+q): {angle_err_pos:.2f}°")
    print(f"Calculated Loop Angular Error (-q): {angle_err_neg:.2f}° <--- 360° Unwinding Path Trap!")
    print("=" * 70)


if __name__ == "__main__":
    run_diagnostics()