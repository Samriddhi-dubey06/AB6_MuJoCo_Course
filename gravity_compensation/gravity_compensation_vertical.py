"""
Gravity Compensation + PD Control using MuJoCo's Inbuilt Function
Uses data.qfrc_bias for gravity compensation + PD for position holding.

Control Law:
    tau = qfrc_bias + Kp*(q_des - q) - Kd*qdot

The vertical robot will hold its position against gravity.
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import os

# -------------------------------------------------
# Load vertical robot model
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(BASE_DIR, "..", "xmls", "scene_vertical.xml")

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

print("=" * 60)
print("GRAVITY COMPENSATION + PD (Vertical Robot)")
print("=" * 60)
print(f"Gravity: {model.opt.gravity}")
print(f"Number of actuators: {model.nu}")
print(f"Number of DOFs: {model.nv}")

# -------------------------------------------------
# Joint addresses
# -------------------------------------------------
j1_qpos = model.joint("robot_joint1").qposadr.item()
j2_qpos = model.joint("robot_joint2").qposadr.item()
j3_qpos = model.joint("gripper_joint").qposadr.item()

j1_qvel = model.joint("robot_joint1").dofadr.item()
j2_qvel = model.joint("robot_joint2").dofadr.item()
j3_qvel = model.joint("gripper_joint").dofadr.item()

# -------------------------------------------------
# Desired configuration (where robot will hold)
# -------------------------------------------------
q_des_deg = np.array([45.0, 30.0, 15.0])  # degrees
q_des = np.deg2rad(q_des_deg)

# -------------------------------------------------
# PD Gains (for position holding on top of gravity comp)
# Keep gains LOW - robot has small inertias!
# Joint 3 needs higher gains due to coupling effects
# -------------------------------------------------
Kp = np.array([0.5, 0.8, 1.5])    # Balanced gains
Kd = np.array([0.15, 0.2, 0.3])   # Higher damping to reduce oscillation

# Per-joint torque limits
MAX_TORQUE = np.array([1.5, 5.5, 1.0])  # Slightly higher limits

# -------------------------------------------------
# Reset simulation and set initial pose
# -------------------------------------------------
mujoco.mj_resetData(model, data)
data.qpos[j1_qpos] = q_des[0]
data.qpos[j2_qpos] = q_des[1]
data.qpos[j3_qpos] = q_des[2]
data.qvel[:] = 0.0
mujoco.mj_forward(model, data)

print(f"\nDesired configuration (deg): {q_des_deg}")
print(f"Desired configuration (rad): {q_des}")
print(f"Kp: {Kp}, Kd: {Kd}")
print("\nRobot should HOLD this position against gravity.")
print("=" * 60)

# -------------------------------------------------
# Simulation parameters
# -------------------------------------------------
dt = float(model.opt.timestep)

# -------------------------------------------------
# Viewer + control loop
# -------------------------------------------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    sim_time = 0.0
    step_count = 0

    while viewer.is_running():
        # Get current joint positions and velocities
        q = np.array([
            data.qpos[j1_qpos],
            data.qpos[j2_qpos],
            data.qpos[j3_qpos]
        ])

        qdot = np.array([
            data.qvel[j1_qvel],
            data.qvel[j2_qvel],
            data.qvel[j3_qvel]
        ])

        # -------------------------------------------------
        # Gravity Compensation + PD Control
        # -------------------------------------------------
        # tau = gravity_comp + Kp*(q_des - q) - Kd*qdot
        # -------------------------------------------------
        tau_gravity = data.qfrc_bias[:model.nv].copy()

        q_error = q_des - q
        tau_pd = Kp * q_error - Kd * qdot

        tau = tau_gravity[:model.nu] + tau_pd

        # Clamp torques to prevent instability (per-joint limits)
        tau = np.clip(tau, -MAX_TORQUE, MAX_TORQUE)

        # Apply torques through actuators
        data.ctrl[:model.nu] = tau

        # Step simulation
        mujoco.mj_step(model, data)
        sim_time += dt
        step_count += 1

        # Print status every 1 second
        if step_count % int(1.0 / dt) == 0:
            q_deg = np.rad2deg(q)
            drift = np.linalg.norm(q - q_des)
            print(f"t={sim_time:.1f}s | drift={drift:.6f} rad | "
                  f"q=({q_deg[0]:.2f}, {q_deg[1]:.2f}, {q_deg[2]:.2f}) deg | "
                  f"tau=({tau[0]:.4f}, {tau[1]:.4f}, {tau[2]:.4f}) Nm")

        viewer.sync()
        time.sleep(dt)

# -------------------------------------------------
# Final report
# -------------------------------------------------
q_final = np.array([
    data.qpos[j1_qpos],
    data.qpos[j2_qpos],
    data.qpos[j3_qpos]
])

print("\n" + "=" * 60)
print("SIMULATION ENDED")
print("=" * 60)
print(f"Desired configuration (rad): {q_des}")
print(f"Final configuration (rad):  {q_final}")
print(f"Total drift: {np.linalg.norm(q_final - q_des):.6f} rad")
print("=" * 60)
