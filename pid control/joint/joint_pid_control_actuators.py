"""
Joint PID Control with Actuators
Uses scene_with_actuators.xml for proper torque control via data.ctrl
"""

import mujoco
import mujoco.viewer
import numpy as np
import os
import time

# -------------------------------------------------
# Load model (WITH ACTUATORS)
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(BASE_DIR,"..", "..", "xmls", "scene_with_actuators.xml")

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

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
# Target angles (degrees)
# -------------------------------------------------
user_j1_deg = 45
user_j2_deg = 30
user_j3_deg = 20

q_target = np.array([
    np.deg2rad(user_j1_deg),
    np.deg2rad(user_j2_deg),
    np.deg2rad(user_j3_deg)
])

# -------------------------------------------------
# PID control parameters
# -------------------------------------------------
Kp = np.array([10.0, 10.0, 5.0])   # Proportional gains
Ki = np.array([0.5, 0.5, 0.2])    # Integral gains
Kd = np.array([0.5, 0.5, 0.2])    # Derivative gains

prev_error = np.zeros(3)
integral = np.zeros(3)

# -------------------------------------------------
# Torque limits
# -------------------------------------------------
MAX_TORQUE = np.array([8.0, 8.0, 2.0])  # Lower limit for joint 3 (low inertia)

def clamp_torque(tau, max_vals):
    return np.clip(tau, -max_vals, max_vals)

# -------------------------------------------------
# Reset simulation
# -------------------------------------------------
mujoco.mj_resetData(model, data)
data.qpos[j1_qpos] = 0.0
data.qpos[j2_qpos] = 0.0
data.qpos[j3_qpos] = 0.0
data.qvel[:] = 0.0
mujoco.mj_forward(model, data)

print("=" * 60)
print("JOINT PID CONTROL (WITH ACTUATORS)")
print("=" * 60)
print(f"Target angles (deg): [{user_j1_deg}, {user_j2_deg}, {user_j3_deg}]")
print(f"Target angles (rad): {q_target}")
print(f"Kp: {Kp}, Ki: {Ki}, Kd: {Kd}")
print("=" * 60)

# -------------------------------------------------
# Simulation parameters
# -------------------------------------------------
dt = float(model.opt.timestep)
sim_time = 0.0
max_sim_time = 10.0

# -------------------------------------------------
# Viewer + PID control loop
# -------------------------------------------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    step_count = 0

    while viewer.is_running() and sim_time < max_sim_time:
        # Current joint positions
        q_current = np.array([
            data.qpos[j1_qpos],
            data.qpos[j2_qpos],
            data.qpos[j3_qpos]
        ])

        # Current joint velocities
        qdot = np.array([
            data.qvel[j1_qvel],
            data.qvel[j2_qvel],
            data.qvel[j3_qvel]
        ])

        # PID computation
        error = q_target - q_current
        integral += error * dt

        # Anti-windup: clamp integral
        integral = np.clip(integral, -1.0, 1.0)

        derivative = -qdot  # Use velocity directly (cleaner than finite difference)

        tau = Kp * error + Ki * integral + Kd * derivative

        # Clamp torques (per-joint limits)
        tau = clamp_torque(tau, MAX_TORQUE)

        # Apply torques through actuators
        data.ctrl[:] = tau

        # Step simulation
        mujoco.mj_step(model, data)
        sim_time += dt
        step_count += 1

        # Print status every 0.5 seconds
        if step_count % int(0.5 / dt) == 0:
            q_deg = np.rad2deg(q_current)
            err_norm = np.linalg.norm(error)
            print(f"t={sim_time:.2f}s | |err|={err_norm:.4f} rad | "
                  f"q=({q_deg[0]:.1f}, {q_deg[1]:.1f}, {q_deg[2]:.1f}) deg")

        # Update previous error
        prev_error = error

        viewer.sync()
        time.sleep(dt)

    # Keep viewer open at final pose
    print("\nHolding final pose... Close viewer to exit.")
    while viewer.is_running():
        # Keep applying control to hold position
        q_current = np.array([
            data.qpos[j1_qpos],
            data.qpos[j2_qpos],
            data.qpos[j3_qpos]
        ])
        qdot = np.array([
            data.qvel[j1_qvel],
            data.qvel[j2_qvel],
            data.qvel[j3_qvel]
        ])
        error = q_target - q_current
        integral += error * dt
        integral = np.clip(integral, -1.0, 1.0)
        tau = Kp * error + Ki * integral - Kd * qdot
        tau = clamp_torque(tau, MAX_TORQUE)
        data.ctrl[:] = tau
        mujoco.mj_step(model, data)

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
print("SIMULATION COMPLETE")
print("=" * 60)
print(f"Final angles (rad): {q_final}")
print(f"Target angles (rad): {q_target}")
print(f"Final error norm: {np.linalg.norm(q_target - q_final):.6f} rad")
print("=" * 60)
