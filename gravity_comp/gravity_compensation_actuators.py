"""
Gravity Compensation with Actuators
Uses scene_with_actuators.xml to apply gravity compensation torques via data.ctrl

The robot holds its current position by compensating for gravity.
data.qfrc_bias contains gravity + Coriolis + centrifugal forces.
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import os

# -------------------------------------------------
# Load model (WITH ACTUATORS)
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(BASE_DIR, "xmls", "scene_with_actuators.xml")

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

print(f"Number of actuators: {model.nu}")
print(f"Number of DOFs: {model.nv}")
print(f"Gravity: {model.opt.gravity}")

# -------------------------------------------------
# Joint addresses
# -------------------------------------------------
j1_qpos = model.joint("robot_joint1").qposadr.item()
j2_qpos = model.joint("robot_joint2").qposadr.item()
j3_qpos = model.joint("gripper_joint").qposadr.item()

# -------------------------------------------------
# Initial configuration (where robot will hold)
# -------------------------------------------------
initial_q = np.array([0.5, 0.3, 0.2])  # radians

# -------------------------------------------------
# Reset simulation
# -------------------------------------------------
mujoco.mj_resetData(model, data)
data.qpos[j1_qpos] = initial_q[0]
data.qpos[j2_qpos] = initial_q[1]
data.qpos[j3_qpos] = initial_q[2]
data.qvel[:] = 0.0
mujoco.mj_forward(model, data)

print("=" * 60)
print("GRAVITY COMPENSATION (WITH ACTUATORS)")
print("=" * 60)
print(f"Initial configuration (rad): {initial_q}")
print(f"Initial configuration (deg): {np.rad2deg(initial_q)}")
print("Robot should hold this position against gravity.")
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
        # Get current joint positions
        q_current = np.array([
            data.qpos[j1_qpos],
            data.qpos[j2_qpos],
            data.qpos[j3_qpos]
        ])

        # Gravity compensation: apply bias forces (gravity + Coriolis)
        # qfrc_bias = C(q, qdot) * qdot + g(q)
        # At zero velocity, this is just gravity
        tau_gravity = data.qfrc_bias[:model.nv].copy()

        # Apply compensation torques through actuators
        data.ctrl[:model.nu] = tau_gravity[:model.nu]

        mujoco.mj_step(model, data)
        sim_time += dt
        step_count += 1

        # Print status every 1 second
        if step_count % int(1.0 / dt) == 0:
            q_deg = np.rad2deg(q_current)
            drift = np.linalg.norm(q_current - initial_q)
            print(f"t={sim_time:.1f}s | drift={drift:.6f} rad | "
                  f"q=({q_deg[0]:.2f}, {q_deg[1]:.2f}, {q_deg[2]:.2f}) deg | "
                  f"tau=({tau_gravity[0]:.3f}, {tau_gravity[1]:.3f}, {tau_gravity[2]:.3f}) Nm")

        viewer.sync()
        time.sleep(dt)

print("\n" + "=" * 60)
print("SIMULATION ENDED")
print("=" * 60)
