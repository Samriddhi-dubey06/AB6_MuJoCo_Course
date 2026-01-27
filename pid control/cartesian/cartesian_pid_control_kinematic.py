"""
Cartesian PID Control - Kinematic (Teaching Version)

This version uses direct position control (qpos) instead of actuators.
Perfect for teaching Cartesian PID concepts without physics complexity.

Approach:
- Compute Cartesian PID force
- Convert to joint velocity using Jacobian pseudoinverse
- Integrate to get joint position change
- Apply directly to qpos (teleporting)
- Use mj_forward() for kinematics only (no dynamics)

Students learn:
- Cartesian space PID: F = Kp*e - Kd*v + Ki*integral(e)
- Jacobian maps Cartesian ↔ Joint space
- Pseudoinverse: Δq = J⁺ × Δx
"""

import sys
import os
import numpy as np
import mujoco
import mujoco.viewer
import time

# Add parent directory to path to import robot_model
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from robox_parameters.robot_model import Robot

# -------------------------------------------------
# Load MuJoCo model
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(BASE_DIR, "..", "..", "xmls", "scene.xml")

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# -------------------------------------------------
# Load Robot from configuration (for Jacobian)
# -------------------------------------------------
CONFIG_PATH = os.path.join(BASE_DIR, "..", "..", "robox_parameters", "robot_parameters.json")
robot = Robot.from_config(CONFIG_PATH)

# -------------------------------------------------
# Joint indices
# -------------------------------------------------
j1 = model.joint("robot_joint1").qposadr[0]
j2 = model.joint("robot_joint2").qposadr[0]
j3 = model.joint("robot_joint3").qposadr[0]

# -------------------------------------------------
# End-effector body (for position feedback)
# -------------------------------------------------
EE_BODY = model.body("gripper_link").id

# -------------------------------------------------
# Desired Cartesian position (meters) - XY plane
# -------------------------------------------------
x_des = np.array([0.15, 0.12, 0.0])

# -------------------------------------------------
# Cartesian PID gains (tune these!)
# -------------------------------------------------
# For kinematic control, these determine how fast
# the end-effector moves toward the target

Kp = np.array([10.0, 10.0, 0.0])   # Proportional (pull toward target)
Ki = np.array([0.5, 0.5, 0.0])     # Integral (eliminate steady-state error)
Kd = np.array([2.0, 2.0, 0.0])     # Derivative (smooth motion)

# -------------------------------------------------
# PID state variables
# -------------------------------------------------
integral = np.zeros(3)
prev_error = np.zeros(3)
dt = 0.01  # Control timestep

# -------------------------------------------------
# Reset simulation
# -------------------------------------------------
mujoco.mj_resetData(model, data)
data.qpos[j1] = 0.0
data.qpos[j2] = 0.0
data.qpos[j3] = 0.0
mujoco.mj_forward(model, data)

# Get initial position
x_init = data.xpos[EE_BODY].copy()

print("=" * 65)
print("CARTESIAN PID CONTROL - KINEMATIC (Teaching Version)")
print("=" * 65)
print(f"Initial EE position: ({x_init[0]:.4f}, {x_init[1]:.4f}, {x_init[2]:.4f}) m")
print(f"Target EE position:  ({x_des[0]:.4f}, {x_des[1]:.4f}, {x_des[2]:.4f}) m")
print(f"Kp: {Kp}, Ki: {Ki}, Kd: {Kd}")
print("=" * 65)

# -------------------------------------------------
# Viewer + control loop
# -------------------------------------------------
sim_time = 0.0
max_sim_time = 10.0

with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running() and sim_time < max_sim_time:

        # --- Current state ---
        q_current = np.array([data.qpos[j1], data.qpos[j2], data.qpos[j3]])
        x_current = data.xpos[EE_BODY].copy()

        # --- Cartesian PID Error ---
        error = x_des - x_current

        # Integral (accumulated error)
        integral = integral + error * dt
        # Anti-windup: limit integral
        integral = np.clip(integral, -1.0, 1.0)

        # Derivative (rate of change of error)
        derivative = (error - prev_error) / dt
        prev_error = error.copy()

        # --- Cartesian PID Formula ---
        # This gives us desired Cartesian velocity
        x_dot_des = Kp * error + Ki * integral + Kd * derivative

        # --- Get Jacobian (using custom library) ---
        J_full = robot.jacobian(q_current)  # 6x3 matrix
        Jp = J_full[:3, :]                   # 3x3 position Jacobian

        # --- Convert Cartesian velocity to joint velocity ---
        # Using pseudoinverse: q_dot = J⁺ × x_dot
        # J⁺ = (JᵀJ)⁻¹Jᵀ for overdetermined, or Jᵀ(JJᵀ)⁻¹ for underdetermined
        try:
            J_pinv = np.linalg.pinv(Jp)
            q_dot_des = J_pinv @ x_dot_des
        except:
            q_dot_des = np.zeros(3)

        # --- Apply as position change (kinematic integration) ---
        delta_q = q_dot_des * dt

        # --- Update joint positions directly (TELEPORTING) ---
        data.qpos[j1] += delta_q[0]
        data.qpos[j2] += delta_q[1]
        data.qpos[j3] += delta_q[2]

        # --- Forward kinematics only (no dynamics!) ---
        mujoco.mj_forward(model, data)

        sim_time += dt

        # --- Print status every 0.5 seconds ---
        if int(sim_time / 0.5) != int((sim_time - dt) / 0.5):
            pos_error = np.linalg.norm(error[:2])  # XY error
            print(f"t={sim_time:.2f}s | "
                  f"EE=({x_current[0]:7.4f}, {x_current[1]:7.4f}) m | "
                  f"error={pos_error*1000:6.2f} mm | "
                  f"q=({np.rad2deg(q_current[0]):6.1f}, {np.rad2deg(q_current[1]):6.1f}, {np.rad2deg(q_current[2]):6.1f}) deg")

        viewer.sync()
        time.sleep(dt)

# -------------------------------------------------
# Final report
# -------------------------------------------------
x_final = data.xpos[EE_BODY].copy()
final_error = np.linalg.norm((x_des - x_final)[:2])

print("\n" + "=" * 65)
print("SIMULATION COMPLETE")
print("=" * 65)
print(f"Target position:  ({x_des[0]:.4f}, {x_des[1]:.4f}) m")
print(f"Final position:   ({x_final[0]:.4f}, {x_final[1]:.4f}) m")
print(f"Final error:      {final_error*1000:.2f} mm")
print("=" * 65)
