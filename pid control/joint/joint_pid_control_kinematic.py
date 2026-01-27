"""
Joint PID Control - Kinematic (Teaching Version)

This version uses direct position control (qpos) instead of actuators.
Perfect for teaching PID concepts without physics complexity.

Approach:
- Compute PID output as velocity command
- Integrate to get position change
- Apply directly to qpos (teleporting)
- Use mj_forward() for kinematics only (no dynamics)

Students learn:
- PID formula: u = Kp*e + Ki*integral(e) + Kd*de/dt
- How P, I, D terms affect response
- Joint space control concept
"""

import numpy as np
import mujoco
import mujoco.viewer
import os
import time

# -------------------------------------------------
# Load MuJoCo model
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(BASE_DIR, "..", "..", "xmls", "scene.xml")

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# -------------------------------------------------
# Joint indices
# -------------------------------------------------
j1 = model.joint("robot_joint1").qposadr[0]
j2 = model.joint("robot_joint2").qposadr[0]
j3 = model.joint("robot_joint3").qposadr[0]

# -------------------------------------------------
# Target joint angles (degrees → radians)
# -------------------------------------------------
q_target_deg = np.array([45.0, 30.0, -20.0])
q_target = np.deg2rad(q_target_deg)

# -------------------------------------------------
# PID Gains (tune these!)
# -------------------------------------------------
# For kinematic control, these act as velocity gains
# Higher Kp = faster response
# Higher Kd = smoother motion (less overshoot)
# Ki = eliminates steady-state error

Kp = np.array([5.0, 5.0, 5.0])    # Proportional gain
Ki = np.array([0.1, 0.1, 0.1])    # Integral gain
Kd = np.array([1.0, 1.0, 1.0])    # Derivative gain

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

print("=" * 60)
print("JOINT PID CONTROL - KINEMATIC (Teaching Version)")
print("=" * 60)
print(f"Target angles (deg): {q_target_deg}")
print(f"Kp: {Kp}, Ki: {Ki}, Kd: {Kd}")
print("=" * 60)

# -------------------------------------------------
# Viewer + control loop
# -------------------------------------------------
sim_time = 0.0
max_sim_time = 10.0

with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running() and sim_time < max_sim_time:

        # --- Current joint angles ---
        q_current = np.array([data.qpos[j1], data.qpos[j2], data.qpos[j3]])

        # --- PID Error Computation ---
        error = q_target - q_current

        # Integral (accumulated error)
        integral = integral + error * dt

        # Derivative (rate of change of error)
        derivative = (error - prev_error) / dt
        prev_error = error.copy()

        # --- PID Formula ---
        # u = Kp * error + Ki * integral + Kd * derivative
        # This gives us a "velocity" command
        u = Kp * error + Ki * integral + Kd * derivative

        # --- Apply as position change (kinematic integration) ---
        # delta_q = u * dt (velocity * time = position change)
        delta_q = u * dt

        # --- Update joint positions directly (TELEPORTING) ---
        data.qpos[j1] += delta_q[0]
        data.qpos[j2] += delta_q[1]
        data.qpos[j3] += delta_q[2]

        # --- Forward kinematics only (no dynamics!) ---
        mujoco.mj_forward(model, data)

        sim_time += dt

        # --- Print status every 0.5 seconds ---
        if int(sim_time / 0.5) != int((sim_time - dt) / 0.5):
            q_deg = np.rad2deg(q_current)
            error_deg = np.rad2deg(error)
            print(f"t={sim_time:.2f}s | "
                  f"q=({q_deg[0]:6.1f}, {q_deg[1]:6.1f}, {q_deg[2]:6.1f}) deg | "
                  f"error=({error_deg[0]:6.2f}, {error_deg[1]:6.2f}, {error_deg[2]:6.2f}) deg")

        viewer.sync()
        time.sleep(dt)

# -------------------------------------------------
# Final report
# -------------------------------------------------
q_final = np.array([data.qpos[j1], data.qpos[j2], data.qpos[j3]])
final_error = q_target - q_final

print("\n" + "=" * 60)
print("SIMULATION COMPLETE")
print("=" * 60)
print(f"Target (deg):  {q_target_deg}")
print(f"Final (deg):   {np.rad2deg(q_final)}")
print(f"Error (deg):   {np.rad2deg(final_error)}")
print("=" * 60)
