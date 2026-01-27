import mujoco
import mujoco.viewer
import numpy as np
import os
import time

# -------------------------------------------------
# Load model
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(BASE_DIR,"..","..","xmls", "scene.xml")

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# -------------------------------------------------
# Joint addresses
# -------------------------------------------------
j1 = model.joint("robot_joint1").qposadr
j2 = model.joint("robot_joint2").qposadr
j3 = model.joint("robot_joint3").qposadr

# -------------------------------------------------
# Target angles (degrees)
# -------------------------------------------------
user_j1_deg = 75
user_j2_deg = 50
user_j3_deg = 90

q_target = np.array([
    np.deg2rad(user_j1_deg),
    np.deg2rad(user_j2_deg),
    np.deg2rad(user_j3_deg)
])

# -------------------------------------------------
# Reset simulation and start at ZERO
# -------------------------------------------------
mujoco.mj_resetData(model, data)
data.qpos[:] = 0.0
mujoco.mj_forward(model, data)

# -------------------------------------------------
# PID control parameters
# -------------------------------------------------
Kp = np.array([0.0, 0.0, 0.0])   # Proportional gains
Ki = np.array([0.0, 0.0, 0.0])        # Integral gains
Kd = np.array([0.0, 0.0, 0.0])        # Derivative gains

prev_error = np.zeros(3)
integral = np.zeros(3)

# Simulation parameters
dt = 0.01
motion_time = 3.0
steps = int(motion_time / dt)

# -------------------------------------------------
# Viewer + PID control loop
# -------------------------------------------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    for step in range(steps):
        # Current joint positions (flatten to 1D)
        q_current = np.array([data.qpos[j1], data.qpos[j2], data.qpos[j3]]).flatten()

        # PID computation
        error = q_target - q_current
        integral += error * dt
        derivative = (error - prev_error) / dt
        torque = Kp * error + Ki * integral + Kd * derivative

        # Apply torques
        data.qfrc_applied[j1] = torque[0]
        data.qfrc_applied[j2] = torque[1]
        data.qfrc_applied[j3] = torque[2]

        # Step simulation
        mujoco.mj_step(model, data)

        # Update previous error
        prev_error = error

        # Viewer sync
        viewer.sync()
        time.sleep(dt)

    # Keep viewer open at final pose
    while viewer.is_running():
        viewer.sync()
        time.sleep(0.01)
