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
# Desired Cartesian position (meters)
# -------------------------------------------------
x_des = np.array([0.10, 0.10, 0.0], dtype=float)

# -------------------------------------------------
# Cartesian PID gains
# -------------------------------------------------
Kp = np.array([200.0, 200.0, 0.0])
Kd = np.array([20.0, 20.0, 0.0])
Ki = np.zeros(3)

integral = np.zeros(3)
dt = float(model.opt.timestep)

# -------------------------------------------------
# Viewer + control loop
# -------------------------------------------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():

        # --- Get current joint angles ---
        q = np.array([data.qpos[j1], data.qpos[j2], data.qpos[j3]])

        # --- Get end-effector position from MuJoCo ---
        x = np.array(data.xpos[EE_BODY], dtype=float).reshape(3,)

        # --- Get end-effector velocity from MuJoCo ---
        xd = np.array(data.cvel[EE_BODY][3:6], dtype=float)  # Linear velocity (last 3 of 6)

        # --- PID error computation ---
        error = x_des - x
        integral = integral + error * dt

        # --- Cartesian force ---
        F = Kp * error - Kd * xd + Ki * integral

        # --- Jacobian using custom Robot library ---
        # Returns 6xn matrix: [linear_velocity; angular_velocity]
        J_full = robot.jacobian(q)

        # Extract position Jacobian (first 3 rows)
        Jp = J_full[:3, :]

        # --- Joint torques via Jacobian transpose ---
        tau = Jp.T @ F

        # --- Apply torques ---
        for i in range(model.nu):
            data.ctrl[i] = tau[i]

        # --- Debug print (optional) ---
        # print(f"q(deg): {np.rad2deg(q)}")
        # print(f"EE pos: {x}")
        # print(f"Error: {error}")
        # print(f"Jacobian (custom):\n{Jp}")
        # print(f"Torques: {tau}\n")

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(dt)
