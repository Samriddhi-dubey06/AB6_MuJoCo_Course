import numpy as np
import mujoco
import mujoco.viewer
import os
import time

# -------------------------------------------------
# Load MuJoCo model
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(BASE_DIR, "..", "xmls", "scene.xml")

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# -------------------------------------------------
# Joint addresses
# -------------------------------------------------
j1 = model.joint("robot_joint1").qposadr
j2 = model.joint("robot_joint2").qposadr
j3 = model.joint("robot_joint3").qposadr

# -------------------------------------------------
# LINK LENGTHS (cm)
# -------------------------------------------------
L1 = 10.0    # base → link1
L2 = 7.0     # link1 → link2
L3 = 5.0     # link2 → gripper

# -------------------------------------------------
# Desired end-effector position (cm)
# -------------------------------------------------
x_target = 15.0
y_target = 8.0
phi = 0.0   # gripper orientation (rad)

# -------------------------------------------------
# INVERSE KINEMATICS (ANALYTICAL)
# -------------------------------------------------

# Wrist position
x_w = x_target - L3 * np.cos(phi)
y_w = y_target - L3 * np.sin(phi)

r = np.sqrt(x_w**2 + y_w**2)

if r > (L1 + L2):
    raise ValueError("Target outside reachable workspace")

# Elbow (joint 2)
cos_theta2 = (r**2 - L1**2 - L2**2) / (2 * L1 * L2)
theta2 = np.arccos(np.clip(cos_theta2, -1.0, 1.0))  # elbow-down

# Shoulder (joint 1)
k1 = L1 + L2 * np.cos(theta2)
k2 = L2 * np.sin(theta2)
theta1 = np.arctan2(y_w, x_w) - np.arctan2(k2, k1)

# Wrist (joint 3)
theta3 = phi - theta1 - theta2

# -------------------------------------------------
# Joint trajectory (0,0,0) → IK solution
# -------------------------------------------------
q_start = np.array([0.0, 0.0, 0.0])
q_goal = np.array([theta1, theta2, theta3])

# Initialize robot at zero pose
data.qpos[j1] = q_start[0]
data.qpos[j2] = q_start[1]
data.qpos[j3] = q_start[2]
mujoco.mj_forward(model, data)

# Trajectory timing
T = 2.0      # seconds
dt = 0.01
steps = int(T / dt)

# -------------------------------------------------
# Print results
# -------------------------------------------------
print("\n===== INVERSE KINEMATICS =====")
print(f"Target position: x = {x_target:.2f} cm, y = {y_target:.2f} cm")
print("Computed joint angles:")
print(f"θ1 = {np.rad2deg(theta1):.2f} deg")
print(f"θ2 = {np.rad2deg(theta2):.2f} deg")
print(f"θ3 = {np.rad2deg(theta3):.2f} deg")

# -------------------------------------------------
# Viewer + smooth motion
# -------------------------------------------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Move from zero pose to target pose
    for i in range(steps):
        alpha = i / steps

        q = (1 - alpha) * q_start + alpha * q_goal

        data.qpos[j1] = q[0]
        data.qpos[j2] = q[1]
        data.qpos[j3] = q[2]

        mujoco.mj_forward(model, data)

        viewer.sync()
        time.sleep(dt)

    # Hold final pose
    while viewer.is_running():
        viewer.sync()
        time.sleep(0.01)
