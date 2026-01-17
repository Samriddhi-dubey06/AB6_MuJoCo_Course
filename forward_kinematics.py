import mujoco
import mujoco.viewer
import numpy as np
import os
import time

# -------------------------------------------------
# Load MuJoCo model
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(BASE_DIR, "scene.xml")

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# -------------------------------------------------
# Joint indices
# -------------------------------------------------
j1 = model.joint("robot_joint1").qposadr
j2 = model.joint("robot_joint2").qposadr
j3 = model.joint("gripper_joint").qposadr

# -------------------------------------------------
# Link lengths (meters)
# -------------------------------------------------
L1 = 0.10
L2 = 0.07
L3 = 0.05

# -------------------------------------------------
# Target angles (degrees)
# -------------------------------------------------
q_target_deg = np.array([90, 45, 0])
q_target = np.deg2rad(q_target_deg)

# -------------------------------------------------
# Homogeneous transforms (FK math)
# -------------------------------------------------
def Tz(theta, L):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0, L*np.cos(theta)],
        [np.sin(theta),  np.cos(theta), 0, L*np.sin(theta)],
        [0,              0,             1, 0],
        [0,              0,             0, 1]
    ])

def forward_kinematics(theta):
    T1 = Tz(theta[0], L1)
    T2 = Tz(theta[1], L2)
    T3 = Tz(theta[2], L3)
    T = T1 @ T2 @ T3
    return T[0:3, 3]  # position

# -------------------------------------------------
# Reset simulation
# -------------------------------------------------
mujoco.mj_resetData(model, data)
data.qpos[:] = 0.0
mujoco.mj_forward(model, data)

# -------------------------------------------------
# Motion parameters
# -------------------------------------------------
motion_time = 3.0
dt = 0.01
steps = int(motion_time / dt)

# -------------------------------------------------
# Viewer
# -------------------------------------------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    for i in range(steps):
        alpha = i / (steps - 1)

        # -------- Mathematical interpolation --------
        q = alpha * q_target

        # -------- Explicit FK math --------
        ee_math = forward_kinematics(q)

        # -------- Apply to MuJoCo --------
        data.qpos[j1] = q[0]
        data.qpos[j2] = q[1]
        data.qpos[j3] = q[2]
        mujoco.mj_forward(model, data)

        # -------- MuJoCo EE position --------
        ee_id = model.body("gripper_link").id
        ee_mj = data.xpos[ee_id]

        # -------- Debug print --------
        print(f"q(deg) = {np.rad2deg(q)}")
        print(f"Math FK : {ee_math}")
        print(f"MuJoCo  : {ee_mj}\n")

        viewer.sync()
        time.sleep(dt)

    while viewer.is_running():
        viewer.sync()
        time.sleep(0.01)
