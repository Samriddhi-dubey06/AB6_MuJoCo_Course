import mujoco
import mujoco.viewer
import numpy as np
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
# Joint qpos indices
# -------------------------------------------------
j1 = model.joint("robot_joint1").qposadr
j2 = model.joint("robot_joint2").qposadr
j3 = model.joint("robot_joint3").qposadr

# -------------------------------------------------
# End-effector body (FK reference)
# -------------------------------------------------
ee_body_id = model.body("gripper_link").id

# -------------------------------------------------
# Target joint angles (degrees → radians)
# -------------------------------------------------
q_target_deg = np.array([90, 45, 0])
q_target = np.deg2rad(q_target_deg)

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

        # -------- Joint interpolation --------
        q = alpha * q_target

        # -------- Apply joints --------
        data.qpos[j1] = q[0]
        data.qpos[j2] = q[1]
        data.qpos[j3] = q[2]

        # -------- MuJoCo Forward Kinematics --------
        mujoco.mj_forward(model, data)

        # -------- End-effector FK (world frame) --------
        ee_pos = data.xpos[ee_body_id]     # (x, y, z)
        ee_rot = data.xmat[ee_body_id]     # 3x3 rotation (flattened)

        # -------- Debug print --------
        print(f"q (deg): {np.rad2deg(q)}")
        print(f"EE position (MuJoCo FK): {ee_pos}")
        print()

        viewer.sync()
        time.sleep(dt)

    while viewer.is_running():
        viewer.sync()
        time.sleep(0.01)
