import sys
import os
import numpy as np
import mujoco
import mujoco.viewer
import time

# Add parent directory to path to import robot_model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from robox_parameters.robot_model import Robot

# -------------------------------------------------
# Load MuJoCo model
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(BASE_DIR, "..", "xmls", "scene.xml")

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# -------------------------------------------------
# Load Robot from configuration
# -------------------------------------------------
CONFIG_PATH = os.path.join(BASE_DIR, "..", "robox_parameters", "robot_parameters.json")
robot = Robot.from_config(CONFIG_PATH)

# -------------------------------------------------
# Joint indices
# -------------------------------------------------
j1 = model.joint("robot_joint1").qposadr
j2 = model.joint("robot_joint2").qposadr
j3 = model.joint("robot_joint3").qposadr

# -------------------------------------------------
# Target joint angles (degrees)
# -------------------------------------------------
q_target_deg = np.array([90, 45, 30])
q_target = np.deg2rad(q_target_deg)

# -------------------------------------------------
# Compute Forward Kinematics using inbuilt function
# -------------------------------------------------
T_end_effector = robot.forward_kinematics(q_target)
ee_position = T_end_effector[:3, 3]
ee_orientation = T_end_effector[:3, :3]

print("\n===== FORWARD KINEMATICS (Using Inbuilt Function) =====")
print(f"Target joint angles (deg): {q_target_deg}")
print(f"Target joint angles (rad): {q_target}")
print(f"\nEnd-effector transformation matrix:")
print(T_end_effector)
print(f"\nEnd-effector position (m): {ee_position}")
print(f"\nEnd-effector orientation matrix:")
print(ee_orientation)

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
# Viewer with smooth motion
# -------------------------------------------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    for i in range(steps):
        alpha = i / (steps - 1) if steps > 1 else 1.0

        # Interpolate joint angles
        q = alpha * q_target

        # Compute FK using inbuilt function
        T = robot.forward_kinematics(q)
        ee_inbuilt = T[:3, 3]

        # Apply to MuJoCo
        data.qpos[j1] = q[0]
        data.qpos[j2] = q[1]
        data.qpos[j3] = q[2]
        mujoco.mj_forward(model, data)

        # Get MuJoCo EE position for comparison
        ee_id = model.body("gripper_link").id
        ee_mj = data.xpos[ee_id]

        # Debug print
        print(f"\nStep {i+1}/{steps}:")
        print(f"q(deg) = {np.rad2deg(q)}")
        print(f"Inbuilt FK : {ee_inbuilt}")
        print(f"MuJoCo     : {ee_mj}")

        viewer.sync()
        time.sleep(dt)

    # Hold final pose
    print("\n===== Motion complete. Final pose held. =====")
    while viewer.is_running():
        viewer.sync()
        time.sleep(0.01)
