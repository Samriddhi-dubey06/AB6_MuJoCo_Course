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
# Define target end-effector position and orientation
# -------------------------------------------------
# Target position (in meters)
target_position = np.array([0.15, 0.08, 0.0])

# Target orientation (identity matrix - no rotation)
# You can modify this to any desired orientation
target_orientation = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

# -------------------------------------------------
# Initial guess for joint angles
# -------------------------------------------------
initial_guess = np.array([0.5, 0.3, 0.2])  # Initial guess in radians

# -------------------------------------------------
# Compute Inverse Kinematics using inbuilt function
# -------------------------------------------------
print("\n===== INVERSE KINEMATICS (Using Inbuilt Function) =====")
print(f"Target position (m): {target_position}")
print(f"Target orientation matrix:")
print(target_orientation)
print(f"\nInitial guess (rad): {initial_guess}")
print(f"Initial guess (deg): {np.rad2deg(initial_guess)}")

# Solve IK
q_solution = robot.inverse_kinematics(
    position=target_position,
    orientation=target_orientation,
    initial_guess=initial_guess,
    tolerance=1e-6,
    max_iter=100
)

print(f"\n===== IK Solution =====")
print(f"Computed joint angles (rad): {q_solution}")
print(f"Computed joint angles (deg): {np.rad2deg(q_solution)}")

# Verify the solution using forward kinematics
T_verify = robot.forward_kinematics(q_solution)
ee_position_verify = T_verify[:3, 3]
ee_orientation_verify = T_verify[:3, :3]

print(f"\n===== Verification (FK of IK solution) =====")
print(f"Achieved position (m): {ee_position_verify}")
print(f"Position error (m): {np.linalg.norm(target_position - ee_position_verify)}")
print(f"Achieved orientation:")
print(ee_orientation_verify)

# -------------------------------------------------
# Initialize robot at zero pose
# -------------------------------------------------
q_start = np.array([0.0, 0.0, 0.0])
data.qpos[j1] = q_start[0]
data.qpos[j2] = q_start[1]
data.qpos[j3] = q_start[2]
mujoco.mj_forward(model, data)

# -------------------------------------------------
# Trajectory timing
# -------------------------------------------------
T = 3.0      # seconds
dt = 0.01
steps = int(T / dt)

# -------------------------------------------------
# Viewer + smooth motion
# -------------------------------------------------
print("\n===== Starting MuJoCo visualization =====")
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Move from zero pose to target pose
    for i in range(steps):
        alpha = i / steps

        # Interpolate from start to IK solution
        q = (1 - alpha) * q_start + alpha * q_solution

        # Apply to MuJoCo
        data.qpos[j1] = q[0]
        data.qpos[j2] = q[1]
        data.qpos[j3] = q[2]
        mujoco.mj_forward(model, data)

        # Get current end-effector position from MuJoCo
        ee_id = model.body("gripper_link").id
        ee_mj = data.xpos[ee_id]

        # Compute FK using inbuilt function for current joints
        T_current = robot.forward_kinematics(q)
        ee_inbuilt = T_current[:3, 3]

        # Debug print (print every 50 steps to avoid clutter)
        if i % 50 == 0:
            print(f"\nStep {i+1}/{steps}:")
            print(f"q(deg) = {np.rad2deg(q)}")
            print(f"Inbuilt FK : {ee_inbuilt}")
            print(f"MuJoCo     : {ee_mj}")

        viewer.sync()
        time.sleep(dt)

    # Hold final pose
    print("\n===== Motion complete. Final pose held. =====")
    print(f"Final joint angles (deg): {np.rad2deg(q_solution)}")
    print(f"Final end-effector position: {ee_position_verify}")

    while viewer.is_running():
        viewer.sync()
        time.sleep(0.01)
