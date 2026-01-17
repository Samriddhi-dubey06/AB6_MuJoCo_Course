import mujoco
import mujoco.viewer
import numpy as np
import os
import time

# ---------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(BASE_DIR, "scene.xml")

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

print("Number of joints:", model.njnt)
print("Number of actuators:", model.nu)

# ---------------------------------------------------------------------
# Initial joint configuration (MATCH joint ref values!)
# ---------------------------------------------------------------------
# robot_joint1 ref =  2.35
# robot_joint2 ref = -0.03
# gripper_joint ref = -0.08
target_qpos = np.array([2.35, -0.03, -0.08], dtype=float)

# Set initial state ONCE
data.qpos[:model.nu] = target_qpos
data.qvel[:] = 0.0

# Forward kinematics & dynamics initialization
mujoco.mj_forward(model, data)

# ---------------------------------------------------------------------
# Launch viewer
# ---------------------------------------------------------------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():

        # -------------------------------------------------------------
        # 1. Compute bias forces (gravity + coriolis + centrifugal)
        # -------------------------------------------------------------
        # Since qvel = 0 → qfrc_bias = gravity torques only
        gravity_torques = data.qfrc_bias[:model.nu].copy()

        # -------------------------------------------------------------
        # 2. Apply gravity compensation
        # -------------------------------------------------------------
        data.ctrl[:] = -gravity_torques

        # -------------------------------------------------------------
        # 3. Step physics
        # -------------------------------------------------------------
        mujoco.mj_step(model, data)

        # -------------------------------------------------------------
        # 4. Debug print (optional)
        # -------------------------------------------------------------
        print("qpos:", data.qpos[:model.nu])
        print("gravity torques:", gravity_torques)
        print("applied torques:", data.ctrl[:])
        print("-" * 40)

        # -------------------------------------------------------------
        # 5. Sync viewer
        # -------------------------------------------------------------
        viewer.sync()
        time.sleep(0.01)
