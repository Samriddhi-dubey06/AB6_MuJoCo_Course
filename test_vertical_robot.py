"""
Test script for vertical robot setup.
The robot arm now operates in a vertical plane - gravity will pull it down!
Without control, the arm will fall. With gravity compensation, it will hold position.
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import os

# -------------------------------------------------
# Load vertical robot model
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(BASE_DIR, "xmls", "scene_vertical.xml")

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

print("=" * 60)
print("VERTICAL ROBOT TEST")
print("=" * 60)
print(f"Gravity: {model.opt.gravity}")
print(f"Number of actuators: {model.nu}")
print(f"Number of DOFs: {model.nv}")
print("=" * 60)
print("\nWithout control, the arm will fall due to gravity.")
print("Close viewer to exit.\n")

# -------------------------------------------------
# Joint addresses
# -------------------------------------------------
j1_qpos = model.joint("robot_joint1").qposadr.item()
j2_qpos = model.joint("robot_joint2").qposadr.item()
j3_qpos = model.joint("gripper_joint").qposadr.item()

# -------------------------------------------------
# Set initial pose (arm extended)
# -------------------------------------------------
mujoco.mj_resetData(model, data)
data.qpos[j1_qpos] = 0.0
data.qpos[j2_qpos] = 0.0
data.qpos[j3_qpos] = 0.0
mujoco.mj_forward(model, data)

# -------------------------------------------------
# Simulation parameters
# -------------------------------------------------
dt = float(model.opt.timestep)
APPLY_GRAVITY_COMPENSATION = False  # Set to True to hold position

# -------------------------------------------------
# Viewer loop
# -------------------------------------------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    sim_time = 0.0

    while viewer.is_running():

        if APPLY_GRAVITY_COMPENSATION:
            # Compensate for gravity - robot holds position
            tau_gravity = data.qfrc_bias[:model.nv].copy()
            data.ctrl[:model.nu] = tau_gravity[:model.nu]
        else:
            # No control - robot falls
            data.ctrl[:] = 0.0

        mujoco.mj_step(model, data)
        sim_time += dt

        viewer.sync()
        time.sleep(dt)

print("Simulation ended.")
