import mujoco
import mujoco.viewer
import numpy as np
import time
import os

# --------------------------------------------------
# Load model
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(BASE_DIR, "scene.xml")

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

assert model.nu == 3, "Expected 3 actuators"

# --------------------------------------------------
# PD gains (stable for planar arm)
# --------------------------------------------------
Kp = np.array([20.0, 15.0, 10.0])
Kd = np.array([2.0, 1.5, 1.0])

# --------------------------------------------------
# Desired joint angles (DEGREES)
# --------------------------------------------------
q_des_deg = np.array([130.0, 50.0, -50.0])
q_des = np.deg2rad(q_des_deg)

qd_des = np.zeros(3)

# --------------------------------------------------
# Initialize state
# --------------------------------------------------
data.qpos[:3] = q_des
data.qvel[:3] = 0.0
mujoco.mj_forward(model, data)

# --------------------------------------------------
# Viewer loop
# --------------------------------------------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():

        # Current state
        q = data.qpos[:3]
        qd = data.qvel[:3]

        # PD torque
        tau = Kp * (q_des - q) + Kd * (qd_des - qd)

        # Torque limits (safety)
        tau = np.clip(tau,
                      [-0.2, -0.15, -0.1],
                      [ 0.2,  0.15,  0.1])

        data.ctrl[:] = tau

        mujoco.mj_step(model, data)

        viewer.sync()
        time.sleep(0.01)
