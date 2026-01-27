import numpy as np
import mujoco
import mujoco.viewer
import os
import time

# -------------------------------------------------
# Load model
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(BASE_DIR, "..", "..", "xmls", "scene.xml")

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# -------------------------------------------------
# End-effector body
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

        # --- FORCE 1-D CARTESIAN VECTORS ---
        x  = np.array(data.xpos[EE_BODY], dtype=float).reshape(3,)
        xd = np.array(data.cvel[EE_BODY][3:6], dtype=float)  # Linear velocity (last 3 of 6)

        # --- PID ---
        error = x_des - x
        integral = integral + error * dt

        F = Kp * error - Kd * xd + Ki * integral

        # --- Jacobian ---
        Jp = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, Jp, None, EE_BODY)

        # --- Joint torques ---
        tau = Jp.T @ F

        # --- Apply torques ---
        for i in range(model.nu):
            data.ctrl[i] = tau[i]

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(dt)
