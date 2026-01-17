import mujoco
import mujoco.viewer
import numpy as np
import time

# =========================
# USER SETTINGS
# =========================
XML_PATH = "scene.xml"   # includes robot.xml
DT = 0.01
TORQUE_GAIN = 1.0        # scaling for gravity compensation
VEL_STEP = 0.02          # how fast joints move per key press

# =========================
# LOAD MODEL
# =========================
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

nu = model.nu
print("Number of actuators:", nu)

# Target joint positions (start pose)
target_qpos = np.zeros(nu)

# Initialize joint positions
data.qpos[:nu] = target_qpos
data.qvel[:] = 0.0
mujoco.mj_forward(model, data)

# =========================
# KEYBOARD CALLBACK
# =========================
def keyboard_callback(keycode):
    global target_qpos

    # Joint 1
    if keycode == ord('Q'):
        target_qpos[0] += VEL_STEP
    elif keycode == ord('A'):
        target_qpos[0] -= VEL_STEP

    # Joint 2
    elif keycode == ord('W'):
        target_qpos[1] += VEL_STEP
    elif keycode == ord('S'):
        target_qpos[1] -= VEL_STEP

    # Gripper
    elif keycode == ord('E'):
        target_qpos[2] += VEL_STEP
    elif keycode == ord('D'):
        target_qpos[2] -= VEL_STEP


# =========================
# MAIN LOOP
# =========================
with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=keyboard_callback) as viewer:

    while viewer.is_running():

        # --------------------------------
        # 1. Hold target positions (no drift)
        # --------------------------------
        data.qpos[:nu] = target_qpos
        data.qvel[:] = 0.0

        # --------------------------------
        # 2. Forward dynamics
        # --------------------------------
        mujoco.mj_forward(model, data)

        # --------------------------------
        # 3. Gravity compensation torques
        # --------------------------------
        gravity_torque = data.qfrc_bias[:nu]

        # --------------------------------
        # 4. Apply torques (ONLY gravity)
        # --------------------------------
        data.ctrl[:] = -TORQUE_GAIN * gravity_torque

        # --------------------------------
        # 5. Step simulation
        # --------------------------------
        mujoco.mj_step(model, data)

        viewer.sync()
        time.sleep(DT)
