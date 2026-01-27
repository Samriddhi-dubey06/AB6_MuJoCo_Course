import os

# --- FIX FOR LINUX GLX ERRORS (Must be at the very top) ---
os.environ['MUJOCO_GL'] = 'egl' 

import mujoco
import mujoco.viewer
import numpy as np
import time

# 1. Load Model and Data
BASE_DIR = os.path.dirname(os.path.abspath(_file_))
XML_PATH = os.path.join(BASE_DIR, "scene.xml")

if not os.path.exists(XML_PATH):
    raise FileNotFoundError(f"Missing: {XML_PATH}")

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# 2. Get Body ID (Using 'gripper_link' from your previous logs)
try:
    EE_NAME = "gripper_link"
    ee_id = model.body(EE_NAME).id
except KeyError:
    # Fallback if names changed
    print("Warning: 'gripper_link' not found. Trying 'robot_link2'...")
    ee_id = model.body(model.nbody - 1).id 

# 3. SET REACHABLE TARGET (15cm away, not 50m!)
target_pos = np.array([0.15, 0.0, 0.10]) 

# 4. Control Constants
K_gain = 5.0      
damping = 0.05    
dt = 0.01

print(f"Simulation active. Target set to: {target_pos}")

# 5. Physics and Viewer Loop
try:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Initial reset
        mujoco.mj_resetData(model, data)
        data.qpos[:3] = 0.0
        mujoco.mj_forward(model, data)

        while viewer.is_running():
            step_start = time.time()

            # --- Jacobian Velocity Control (Differential IK) ---
            # 1. Current position
            current_pos = data.xpos[ee_id]
            error = target_pos - current_pos
            
            # 2. Calculate Jacobian
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, jacp, jacr, ee_id)

            # 3. Solve for joint velocities (dq)
            # Using Damped Least Squares to avoid math crashes
            n = jacp.shape[0]
            dq = jacp.T @ np.linalg.solve(jacp @ jacp.T + damping**2 * np.eye(n), error * K_gain)

            # 4. Update joint positions (Integration)
            # Use data.qpos[:3] to target your 3 revolute joints
            data.qpos[:3] += dq[:3] * model.opt.timestep

            # 5. Physics Step & Sync
            mujoco.mj_step(model, data)
            viewer.sync()

            # 6. Real-time sync
            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)

except Exception as e:
    print(f"\nSimulation stopped: {e}")

print("Clean exit.")