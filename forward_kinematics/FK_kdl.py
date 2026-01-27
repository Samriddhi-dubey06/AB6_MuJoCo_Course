import numpy as np
import PyKDL as kdl
import mujoco
import mujoco.viewer
import os
import time
import json

# =================================================
# 1. Load robot parameters from JSON
# =================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARAMS_PATH = os.path.join(BASE_DIR, "..", "robox_parameters", "robot_parameters.json")

with open(PARAMS_PATH, 'r') as f:
    robot_params = json.load(f)

# =================================================
# 2. KDL: Build chain from robot parameters
# =================================================
chain = kdl.Chain()

for link in robot_params['links']:
    chain.addSegment(
        kdl.Segment(
            kdl.Joint(kdl.Joint.RotZ),
            kdl.Frame(kdl.Vector(link['a'], 0.0, 0.0))
        )
    )

n_joints = chain.getNrOfJoints()

# FK solver
fk_solver = kdl.ChainFkSolverPos_recursive(chain)

# =================================================
# 3. Define joint angles for FK
# =================================================
q_fk = kdl.JntArray(n_joints)

# Example joint configuration (can be changed)
q_fk[0] = np.deg2rad(90)
q_fk[1] = np.deg2rad(30)
q_fk[2] = np.deg2rad(0)

# =================================================
# 4. Compute Forward Kinematics using KDL
# =================================================
ee_frame = kdl.Frame()
fk_solver.JntToCart(q_fk, ee_frame)

ee_pos_kdl = np.array([
    ee_frame.p.x(),
    ee_frame.p.y(),
    ee_frame.p.z()
])

print("\n===== KDL FORWARD KINEMATICS =====")
print("Joint angles (deg):", np.rad2deg([q_fk[i] for i in range(n_joints)]))
print("EE position from KDL (m):", ee_pos_kdl)

# =================================================
# 5. Load MuJoCo model
# =================================================
XML_PATH = os.path.join(BASE_DIR, "..", "xmls", "scene.xml")

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# Joint addresses in MuJoCo
j1 = model.joint("robot_joint1").qposadr[0]
j2 = model.joint("robot_joint2").qposadr[0]
j3 = model.joint("robot_joint3").qposadr[0]

# End-effector body
ee_body_id = model.body("gripper_link").id

# =================================================
# 6. Visualize motion in MuJoCo
# =================================================
with mujoco.viewer.launch_passive(model, data) as viewer:

    # Smooth transition from zero pose
    q_start = np.zeros(3)
    q_goal = np.array([q_fk[0], q_fk[1], q_fk[2]])

    T = 2.0
    dt = 0.01
    steps = int(T / dt)

    for i in range(steps):
        alpha = i / steps
        q = (1 - alpha) * q_start + alpha * q_goal

        data.qpos[j1] = q[0]
        data.qpos[j2] = q[1]
        data.qpos[j3] = q[2]

        mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(dt)

    # Print final position after motion completes
    ee_pos_mj = data.xpos[ee_body_id]
    print("\n===== MUJOCO FK CHECK =====")
    print("EE position from MuJoCo (m):", ee_pos_mj)

    # Hold final pose - freeze robot
    final_qpos = data.qpos.copy()
    data.qvel[:] = 0.0

    while viewer.is_running():
        # Lock position and velocity to prevent drift
        data.qpos[:] = final_qpos
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(0.01)
