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
    # Add segment with revolute joint and link length
    chain.addSegment(
        kdl.Segment(
            kdl.Joint(kdl.Joint.RotZ),
            kdl.Frame(kdl.Vector(link['a'], 0.0, 0.0))
        )
    )

n_joints = chain.getNrOfJoints()

# Joint limits from robot parameters
q_min = kdl.JntArray(n_joints)
q_max = kdl.JntArray(n_joints)

for i, link in enumerate(robot_params['links']):
    q_min[i] = link['joint_limits']['position']['min']
    q_max[i] = link['joint_limits']['position']['max']

# FK & IK solvers with joint limits
fk_solver = kdl.ChainFkSolverPos_recursive(chain)
ik_vel = kdl.ChainIkSolverVel_pinv(chain)
ik_pos = kdl.ChainIkSolverPos_NR_JL(chain, q_min, q_max, fk_solver, ik_vel, 100, 1e-6)

# =================================================
# 3. Desired EE pose (for IK)
# =================================================
# First, let's verify our FK works by testing with known joint angles
print("\n===== TESTING KDL CHAIN =====")
q_test = kdl.JntArray(n_joints)
q_test[0] = np.deg2rad(45)  # 45 degrees
q_test[1] = np.deg2rad(30)  # 30 degrees
q_test[2] = np.deg2rad(0)   # 0 degrees

fk_result = kdl.Frame()
fk_solver.JntToCart(q_test, fk_result)
print(f"FK test: joints=[45°, 30°, 0°] → pos=[{fk_result.p.x():.3f}, {fk_result.p.y():.3f}, {fk_result.p.z():.3f}]")

# Now use this FK result as our IK target to verify IK works
target_frame = fk_result
print(f"Using FK result as IK target: [{target_frame.p.x():.3f}, {target_frame.p.y():.3f}, {target_frame.p.z():.3f}]")

# Start from zero and try to reach the target
q_init = kdl.JntArray(n_joints)
for i in range(n_joints):
    q_init[i] = 0.0

q_sol = kdl.JntArray(n_joints)

ret = ik_pos.CartToJnt(q_init, target_frame, q_sol)
if ret < 0:
    print(f"KDL IK return code: {ret}")
    # Try with the test angles as initial guess
    ret = ik_pos.CartToJnt(q_test, target_frame, q_sol)
    if ret < 0:
        raise RuntimeError(f"KDL IK did not converge (return code: {ret})")

q_kdl = np.array([q_sol[i] for i in range(n_joints)])

print("\n===== KDL IK SOLUTION =====")
print("Joint angles (deg):", np.rad2deg(q_kdl))

# =================================================
# 4. MuJoCo model loading
# =================================================
XML_PATH = os.path.join(BASE_DIR, "..", "xmls", "scene.xml")

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# Joint addresses in MuJoCo (extract scalar indices)
j1 = model.joint("robot_joint1").qposadr[0]
j2 = model.joint("robot_joint2").qposadr[0]
j3 = model.joint("robot_joint3").qposadr[0]

# End-effector body (for FK check)
ee_body_id = model.body("gripper_link").id

# =================================================
# 5. Apply KDL solution to MuJoCo
# =================================================
mujoco.mj_resetData(model, data)

data.qpos[j1] = q_kdl[0]
data.qpos[j2] = q_kdl[1]
data.qpos[j3] = q_kdl[2]

mujoco.mj_forward(model, data)

ee_pos_mj = data.xpos[ee_body_id]  # Use body position for end-effector

print("\n===== MUJOCO FK CHECK =====")
print("EE position from MuJoCo (m):", ee_pos_mj)
print("Target position (KDL):", [target_frame.p.x(), target_frame.p.y(), target_frame.p.z()])

# =================================================
# 6. Visualize motion in MuJoCo
# =================================================
with mujoco.viewer.launch_passive(model, data) as viewer:

    # Smooth transition from zero pose
    q_start = np.zeros(3)
    q_goal = q_kdl

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
