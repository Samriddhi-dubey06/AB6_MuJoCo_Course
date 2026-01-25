import numpy as np
import mujoco
import mujoco.viewer
import os
import time

# -------------------------------------------------
# Load MuJoCo model
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(BASE_DIR, "scene.xml")

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# -------------------------------------------------
# Joint names
# -------------------------------------------------
joint_names = ["robot_joint1", "robot_joint2", "robot_joint3"]

# Joint IDs and qpos addresses
jids = [model.joint(name).id for name in joint_names]
jpos = [model.jnt_qposadr[jid] for jid in jids]

# -------------------------------------------------
# LINK LENGTHS (cm) — used for IK only
# -------------------------------------------------
L1 = 10.0
L2 = 7.0
L3 = 5.0

# -------------------------------------------------
# Clamp using XML joint limits
# -------------------------------------------------
def clamp_to_joint_limits(joint_id, value):
    value = float(value)
    if model.jnt_limited[joint_id]:
        low, high = model.jnt_range[joint_id]
        return float(np.clip(value, low, high))
    return value

# -------------------------------------------------
# Identify robot geoms (EDIT PREFIX IF NEEDED)
# -------------------------------------------------
robot_geom_ids = set()
for g in range(model.ngeom):
    body_id = model.geom_bodyid[g]
    body_name = model.body(body_id).name
    if body_name.startswith("link"):
        robot_geom_ids.add(g)

# -------------------------------------------------
# Filtered self-collision check
# -------------------------------------------------
def real_self_collision(model, data):
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = c.geom1, c.geom2

        if g1 in robot_geom_ids and g2 in robot_geom_ids:
            b1 = model.geom_bodyid[g1]
            b2 = model.geom_bodyid[g2]

            # Ignore parent-child collisions
            if model.body_parentid[b1] == b2:
                continue
            if model.body_parentid[b2] == b1:
                continue

            return True
    return False

# -------------------------------------------------
# ZERO POSITION (PLANAR) — FROM XML LINK OFFSETS
# -------------------------------------------------
# Link positions relative to previous body
p1 = np.array([0.0,        0.0245,   0.0168])
p2 = np.array([-0.017212,  0.017435, 0.0168])
p3 = np.array([-0.016087,  0.018479, 0.0168])

# Planar joint zero angles
theta1_0 = np.arctan2(p1[1], p1[0])
theta2_0 = np.arctan2(p2[1], p2[0])
theta3_0 = np.arctan2(p3[1], p3[0])

theta1_0 = clamp_to_joint_limits(jids[0], theta1_0)
theta2_0 = clamp_to_joint_limits(jids[1], theta2_0)
theta3_0 = clamp_to_joint_limits(jids[2], theta3_0)

# Set initial pose
data.qpos[jpos[0]] = theta1_0
data.qpos[jpos[1]] = theta2_0
data.qpos[jpos[2]] = theta3_0

mujoco.mj_forward(model, data)

print("Zero position initialized (deg):")
print(f"θ1 = {np.rad2deg(theta1_0):.2f}")
print(f"θ2 = {np.rad2deg(theta2_0):.2f}")
print(f"θ3 = {np.rad2deg(theta3_0):.2f}")

# -------------------------------------------------
# Interactive loop
# -------------------------------------------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("\nInteractive MuJoCo planar robot control")
    print("Enter target x,y (cm). Type 'exit' to quit.")

    while viewer.is_running():

        user_input = input("\nEnter x,y: ")

        if user_input.lower() == "exit":
            break

        try:
            x_target, y_target = map(float, user_input.split(","))
        except ValueError:
            print("Invalid input. Use: x,y")
            continue

        # -------------------------------------------------
        # Wrist position
        # -------------------------------------------------
        x_w = x_target - L3
        y_w = y_target
        r = np.sqrt(x_w**2 + y_w**2)

        if r > (L1 + L2):
            print("Target outside workspace")
            continue

        # -------------------------------------------------
        # Inverse kinematics (elbow-down)
        # -------------------------------------------------
        cos_t2 = (r**2 - L1**2 - L2**2) / (2 * L1 * L2)
        cos_t2 = np.clip(cos_t2, -1.0, 1.0)

        theta2 = np.arccos(cos_t2)
        k1 = L1 + L2 * np.cos(theta2)
        k2 = L2 * np.sin(theta2)

        theta1 = np.arctan2(y_w, x_w) - np.arctan2(k2, k1)
        theta3 = -(theta1 + theta2)

        theta1 = clamp_to_joint_limits(jids[0], theta1)
        theta2 = clamp_to_joint_limits(jids[1], theta2)
        theta3 = clamp_to_joint_limits(jids[2], theta3)

        q_goal = np.array([theta1, theta2, theta3], dtype=float)

        q_current = np.array(
            [data.qpos[jpos[i]] for i in range(3)],
            dtype=float
        )

        print("\nJoint angles (deg):")
        print(f"θ1 = {np.rad2deg(theta1):.2f}")
        print(f"θ2 = {np.rad2deg(theta2):.2f}")
        print(f"θ3 = {np.rad2deg(theta3):.2f}")

        # -------------------------------------------------
        # Smooth trajectory with collision checking
        # -------------------------------------------------
        T = 2.0
        dt = 0.01
        steps = int(T / dt)

        collision = False

        for i in range(steps):
            alpha = i / steps
            q = (1 - alpha) * q_current + alpha * q_goal

            for j in range(3):
                data.qpos[jpos[j]] = clamp_to_joint_limits(jids[j], q[j])

            mujoco.mj_step(model, data)

            if real_self_collision(model, data):
                print("Real self-collision detected — stopping")
                collision = True
                break

            viewer.sync()
            time.sleep(dt)

        if not collision:
            print("Target reached successfully")
