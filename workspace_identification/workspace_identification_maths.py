import numpy as np
import mujoco
import mujoco.viewer
import os
import time

# -------------------------------------------------
# Load MuJoCo model
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
XML_PATH = os.path.join(PROJECT_ROOT, "xmls", "scene.xml")

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
# LINK LENGTHS (m) — from robot_parameters.json
# -------------------------------------------------
L1 = 0.10  # 10 cm
L2 = 0.07  # 7 cm
L3 = 0.05  # 5 cm

# -------------------------------------------------
# Joint limits (from robot_parameters.json)
# -------------------------------------------------
JOINT_LIMITS = [
    (-3.14, 3.14),   # Joint 1: ±180°
    (-1.57, 1.57),   # Joint 2: ±90°
    (-1.57, 1.57),   # Joint 3: ±90°
]

# -------------------------------------------------
# Clamp to joint limits
# -------------------------------------------------
def clamp_to_joint_limits(joint_idx, value):
    low, high = JOINT_LIMITS[joint_idx]
    return float(np.clip(value, low, high))

# -------------------------------------------------
# Analytical IK with flexible end-effector orientation
# -------------------------------------------------
def inverse_kinematics_analytical(x_target, y_target, phi=None):
    """
    Compute IK for 3-link planar robot.
    If phi is None, tries multiple orientations to find a valid solution.

    Args:
        x_target, y_target: Target position (meters)
        phi: End-effector orientation (radians), None for auto

    Returns:
        (theta1, theta2, theta3) or None if unreachable
    """
    # If phi is specified, try only that orientation
    if phi is not None:
        phi_values = [phi]
    else:
        # Try multiple end-effector orientations
        phi_values = np.linspace(-np.pi, np.pi, 36)  # Every 10 degrees

    best_solution = None
    best_error = float('inf')

    for phi in phi_values:
        # Wrist position (joint 3 location)
        x_w = x_target - L3 * np.cos(phi)
        y_w = y_target - L3 * np.sin(phi)

        r = np.sqrt(x_w**2 + y_w**2)

        # Check if wrist is reachable by first two links
        if r > (L1 + L2) or r < abs(L1 - L2):
            continue

        # Elbow angle (joint 2) - try both elbow-up and elbow-down
        cos_theta2 = (r**2 - L1**2 - L2**2) / (2 * L1 * L2)
        cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)

        for elbow_sign in [1, -1]:  # elbow-down and elbow-up
            theta2 = elbow_sign * np.arccos(cos_theta2)

            # Shoulder angle (joint 1)
            k1 = L1 + L2 * np.cos(theta2)
            k2 = L2 * np.sin(theta2)
            theta1 = np.arctan2(y_w, x_w) - np.arctan2(k2, k1)

            # Wrist angle (joint 3)
            theta3 = phi - theta1 - theta2

            # Check joint limits
            if not (JOINT_LIMITS[0][0] <= theta1 <= JOINT_LIMITS[0][1]):
                continue
            if not (JOINT_LIMITS[1][0] <= theta2 <= JOINT_LIMITS[1][1]):
                continue
            if not (JOINT_LIMITS[2][0] <= theta3 <= JOINT_LIMITS[2][1]):
                continue

            # Verify FK
            x_fk = L1*np.cos(theta1) + L2*np.cos(theta1+theta2) + L3*np.cos(theta1+theta2+theta3)
            y_fk = L1*np.sin(theta1) + L2*np.sin(theta1+theta2) + L3*np.sin(theta1+theta2+theta3)

            error = np.sqrt((x_fk - x_target)**2 + (y_fk - y_target)**2)

            if error < best_error:
                best_error = error
                best_solution = (theta1, theta2, theta3)

    if best_solution is not None and best_error < 0.001:  # 1mm tolerance
        return best_solution
    return None

# -------------------------------------------------
# Forward Kinematics
# -------------------------------------------------
def forward_kinematics(theta1, theta2, theta3):
    """Compute end-effector position from joint angles."""
    x = L1*np.cos(theta1) + L2*np.cos(theta1+theta2) + L3*np.cos(theta1+theta2+theta3)
    y = L1*np.sin(theta1) + L2*np.sin(theta1+theta2) + L3*np.sin(theta1+theta2+theta3)
    return x, y

# -------------------------------------------------
# Identify robot geoms for collision check
# -------------------------------------------------
robot_geom_ids = set()
for g in range(model.ngeom):
    body_id = model.geom_bodyid[g]
    body_name = model.body(body_id).name
    if body_name.startswith("link") or "robot" in body_name.lower():
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
            if model.body_parentid[b1] == b2:
                continue
            if model.body_parentid[b2] == b1:
                continue
            return True
    return False

# -------------------------------------------------
# Initialize robot
# -------------------------------------------------
data.qpos[jpos[0]] = 0.0
data.qpos[jpos[1]] = 0.0
data.qpos[jpos[2]] = 0.0
mujoco.mj_forward(model, data)

# -------------------------------------------------
# Print workspace info
# -------------------------------------------------
print("=" * 50)
print("WORKSPACE IDENTIFICATION (Mathematical IK)")
print("=" * 50)
print(f"Link lengths: L1={L1*100:.1f}cm, L2={L2*100:.1f}cm, L3={L3*100:.1f}cm")
print(f"Max reach: {(L1+L2+L3)*100:.1f} cm")
print(f"Min reach: ~{abs(L1-L2-L3)*100:.1f} cm (approximate)")
print("=" * 50)

# -------------------------------------------------
# Interactive loop
# -------------------------------------------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("\nInteractive MuJoCo planar robot control")
    print("Enter target x,y in cm (e.g., '10,10' or '-5,15')")
    print("Type 'exit' to quit.\n")

    while viewer.is_running():
        user_input = input("Enter x,y (cm): ")

        if user_input.lower() == "exit":
            break

        try:
            parts = user_input.replace(" ", "").split(",")
            x_target_cm, y_target_cm = float(parts[0]), float(parts[1])
        except (ValueError, IndexError):
            print("Invalid input. Use format: x,y (e.g., 10,5)")
            continue

        # Convert cm to meters
        x_target = x_target_cm / 100.0
        y_target = y_target_cm / 100.0

        # Check basic reachability
        r = np.sqrt(x_target**2 + y_target**2)
        if r > (L1 + L2 + L3):
            print(f"Target outside max workspace ({r*100:.1f}cm > {(L1+L2+L3)*100:.1f}cm)")
            continue

        # Solve IK
        solution = inverse_kinematics_analytical(x_target, y_target)

        if solution is None:
            print("No valid IK solution found (joint limits may prevent this pose)")
            continue

        theta1, theta2, theta3 = solution

        # Verify with FK
        x_fk, y_fk = forward_kinematics(theta1, theta2, theta3)

        print(f"\nTarget: ({x_target_cm:.1f}, {y_target_cm:.1f}) cm")
        print(f"FK verification: ({x_fk*100:.2f}, {y_fk*100:.2f}) cm")
        print(f"Joint angles: θ1={np.rad2deg(theta1):.1f}°, θ2={np.rad2deg(theta2):.1f}°, θ3={np.rad2deg(theta3):.1f}°")

        q_goal = np.array([theta1, theta2, theta3])
        q_current = np.array([data.qpos[jpos[i]] for i in range(3)])

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
                data.qpos[jpos[j]] = clamp_to_joint_limits(j, q[j])

            mujoco.mj_step(model, data)

            if real_self_collision(model, data):
                print("Self-collision detected — stopping")
                collision = True
                break

            viewer.sync()
            time.sleep(dt)

        if not collision:
            print("Target reached successfully\n")
