import numpy as np
import mujoco
import mujoco.viewer
import os
import sys
import time

# -------------------------------------------------
# Add project root to path for imports
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "robox_parameters"))

from robox_parameters.robot_model import Robot

# -------------------------------------------------
# Load MuJoCo model
# -------------------------------------------------
XML_PATH = os.path.join(PROJECT_ROOT, "xmls", "scene.xml")

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# -------------------------------------------------
# Load robot model from config
# -------------------------------------------------
CONFIG_PATH = os.path.join(PROJECT_ROOT, "robox_parameters", "robot_parameters.json")

# Create robot manually since from_config expects a specific path structure
import json
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

robot = Robot(config['robot_name'])

for link_params in config['links']:
    robot.add_link(
        joint_type=link_params['joint_type'],
        d=link_params['d'],
        a=link_params['a'],
        alpha=link_params['alpha'],
        mass=link_params['mass'],
        inertia_tensor=link_params['inertia_tensor'],
        center_of_mass=link_params['center_of_mass'],
        theta_offset=link_params.get('theta_offset', 0),
        actuator_id=link_params.get('actuator_id', None),
        joint_limits=link_params.get('joint_limits')
    )
    if 'joint_limits' in link_params:
        robot.joint_limits.append(link_params['joint_limits'])
    else:
        robot.joint_limits.append({
            'position': {'min': -np.pi/2, 'max': np.pi/2},
            'velocity': {'min': -np.pi, 'max': np.pi}
        })

if 'gravity_vector' in config:
    robot.set_gravity_vector(config['gravity_vector'])

# -------------------------------------------------
# Joint names for MuJoCo
# -------------------------------------------------
joint_names = ["robot_joint1", "robot_joint2", "robot_joint3"]

# Joint IDs and qpos addresses
jids = [model.joint(name).id for name in joint_names]
jpos = [model.jnt_qposadr[jid] for jid in jids]

# -------------------------------------------------
# Get link lengths from robot model (DH parameter 'a')
# -------------------------------------------------
L1 = robot.links[0].a
L2 = robot.links[1].a
L3 = robot.links[2].a

# -------------------------------------------------
# Clamp to joint limits using robot model
# -------------------------------------------------
def clamp_to_joint_limits(joint_idx, value):
    limits = robot.joint_limits[joint_idx]
    low = limits['position']['min']
    high = limits['position']['max']
    return float(np.clip(value, low, high))

# -------------------------------------------------
# Planar FK using link lengths (for IK solver)
# -------------------------------------------------
def planar_fk(q):
    """Compute end-effector (x, y) position for planar 3-link robot."""
    theta1, theta2, theta3 = q
    x = L1*np.cos(theta1) + L2*np.cos(theta1+theta2) + L3*np.cos(theta1+theta2+theta3)
    y = L1*np.sin(theta1) + L2*np.sin(theta1+theta2) + L3*np.sin(theta1+theta2+theta3)
    return np.array([x, y])

# -------------------------------------------------
# Numerical IK for planar robot (position only)
# -------------------------------------------------
def solve_ik(x_target, y_target, initial_guess=None):
    """
    Solve IK for planar 3-link robot using scipy least_squares.
    Only optimizes for position (x, y), not orientation.

    Args:
        x_target, y_target: Target position (meters)
        initial_guess: Initial joint angles (optional)

    Returns:
        joint_angles array or None if failed
    """
    from scipy.optimize import least_squares

    target = np.array([x_target, y_target])

    def objective(q):
        pos = planar_fk(q)
        return pos - target

    if initial_guess is None:
        initial_guess = np.array([0.0, 0.0, 0.0])

    # Get joint limits
    lower_bounds = [robot.joint_limits[i]['position']['min'] for i in range(3)]
    upper_bounds = [robot.joint_limits[i]['position']['max'] for i in range(3)]

    # Try multiple initial guesses if first fails
    initial_guesses = [
        initial_guess,
        np.array([0.5, 0.5, 0.0]),
        np.array([-0.5, 0.5, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, -0.5]),
    ]

    best_solution = None
    best_error = float('inf')

    for guess in initial_guesses:
        try:
            result = least_squares(
                objective,
                guess,
                bounds=(lower_bounds, upper_bounds),
                ftol=1e-6,
                max_nfev=500,
                method='trf'
            )

            pos = planar_fk(result.x)
            error = np.linalg.norm(pos - target)

            if error < best_error:
                best_error = error
                best_solution = result.x

            if error < 0.001:  # 1mm - good enough
                break
        except Exception:
            continue

    if best_solution is not None and best_error < 0.005:  # 5mm tolerance
        return best_solution
    return None

# -------------------------------------------------
# Forward Kinematics using planar model
# -------------------------------------------------
def forward_kinematics(joint_angles):
    """Compute end-effector position from joint angles."""
    pos = planar_fk(joint_angles)
    return pos[0], pos[1]  # x, y position

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
print("WORKSPACE IDENTIFICATION (Robot Model - Numerical IK)")
print("=" * 50)
print(f"Robot: {robot.name}")
print(f"Link lengths: L1={L1*100:.1f}cm, L2={L2*100:.1f}cm, L3={L3*100:.1f}cm")
print(f"Max reach: {(L1+L2+L3)*100:.1f} cm")
print("Joint limits:")
for i, limits in enumerate(robot.joint_limits):
    pos_min = np.rad2deg(limits['position']['min'])
    pos_max = np.rad2deg(limits['position']['max'])
    print(f"  Joint {i+1}: [{pos_min:.1f}°, {pos_max:.1f}°]")
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

        # Get current joint angles as initial guess
        q_current = np.array([data.qpos[jpos[i]] for i in range(3)])

        # Solve IK using robot model
        solution = solve_ik(x_target, y_target, initial_guess=q_current)

        if solution is None:
            print("No valid IK solution found (try a different target)")
            continue

        theta1, theta2, theta3 = solution

        # Verify with FK
        x_fk, y_fk = forward_kinematics(solution)

        print(f"\nTarget: ({x_target_cm:.1f}, {y_target_cm:.1f}) cm")
        print(f"FK verification: ({x_fk*100:.2f}, {y_fk*100:.2f}) cm")
        print(f"Joint angles: θ1={np.rad2deg(theta1):.1f}°, θ2={np.rad2deg(theta2):.1f}°, θ3={np.rad2deg(theta3):.1f}°")

        q_goal = np.array([theta1, theta2, theta3])

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
