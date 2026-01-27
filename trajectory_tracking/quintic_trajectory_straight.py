"""
Trajectory tracking using quintic polynomials
The robot draws shapes (circle, square, triangle) on a grid using smooth trajectories
"""
import numpy as np
import mujoco
import mujoco.viewer
import os
import sys
import time
import json

# Add parent directory to import robot_model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from robox_parameters.robot_model import Robot

# =================================================
# 1. Quintic Polynomial Trajectory Generator
# =================================================

class QuinticPolynomial:
    """
    Generates quintic (5th order) polynomial trajectory
    Ensures smooth position, velocity, and acceleration
    """
    def __init__(self, p0, pf, v0, vf, a0, af, T):
        """
        p0, pf: initial and final positions
        v0, vf: initial and final velocities
        a0, af: initial and final accelerations
        T: total time duration
        """
        self.p0 = p0
        self.pf = pf
        self.v0 = v0
        self.vf = vf
        self.a0 = a0
        self.af = af
        self.T = T

        # Compute coefficients
        self.a0_coeff = p0
        self.a1_coeff = v0
        self.a2_coeff = a0 / 2.0

        A = np.array([
            [T**3, T**4, T**5],
            [3*T**2, 4*T**3, 5*T**4],
            [6*T, 12*T**2, 20*T**3]
        ])

        b = np.array([
            pf - p0 - v0*T - (a0/2.0)*T**2,
            vf - v0 - a0*T,
            af - a0
        ])

        x = np.linalg.solve(A, b)
        self.a3_coeff = x[0]
        self.a4_coeff = x[1]
        self.a5_coeff = x[2]

    def position(self, t):
        """Get position at time t"""
        return (self.a0_coeff +
                self.a1_coeff * t +
                self.a2_coeff * t**2 +
                self.a3_coeff * t**3 +
                self.a4_coeff * t**4 +
                self.a5_coeff * t**5)

    def velocity(self, t):
        """Get velocity at time t"""
        return (self.a1_coeff +
                2 * self.a2_coeff * t +
                3 * self.a3_coeff * t**2 +
                4 * self.a4_coeff * t**3 +
                5 * self.a5_coeff * t**4)

    def acceleration(self, t):
        """Get acceleration at time t"""
        return (2 * self.a2_coeff +
                6 * self.a3_coeff * t +
                12 * self.a4_coeff * t**2 +
                20 * self.a5_coeff * t**3)

# =================================================
# 2. Shape Generators
# =================================================

def generate_circle(center, radius, num_points=50, plane='xy'):
    """Generate points for a circle in specified plane"""
    theta = np.linspace(0, 2*np.pi, num_points)

    if plane == 'xy':
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        z = np.full_like(x, center[2])
    elif plane == 'yz':
        x = np.full_like(theta, center[0])
        y = center[1] + radius * np.cos(theta)
        z = center[2] + radius * np.sin(theta)
    elif plane == 'xz':
        x = center[0] + radius * np.cos(theta)
        y = np.full_like(theta, center[1])
        z = center[2] + radius * np.sin(theta)

    return np.column_stack([x, y, z])

def generate_square(center, side_length, num_points=40):
    """Generate points for a square"""
    half = side_length / 2.0
    points_per_side = num_points // 4

    # Bottom side (left to right)
    x1 = np.linspace(center[0] - half, center[0] + half, points_per_side)
    y1 = np.full_like(x1, center[1] - half)

    # Right side (bottom to top)
    x2 = np.full(points_per_side, center[0] + half)
    y2 = np.linspace(center[1] - half, center[1] + half, points_per_side)

    # Top side (right to left)
    x3 = np.linspace(center[0] + half, center[0] - half, points_per_side)
    y3 = np.full_like(x3, center[1] + half)

    # Left side (top to bottom)
    x4 = np.full(points_per_side, center[0] - half)
    y4 = np.linspace(center[1] + half, center[1] - half, points_per_side)

    x = np.concatenate([x1, x2, x3, x4])
    y = np.concatenate([y1, y2, y3, y4])
    z = np.full_like(x, center[2])

    return np.column_stack([x, y, z])

def generate_triangle(center, side_length, num_points=30):
    """Generate points for an equilateral triangle"""
    points_per_side = num_points // 3
    height = side_length * np.sqrt(3) / 2.0

    # Vertices
    v1 = [center[0], center[1] + 2*height/3]
    v2 = [center[0] - side_length/2, center[1] - height/3]
    v3 = [center[0] + side_length/2, center[1] - height/3]

    # Side 1: v1 to v2
    x1 = np.linspace(v1[0], v2[0], points_per_side)
    y1 = np.linspace(v1[1], v2[1], points_per_side)

    # Side 2: v2 to v3
    x2 = np.linspace(v2[0], v3[0], points_per_side)
    y2 = np.linspace(v2[1], v3[1], points_per_side)

    # Side 3: v3 to v1
    x3 = np.linspace(v3[0], v1[0], points_per_side)
    y3 = np.linspace(v3[1], v1[1], points_per_side)

    x = np.concatenate([x1, x2, x3])
    y = np.concatenate([y1, y2, y3])
    z = np.full_like(x, center[2])

    return np.column_stack([x, y, z])

def generate_star(center, outer_radius, inner_radius, num_points=50):
    """Generate points for a 5-pointed star"""
    points = []
    angles = np.linspace(0, 2*np.pi, 11)  # 5 outer + 5 inner + 1 to close

    for i, angle in enumerate(angles[:-1]):
        if i % 2 == 0:
            # Outer point
            r = outer_radius
        else:
            # Inner point
            r = inner_radius
        x = center[0] + r * np.cos(angle - np.pi/2)
        y = center[1] + r * np.sin(angle - np.pi/2)
        points.append([x, y, center[2]])

    # Interpolate to get smooth path
    points = np.array(points)
    t = np.linspace(0, len(points)-1, num_points)
    x = np.interp(t, range(len(points)), points[:, 0])
    y = np.interp(t, range(len(points)), points[:, 1])
    z = np.full_like(x, center[2])

    return np.column_stack([x, y, z])

# =================================================
# 3. Inverse Kinematics Solver
# =================================================

def solve_ik_custom(robot, target_pos, q_init, joint_limits, verbose=False):
    """Solve IK using custom Robot class with joint limit checking"""
    try:
        # Identity orientation (only position control)
        target_orientation = np.eye(3)

        q_sol = robot.inverse_kinematics(
            position=target_pos,
            orientation=target_orientation,
            initial_guess=q_init,
            tolerance=1e-4,
            max_iter=50
        )

        # Check if solution respects joint limits
        for i in range(len(q_sol)):
            q_min = joint_limits[i]['min']
            q_max = joint_limits[i]['max']

            if q_sol[i] < q_min or q_sol[i] > q_max:
                if verbose:
                    print(f"  WARNING: Joint {i+1} out of limits: {q_sol[i]:.3f} (limits: [{q_min:.3f}, {q_max:.3f}])")
                # Clamp to limits
                q_sol[i] = np.clip(q_sol[i], q_min, q_max)

        return q_sol, True
    except Exception as e:
        if verbose:
            print(f"  WARNING: IK failed for target {target_pos}: {e}")
        return q_init, False

# =================================================
# 4. Main Program
# =================================================

# Load robot parameters
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARAMS_PATH = os.path.join(BASE_DIR, "..", "robox_parameters", "robot_parameters.json")

with open(PARAMS_PATH, 'r') as f:
    robot_params = json.load(f)

# Load MuJoCo model
XML_PATH = os.path.join(BASE_DIR, "..", "xmls", "scene.xml")
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

print("\n" + "="*60)
print("QUINTIC POLYNOMIAL TRAJECTORY TRACKING")
print("="*60)

# Setup Robot class for IK
robot = Robot.from_config(PARAMS_PATH)
print("Using custom Robot class for IK")

# Extract joint limits from robot parameters
joint_limits = [
    {
        'min': robot_params['links'][i]['joint_limits']['position']['min'],
        'max': robot_params['links'][i]['joint_limits']['position']['max']
    }
    for i in range(len(robot_params['links']))
]

print(f"Joint limits (rad):")
for i, limits in enumerate(joint_limits):
    print(f"  Joint {i+1}: [{limits['min']:.3f}, {limits['max']:.3f}] ({np.rad2deg(limits['min']):.1f}°, {np.rad2deg(limits['max']):.1f}°)")

# Get initial end-effector position
mujoco.mj_resetData(model, data)
data.qpos[:] = 0
mujoco.mj_forward(model, data)
ee_body_id = model.body("gripper_link").id
ee_initial = data.xpos[ee_body_id].copy()

print(f"Initial EE position: {ee_initial}")

# Define straight line from (0.10, 0.10) to (-0.10, 0.10)
# Using meters (0.10 m = 10 cm)
print("\n" + "="*60)
print("DRAWING STRAIGHT LINE")
print("="*60)

# Start and end points
start_point = np.array([0.10, 0.10, ee_initial[2]])  # (10cm, 10cm, same Z)
end_point = np.array([-0.10, 0.10, ee_initial[2]])   # (-10cm, 10cm, same Z)

print(f"Start point: {start_point}")
print(f"End point: {end_point}")

# Generate waypoints along the line
num_points = 20
waypoints = np.linspace(start_point, end_point, num_points)

print(f"Generated {len(waypoints)} waypoints for straight line")

# =================================================
# 5. Generate Quintic Polynomial Trajectories
# =================================================

print("\n" + "="*60)
print("GENERATING QUINTIC TRAJECTORIES")
print("="*60)

# Time parameters
segment_time = 1.0  # Time for each segment (slower for visibility)
dt = 0.02  # Simulation timestep

# Generate trajectory for each segment
trajectories_x = []
trajectories_y = []
trajectories_z = []

for i in range(len(waypoints) - 1):
    p_start = waypoints[i]
    p_end = waypoints[i + 1]

    # Quintic polynomial for each axis
    # Start and end with zero velocity and acceleration for smooth motion
    traj_x = QuinticPolynomial(p_start[0], p_end[0], 0, 0, 0, 0, segment_time)
    traj_y = QuinticPolynomial(p_start[1], p_end[1], 0, 0, 0, 0, segment_time)
    traj_z = QuinticPolynomial(p_start[2], p_end[2], 0, 0, 0, 0, segment_time)

    trajectories_x.append(traj_x)
    trajectories_y.append(traj_y)
    trajectories_z.append(traj_z)

print(f"Generated {len(trajectories_x)} trajectory segments")

# =================================================
# 6. Execute Trajectory with MuJoCo Visualization
# =================================================

print("\n" + "="*60)
print("EXECUTING TRAJECTORY")
print("="*60)

with mujoco.viewer.launch_passive(model, data) as viewer:

    # Initialize robot at home position
    mujoco.mj_resetData(model, data)
    data.qpos[:] = 0
    mujoco.mj_forward(model, data)

    print("\nStarting position:")
    print(f"  Joint angles: {data.qpos}")
    print(f"  EE position: {data.xpos[ee_body_id]}")

    # Wait a bit
    for _ in range(100):
        viewer.sync()
        time.sleep(0.01)

    # Move to the first waypoint first
    print(f"\nMoving to first waypoint: {waypoints[0]}")

    q_current = data.qpos.copy()

    # Solve IK for first waypoint
    q_start, success = solve_ik_custom(robot, waypoints[0], q_current, joint_limits, verbose=True)
    print(f"First waypoint joint angles: {q_start} (success: {success})")

    # Smoothly move to first waypoint
    q_init = data.qpos.copy()
    for alpha in np.linspace(0, 1, 100):
        q_interp = (1 - alpha) * q_init + alpha * q_start
        data.qpos[:] = q_interp
        mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(0.01)

    print(f"Reached first waypoint. EE at: {data.xpos[ee_body_id]}")

    # Update current config
    q_current = q_start

    # Execute each trajectory segment
    for seg_idx, (traj_x, traj_y, traj_z) in enumerate(zip(trajectories_x, trajectories_y, trajectories_z)):

        t = 0
        while t <= segment_time and viewer.is_running():
            # Get desired position from quintic polynomial
            x_des = traj_x.position(t)
            y_des = traj_y.position(t)
            z_des = traj_z.position(t)

            target_pos = np.array([x_des, y_des, z_des])

            # Solve IK
            q_sol, success = solve_ik_custom(robot, target_pos, q_current, joint_limits)

            if success:
                # Update current config
                q_current = q_sol
            else:
                # Keep previous config if IK fails
                q_sol = q_current

            # Apply to MuJoCo
            data.qpos[:] = q_sol
            data.qvel[:] = 0  # Zero velocity for position control
            mujoco.mj_forward(model, data)

            # Visualize
            viewer.sync()
            time.sleep(dt)

            t += dt

        # Print progress every 5 segments
        if (seg_idx + 1) % 5 == 0 or seg_idx == 0:
            ee_current = data.xpos[ee_body_id]
            print(f"Segment {seg_idx + 1}/{len(trajectories_x)} - EE at: {ee_current}")

    print(f"\nCompleted drawing straight line!")
    print("Holding final position...")

    # Hold final position
    final_qpos = data.qpos.copy()
    while viewer.is_running():
        data.qpos[:] = final_qpos
        mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(0.01)
