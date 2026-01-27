"""
Trajectory tracking using quintic polynomials - SQUARE SHAPE
Draws a square with corners at specified points
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

# =================================================
# 2. Inverse Kinematics Solver
# =================================================

def solve_ik_custom(robot, target_pos, q_init, joint_limits, verbose=False):
    """Solve IK using custom Robot class WITH joint limit checking"""
    try:
        # Identity orientation (only position control)
        target_orientation = np.eye(3)

        q_sol = robot.inverse_kinematics(
            position=target_pos,
            orientation=target_orientation,
            initial_guess=q_init,
            tolerance=1e-4,
            max_iter=100  # More iterations for better convergence
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
# 3. Main Program
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
print("QUINTIC POLYNOMIAL TRAJECTORY - SQUARE SHAPE")
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

# Define square corners
# Points: (10cm, 10cm), (15cm, 10cm), (-10cm, 10cm), (-10cm, 15cm)
# Converting to meters: divide by 100
print("\n" + "="*60)
print("DRAWING SQUARE")
print("="*60)

# Square corners (in meters)
corners = [
    np.array([0.10, 0.10, ee_initial[2]]),   # Corner 1: (10cm, 10cm)
    np.array([0.15, 0.10, ee_initial[2]]),   # Corner 2: (15cm, 10cm)
    np.array([-0.10, 0.10, ee_initial[2]]),  # Corner 3: (-10cm, 10cm)
    np.array([-0.10, 0.15, ee_initial[2]]),  # Corner 4: (-10cm, 15cm)
    np.array([0.10, 0.10, ee_initial[2]]),   # Back to Corner 1 to close the square
]

print("Square corners:")
for i, corner in enumerate(corners):
    print(f"  Corner {i+1}: {corner} = ({corner[0]*100:.1f}cm, {corner[1]*100:.1f}cm)")

# Generate waypoints along each edge
waypoints = []
points_per_edge = 10

for i in range(len(corners) - 1):
    edge_waypoints = np.linspace(corners[i], corners[i+1], points_per_edge, endpoint=False)
    waypoints.extend(edge_waypoints)

# Add the last point to close the square
waypoints.append(corners[-1])
waypoints = np.array(waypoints)

print(f"\nGenerated {len(waypoints)} waypoints total")

# =================================================
# 4. Generate Quintic Polynomial Trajectories
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
# 5. Execute Trajectory with MuJoCo Visualization
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

    # Track which corner we're at (10 segments per edge)
    points_per_edge = 10
    corner_segments = [0, points_per_edge, 2*points_per_edge, 3*points_per_edge, 4*points_per_edge]
    corner_names = ["Corner 1 (10cm, 10cm)", "Corner 2 (15cm, 10cm)", "Corner 3 (-10cm, 10cm)", "Corner 4 (-10cm, 15cm)", "Back to Corner 1"]

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

        # Check if we reached a corner (at the END of each edge)
        # Corners are at segments: 9, 19, 29, 39 (because we have 10 segments per edge, 0-indexed)
        ee_current = data.xpos[ee_body_id]

        if seg_idx == 9:  # End of edge 1, reached corner 2
            print(f"\n✓ Reached {corner_names[1]}")
            print(f"  Actual EE position: [{ee_current[0]*100:.1f}cm, {ee_current[1]*100:.1f}cm, {ee_current[2]*100:.1f}cm]")
        elif seg_idx == 19:  # End of edge 2, reached corner 3
            print(f"\n✓ Reached {corner_names[2]}")
            print(f"  Actual EE position: [{ee_current[0]*100:.1f}cm, {ee_current[1]*100:.1f}cm, {ee_current[2]*100:.1f}cm]")
        elif seg_idx == 29:  # End of edge 3, reached corner 4
            print(f"\n✓ Reached {corner_names[3]}")
            print(f"  Actual EE position: [{ee_current[0]*100:.1f}cm, {ee_current[1]*100:.1f}cm, {ee_current[2]*100:.1f}cm]")
        elif seg_idx == 39:  # End of edge 4, back to corner 1
            print(f"\n✓ Reached {corner_names[4]}")
            print(f"  Actual EE position: [{ee_current[0]*100:.1f}cm, {ee_current[1]*100:.1f}cm, {ee_current[2]*100:.1f}cm]")
        # Print regular progress every 5 segments
        elif (seg_idx + 1) % 5 == 0:
            print(f"  Segment {seg_idx + 1}/{len(trajectories_x)} - EE at: [{ee_current[0]*100:.1f}cm, {ee_current[1]*100:.1f}cm]")

    print(f"\nCompleted drawing square!")
    print("Holding final position...")

    # Hold final position
    final_qpos = data.qpos.copy()
    while viewer.is_running():
        data.qpos[:] = final_qpos
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(0.01)
