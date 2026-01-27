"""
Cartesian Teleoperation Script - Using KDL for IK
Control the robot end-effector in Cartesian space
Uses PyKDL (Kinematics and Dynamics Library) for inverse kinematics

USAGE: Type single-letter commands and press ENTER
"""

import numpy as np
import mujoco
import mujoco.viewer
import os
import sys
import time
import json
import threading

# KDL imports
try:
    import PyKDL as kdl
    from PyKDL import ChainFkSolverPos_recursive, ChainIkSolverPos_LMA, ChainIkSolverVel_pinv
except ImportError:
    print("ERROR: PyKDL is required for this script")
    print("Install it with: pip install python-orocos-kdl")
    sys.exit(1)

# Add parent directory to import robot_model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from robox_parameters.robot_model import Robot

# =================================================
# Configuration
# =================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARAMS_PATH = os.path.join(BASE_DIR, "..", "robox_parameters", "robot_parameters.json")
XML_PATH = os.path.join(BASE_DIR, "..", "xmls", "scene.xml")

with open(PARAMS_PATH, 'r') as f:
    robot_params = json.load(f)

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

joint_limits = [
    {
        'min': robot_params['links'][i]['joint_limits']['position']['min'],
        'max': robot_params['links'][i]['joint_limits']['position']['max']
    }
    for i in range(len(robot_params['links']))
]

# =================================================
# Build KDL Chain from Robot Parameters
# =================================================

def build_kdl_chain(robot_params):
    """
    Build a KDL chain from robot parameters (DH parameters)

    Returns:
        kdl.Chain: KDL chain representing the robot
    """
    chain = kdl.Chain()

    print("\nBuilding KDL chain from DH parameters:")
    for i, link_params in enumerate(robot_params['links']):
        # Extract DH parameters
        d = link_params['d']
        a = link_params['a']
        alpha = link_params['alpha']
        theta_offset = link_params.get('theta_offset', 0)

        print(f"  Link {i+1}: d={d:.4f}, a={a:.4f}, alpha={alpha:.4f}, offset={theta_offset:.4f}")

        # Create KDL joint (revolute)
        joint_type = link_params.get('joint_type', 'revolute')
        if joint_type == 'revolute':
            joint = kdl.Joint(kdl.Joint.RotZ)
        else:
            joint = kdl.Joint(kdl.Joint.TransZ)

        # Create KDL segment using DH parameters
        # KDL uses DH convention: Rot(Z, theta) * Trans(Z, d) * Trans(X, a) * Rot(X, alpha)
        frame = kdl.Frame.DH(a, alpha, d, theta_offset)

        segment = kdl.Segment(f"link_{i}", joint, frame)
        chain.addSegment(segment)

    print(f"KDL chain built with {chain.getNrOfJoints()} joints\n")
    return chain

# Build KDL chain
kdl_chain = build_kdl_chain(robot_params)

# Create KDL solvers
fk_solver = ChainFkSolverPos_recursive(kdl_chain)
ik_vel_solver = ChainIkSolverVel_pinv(kdl_chain)
ik_solver = ChainIkSolverPos_LMA(kdl_chain)

print("KDL solvers created:")
print("  - Forward Kinematics: ChainFkSolverPos_recursive")
print("  - Inverse Kinematics: ChainIkSolverPos_LMA (Levenberg-Marquardt)")

# =================================================
# Global State
# =================================================

class State:
    def __init__(self, initial_pos):
        self.target_pos = initial_pos.copy()
        self.current_joints = np.zeros(kdl_chain.getNrOfJoints())
        self.step_size = 0.01  # 1cm
        self.running = True

state = None

# =================================================
# KDL IK Solver
# =================================================

def solve_ik_kdl(target_pos, q_init):
    """
    Solve IK using KDL

    Args:
        target_pos: Desired end-effector position [x, y, z]
        q_init: Initial joint configuration

    Returns:
        (q_solution, success): Joint angles and success flag
    """
    try:
        # Convert numpy array to KDL JntArray
        q_init_kdl = kdl.JntArray(len(q_init))
        for i in range(len(q_init)):
            q_init_kdl[i] = q_init[i]

        # Create target frame (position only, identity rotation)
        target_frame = kdl.Frame(kdl.Vector(target_pos[0], target_pos[1], target_pos[2]))

        # Solve IK
        q_out_kdl = kdl.JntArray(kdl_chain.getNrOfJoints())

        # Use LMA solver (Levenberg-Marquardt)
        result = ik_solver.CartToJnt(q_init_kdl, target_frame, q_out_kdl)

        if result < 0:
            # IK failed
            return q_init, False

        # Convert KDL JntArray back to numpy array
        q_sol = np.array([q_out_kdl[i] for i in range(q_out_kdl.rows())])

        # Apply joint limits
        for i in range(len(q_sol)):
            q_min = joint_limits[i]['min']
            q_max = joint_limits[i]['max']

            if q_sol[i] < q_min or q_sol[i] > q_max:
                # Clamp to limits
                q_sol[i] = np.clip(q_sol[i], q_min, q_max)

        return q_sol, True

    except Exception as e:
        print(f"KDL IK Error: {e}")
        return q_init, False

def forward_kinematics_kdl(q):
    """
    Compute forward kinematics using KDL

    Args:
        q: Joint angles

    Returns:
        position: End-effector position [x, y, z]
    """
    q_kdl = kdl.JntArray(len(q))
    for i in range(len(q)):
        q_kdl[i] = q[i]

    frame_out = kdl.Frame()
    fk_solver.JntToCart(q_kdl, frame_out)

    pos = frame_out.p
    return np.array([pos.x(), pos.y(), pos.z()])

# =================================================
# MuJoCo Viewer Thread
# =================================================

def viewer_thread_func():
    """Run MuJoCo viewer in separate thread"""
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and state.running:
            # Update simulation with current joint angles
            data.qpos[:] = state.current_joints
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.02)

# =================================================
# Main Function
# =================================================

def main():
    global state

    print("\n" + "="*70)
    print("CARTESIAN TELEOPERATION - KDL IK SOLVER")
    print("="*70)
    print("\nCommands (type letter and press ENTER):")
    print("  w : Move +Y (forward)     | s : Move -Y (backward)")
    print("  a : Move -X (left)        | d : Move +X (right)")
    print("  r : Move +Z (up)          | f : Move -Z (down)")
    print("  + : Increase step size    | - : Decrease step size")
    print("  h : Home position         | p : Print status")
    print("  t : Test KDL FK/IK        | q : Quit")
    print("="*70)

    # Initialize
    mujoco.mj_resetData(model, data)
    data.qpos[:] = 0
    mujoco.mj_forward(model, data)

    ee_body_id = model.body("gripper_link").id
    ee_initial = data.xpos[ee_body_id].copy()

    state = State(ee_initial)
    state.current_joints = data.qpos.copy()

    # Verify KDL FK matches MuJoCo
    kdl_pos = forward_kinematics_kdl(state.current_joints)
    print(f"\nInitial position verification:")
    print(f"  MuJoCo FK: [{ee_initial[0]*100:.1f}, {ee_initial[1]*100:.1f}, {ee_initial[2]*100:.1f}] cm")
    print(f"  KDL FK:    [{kdl_pos[0]*100:.1f}, {kdl_pos[1]*100:.1f}, {kdl_pos[2]*100:.1f}] cm")
    print(f"  Difference: {np.linalg.norm(ee_initial - kdl_pos)*100:.3f} cm")
    print(f"\nStep size: {state.step_size*100:.1f} cm\n")

    # Start viewer thread
    viewer_thread = threading.Thread(target=viewer_thread_func, daemon=True)
    viewer_thread.start()
    time.sleep(1)  # Wait for viewer to start

    print("Viewer started! Enter commands below:\n")

    # Command loop
    while state.running:
        try:
            cmd = input("> ").strip().lower()

            if not cmd:
                continue

            # Store current target before modification
            old_target = state.target_pos.copy()

            # Process command
            if cmd == 'w':
                state.target_pos[1] += state.step_size
                print(f"→ Moving +Y by {state.step_size*100:.1f}cm")
            elif cmd == 's':
                state.target_pos[1] -= state.step_size
                print(f"→ Moving -Y by {state.step_size*100:.1f}cm")
            elif cmd == 'a':
                state.target_pos[0] -= state.step_size
                print(f"→ Moving -X by {state.step_size*100:.1f}cm")
            elif cmd == 'd':
                state.target_pos[0] += state.step_size
                print(f"→ Moving +X by {state.step_size*100:.1f}cm")
            elif cmd == 'r':
                state.target_pos[2] += state.step_size
                print(f"→ Moving +Z by {state.step_size*100:.1f}cm")
            elif cmd == 'f':
                state.target_pos[2] -= state.step_size
                print(f"→ Moving -Z by {state.step_size*100:.1f}cm")
            elif cmd == '+' or cmd == '=':
                state.step_size = min(0.05, state.step_size + 0.005)
                print(f"→ Step size: {state.step_size*100:.1f}cm")
                continue
            elif cmd == '-' or cmd == '_':
                state.step_size = max(0.001, state.step_size - 0.005)
                print(f"→ Step size: {state.step_size*100:.1f}cm")
                continue
            elif cmd == 'h':
                state.target_pos = ee_initial.copy()
                state.current_joints = np.zeros(kdl_chain.getNrOfJoints())
                print("→ Returning to home")
            elif cmd == 't':
                # Test KDL FK/IK
                print("\n=== KDL FK/IK Test ===")
                test_q = state.current_joints
                print(f"Current joints: {np.rad2deg(test_q).round(2)}°")

                # FK test
                fk_pos = forward_kinematics_kdl(test_q)
                print(f"FK result: [{fk_pos[0]*100:.1f}, {fk_pos[1]*100:.1f}, {fk_pos[2]*100:.1f}] cm")

                # IK test
                ik_q, success = solve_ik_kdl(fk_pos, np.zeros(len(test_q)))
                print(f"IK success: {success}")
                print(f"IK result: {np.rad2deg(ik_q).round(2)}°")
                print(f"Joint difference: {np.rad2deg(ik_q - test_q).round(3)}°\n")
                continue
            elif cmd == 'p':
                actual_mujoco = data.xpos[ee_body_id].copy()
                actual_kdl = forward_kinematics_kdl(state.current_joints)
                print(f"\nStatus:")
                print(f"  Target EE:     [{state.target_pos[0]*100:.1f}, {state.target_pos[1]*100:.1f}, {state.target_pos[2]*100:.1f}] cm")
                print(f"  Actual (MuJoCo): [{actual_mujoco[0]*100:.1f}, {actual_mujoco[1]*100:.1f}, {actual_mujoco[2]*100:.1f}] cm")
                print(f"  Actual (KDL):    [{actual_kdl[0]*100:.1f}, {actual_kdl[1]*100:.1f}, {actual_kdl[2]*100:.1f}] cm")
                print(f"  Joints: {np.rad2deg(state.current_joints).round(1)}°")
                print(f"  Step size: {state.step_size*100:.1f} cm\n")
                continue
            elif cmd == 'q':
                print("→ Quitting...")
                state.running = False
                break
            else:
                print(f"Unknown command: '{cmd}' (try w/s/a/d/r/f/+/-/h/p/t/q)")
                continue

            # Solve IK using KDL for new target
            print(f"   Solving IK with KDL...")
            q_sol, success = solve_ik_kdl(state.target_pos, state.current_joints)

            if success:
                state.current_joints = q_sol

                # Verify with both MuJoCo and KDL FK
                actual_mujoco = data.xpos[ee_body_id].copy()
                actual_kdl = forward_kinematics_kdl(q_sol)

                error_mujoco = np.linalg.norm(actual_mujoco - state.target_pos)
                error_kdl = np.linalg.norm(actual_kdl - state.target_pos)

                print(f"   Target:      [{state.target_pos[0]*100:.1f}, {state.target_pos[1]*100:.1f}, {state.target_pos[2]*100:.1f}] cm")
                print(f"   MuJoCo FK:   [{actual_mujoco[0]*100:.1f}, {actual_mujoco[1]*100:.1f}, {actual_mujoco[2]*100:.1f}] cm (error: {error_mujoco*100:.2f}cm)")
                print(f"   KDL FK:      [{actual_kdl[0]*100:.1f}, {actual_kdl[1]*100:.1f}, {actual_kdl[2]*100:.1f}] cm (error: {error_kdl*100:.2f}cm)")
                print(f"   Joints: {np.rad2deg(q_sol).round(1)}°")
            else:
                print("   ✗ KDL IK FAILED - target unreachable! Reverting.")
                state.target_pos = old_target

        except (KeyboardInterrupt, EOFError):
            print("\n→ Quitting...")
            state.running = False
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    # Cleanup
    viewer_thread.join(timeout=2.0)
    print("\nTeleoperation ended.\n")

if __name__ == "__main__":
    main()
