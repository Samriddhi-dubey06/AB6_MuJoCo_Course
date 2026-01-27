"""
Cartesian Teleoperation Script - Simple Interactive Version
Control the robot end-effector in Cartesian space
Uses the built-in robot.inverse_kinematics() method

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
robot = Robot.from_config(PARAMS_PATH)

joint_limits = [
    {
        'min': robot_params['links'][i]['joint_limits']['position']['min'],
        'max': robot_params['links'][i]['joint_limits']['position']['max']
    }
    for i in range(len(robot_params['links']))
]

# =================================================
# Global State
# =================================================

class State:
    def __init__(self, initial_pos):
        self.target_pos = initial_pos.copy()
        self.current_joints = np.zeros(len(robot.links))
        self.step_size = 0.01  # 1cm
        self.running = True

state = None

# =================================================
# IK Solver
# =================================================

def solve_ik(target_pos, q_init):
    """Solve IK with joint limits"""
    try:
        target_orientation = np.eye(3)
        q_sol = robot.inverse_kinematics(
            position=target_pos,
            orientation=target_orientation,
            initial_guess=q_init,
            tolerance=1e-4,
            max_iter=100
        )

        # Clip to joint limits
        for i in range(len(q_sol)):
            q_sol[i] = np.clip(q_sol[i], joint_limits[i]['min'], joint_limits[i]['max'])

        return q_sol, True
    except Exception as e:
        return q_init, False

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
    print("CARTESIAN TELEOPERATION - INTERACTIVE MODE")
    print("="*70)
    print("\nCommands (type letter and press ENTER):")
    print("  w : Move +Y (forward)     | s : Move -Y (backward)")
    print("  a : Move -X (left)        | d : Move +X (right)")
    print("  r : Move +Z (up)          | f : Move -Z (down)")
    print("  + : Increase step size    | - : Decrease step size")
    print("  h : Home position         | p : Print status")
    print("  q : Quit")
    print("="*70)

    # Initialize
    mujoco.mj_resetData(model, data)
    data.qpos[:] = 0
    mujoco.mj_forward(model, data)

    ee_body_id = model.body("gripper_link").id
    ee_initial = data.xpos[ee_body_id].copy()

    state = State(ee_initial)
    state.current_joints = data.qpos.copy()

    print(f"\nInitial EE position: [{ee_initial[0]*100:.1f}, {ee_initial[1]*100:.1f}, {ee_initial[2]*100:.1f}] cm")
    print(f"Step size: {state.step_size*100:.1f} cm\n")

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
                state.current_joints = np.zeros(len(robot.links))
                print("→ Returning to home")
            elif cmd == 'p':
                actual = data.xpos[ee_body_id].copy()
                print(f"\nStatus:")
                print(f"  Target EE: [{state.target_pos[0]*100:.1f}, {state.target_pos[1]*100:.1f}, {state.target_pos[2]*100:.1f}] cm")
                print(f"  Actual EE: [{actual[0]*100:.1f}, {actual[1]*100:.1f}, {actual[2]*100:.1f}] cm")
                print(f"  Joints: {np.rad2deg(state.current_joints).round(1)}°")
                print(f"  Step size: {state.step_size*100:.1f} cm\n")
                continue
            elif cmd == 'q':
                print("→ Quitting...")
                state.running = False
                break
            else:
                print(f"Unknown command: '{cmd}' (try w/s/a/d/r/f/+/-/h/p/q)")
                continue

            # Solve IK for new target
            q_sol, success = solve_ik(state.target_pos, state.current_joints)

            if success:
                state.current_joints = q_sol
                actual = data.xpos[ee_body_id].copy()
                error = np.linalg.norm(actual - state.target_pos)
                print(f"   Target: [{state.target_pos[0]*100:.1f}, {state.target_pos[1]*100:.1f}, {state.target_pos[2]*100:.1f}] cm")
                print(f"   Actual: [{actual[0]*100:.1f}, {actual[1]*100:.1f}, {actual[2]*100:.1f}] cm (error: {error*100:.2f}cm)")
            else:
                print("   ✗ IK FAILED - target unreachable! Reverting.")
                state.target_pos = old_target

        except (KeyboardInterrupt, EOFError):
            print("\n→ Quitting...")
            state.running = False
            break
        except Exception as e:
            print(f"Error: {e}")

    # Cleanup
    viewer_thread.join(timeout=2.0)
    print("\nTeleoperation ended.\n")

if __name__ == "__main__":
    main()
