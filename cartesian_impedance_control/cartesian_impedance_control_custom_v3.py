"""
Cartesian Impedance Control - Custom Jacobian Version 3
Uses Robot class from robot_model.py for FK, Jacobian, IK
WITHOUT joint damping

Control Law (Pure Cartesian Impedance):
    F = K * (x_des - x) - D * xdot
    tau = J^T * F + g(q)
"""

import numpy as np
import mujoco
import mujoco.viewer
import os
import sys
import time

# Add parent directory to path for importing robot_model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from robox_parameters.robot_model import Robot

# -------------------------------------------------
# Load MuJoCo model
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(BASE_DIR, "..", "xmls", "scene_with_actuators.xml")
PARAMS_PATH = os.path.join(BASE_DIR, "..", "robox_parameters", "robot_parameters.json")

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# -------------------------------------------------
# Create Robot from robot_model.py using JSON parameters
# -------------------------------------------------
robot = Robot("Planar_3DOF")

# Load parameters from JSON and create robot
import json
with open(PARAMS_PATH, 'r') as f:
    robot_params = json.load(f)

for link_data in robot_params['links']:
    robot.add_link(
        joint_type=link_data['joint_type'],
        d=link_data['d'],
        a=link_data['a'],
        alpha=link_data['alpha'],
        mass=link_data['mass'],
        inertia_tensor=link_data['inertia_tensor'],
        center_of_mass=link_data['center_of_mass'],
        theta_offset=link_data.get('theta_offset', 0),
        actuator_id=link_data.get('actuator_id'),
        joint_limits=link_data.get('joint_limits')
    )
    # Also add to robot's joint_limits list
    if 'joint_limits' in link_data:
        robot.joint_limits.append(link_data['joint_limits'])

# Set gravity vector from JSON
if 'gravity_vector' in robot_params:
    robot.set_gravity_vector(robot_params['gravity_vector'])

# Extract joint position limits as numpy array for easy access
JOINT_LIMITS = np.array([
    [lim['position']['min'], lim['position']['max']]
    for lim in robot.joint_limits
])

print(f"Robot: {robot.name}")
print(f"Number of links: {len(robot.links)}")
print(f"Link lengths: {robot.links[0].a}, {robot.links[1].a}, {robot.links[2].a} m")
print(f"Joint limits (rad): q1=[{JOINT_LIMITS[0,0]:.2f}, {JOINT_LIMITS[0,1]:.2f}], "
      f"q2=[{JOINT_LIMITS[1,0]:.2f}, {JOINT_LIMITS[1,1]:.2f}], "
      f"q3=[{JOINT_LIMITS[2,0]:.2f}, {JOINT_LIMITS[2,1]:.2f}]")

# -------------------------------------------------
# Joint addresses (MuJoCo)
# -------------------------------------------------
j1_qpos = model.joint("robot_joint1").qposadr.item()
j2_qpos = model.joint("robot_joint2").qposadr.item()
j3_qpos = model.joint("gripper_joint").qposadr.item()

j1_qvel = model.joint("robot_joint1").dofadr.item()
j2_qvel = model.joint("robot_joint2").dofadr.item()
j3_qvel = model.joint("gripper_joint").dofadr.item()

EE_SITE = model.site("ee_site").id

# -------------------------------------------------
# Target position
# -------------------------------------------------
x_des = np.array([0.15, 0.08])

# -------------------------------------------------
# Gains (Cartesian only - no joint damping)
# -------------------------------------------------
K = np.diag([55.0, 55.0])    # Cartesian stiffness (N/m)
D = np.diag([10.0, 10.0])    # Cartesian damping (Ns/m)

# -------------------------------------------------
# Helper functions using robot_model.py
# -------------------------------------------------
def get_custom_jacobian_2d(robot, q):
    """Get 2D Jacobian (XY only) using Robot.jacobian()"""
    J_full = robot.jacobian(q)  # 6x3 geometric Jacobian
    return J_full[:2, :]        # 2x3 (X, Y only)

def get_forward_kinematics(robot, q):
    """Get end-effector position using Robot.forward_kinematics()"""
    T = robot.forward_kinematics(q)  # 4x4 transformation matrix
    return T[:3, 3]  # Position (x, y, z)

def clamp_torque(tau, max_torque=5.0):
    return np.clip(tau, -max_torque, max_torque)

# -------------------------------------------------
# Reset
# -------------------------------------------------
mujoco.mj_resetData(model, data)
data.qpos[:] = 0.0
data.qvel[:] = 0.0
mujoco.mj_forward(model, data)

x_init = data.site_xpos[EE_SITE][:2].copy()

# Also compute FK using robot_model to verify
q_init = np.array([0.0, 0.0, 0.0])
fk_pos = get_forward_kinematics(robot, q_init)

print("=" * 65)
print("CUSTOM JACOBIAN V3 (Using robot_model.py - No Joint Damping)")
print("=" * 65)
print(f"Initial EE (MuJoCo): ({x_init[0]:.4f}, {x_init[1]:.4f}) m")
print(f"Initial EE (FK):     ({fk_pos[0]:.4f}, {fk_pos[1]:.4f}) m")
print(f"Target:              ({x_des[0]:.4f}, {x_des[1]:.4f}) m")
print(f"K={K[0,0]} N/m, D={D[0,0]} Ns/m")
print("=" * 65)

# -------------------------------------------------
# Simulation
# -------------------------------------------------
dt = model.opt.timestep
sim_time = 0.0
max_sim_time = 15.0

with mujoco.viewer.launch_passive(model, data) as viewer:
    step_count = 0

    while viewer.is_running() and sim_time < max_sim_time:
        q = np.array([data.qpos[j1_qpos], data.qpos[j2_qpos], data.qpos[j3_qpos]])
        qdot = np.array([data.qvel[j1_qvel], data.qvel[j2_qvel], data.qvel[j3_qvel]])

        x = data.site_xpos[EE_SITE][:2].copy()

        # Custom Jacobian from robot_model.py
        J = get_custom_jacobian_2d(robot, q)
        xdot = J @ qdot

        # Pure Cartesian Impedance Control
        error = x_des - x
        F = K @ error - D @ xdot
        tau_impedance = J.T @ F

        # Gravity compensation only (no joint damping)
        tau_gravity = data.qfrc_bias[:model.nv].copy()

        tau = tau_impedance + tau_gravity
        tau = clamp_torque(tau, 5.0)

        data.ctrl[:] = tau
        mujoco.mj_step(model, data)
        sim_time += dt
        step_count += 1

        if step_count % int(0.5 / dt) == 0:
            q_deg = np.rad2deg(q)
            print(f"t={sim_time:.2f}s | EE=({x[0]:.4f}, {x[1]:.4f}) | "
                  f"err={np.linalg.norm(error)*1000:.1f}mm | "
                  f"q=({q_deg[0]:.1f}, {q_deg[1]:.1f}, {q_deg[2]:.1f})deg")

        viewer.sync()
        time.sleep(dt)

print("\n" + "=" * 65)
print("SIMULATION COMPLETE")
print("=" * 65)
x_final = data.site_xpos[EE_SITE][:2]
print(f"Final: ({x_final[0]:.4f}, {x_final[1]:.4f}) m")
print(f"Target: ({x_des[0]:.4f}, {x_des[1]:.4f}) m")
print(f"Final Error: {np.linalg.norm(x_des - x_final)*1000:.2f} mm")
print("=" * 65)
