"""
Joint Impedance Control
Pure joint-space spring–damper with bias compensation

Control Law:
    tau = Kq (q_des - q) - Dq qdot + bias(q, qdot)
"""

import numpy as np
import mujoco
import mujoco.viewer
import os
import sys
import time
import json

# -------------------------------------------------
# Path setup
# -------------------------------------------------
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
# Robot model (for consistency)
# -------------------------------------------------
robot = Robot("Planar_3DOF")

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
    if 'joint_limits' in link_data:
        robot.joint_limits.append(link_data['joint_limits'])

if 'gravity_vector' in robot_params:
    robot.set_gravity_vector(robot_params['gravity_vector'])

JOINT_LIMITS = np.array([
    [lim['position']['min'], lim['position']['max']]
    for lim in robot.joint_limits
])

print(f"Robot: {robot.name}")
print(f"Number of links: {len(robot.links)}")
print(f"Link lengths: {[l.a for l in robot.links]} m")
print("Joint limits (rad):", JOINT_LIMITS)

# -------------------------------------------------
# Joint indices
# -------------------------------------------------
j1_qpos = model.joint("robot_joint1").qposadr.item()
j2_qpos = model.joint("robot_joint2").qposadr.item()
j3_qpos = model.joint("gripper_joint").qposadr.item()

j1_qvel = model.joint("robot_joint1").dofadr.item()
j2_qvel = model.joint("robot_joint2").dofadr.item()
j3_qvel = model.joint("gripper_joint").dofadr.item()

# -------------------------------------------------
# Desired joint configuration (ABSOLUTE)
# -------------------------------------------------
q_des = np.array([0.3, 0.3, 0.6])  # radians

# -------------------------------------------------
# Joint impedance gains
# -------------------------------------------------
Kq = np.diag([8.0, 8.0, 18.0])   # Joint 3 needs higher stiffness to hold position
Dq = np.diag([0.1, 0.1, 0.3])  # Damping ratio tuned for stability


# -------------------------------------------------
# Torque saturation (SAFE for these gains)
# -------------------------------------------------
def clamp_torque(tau, max_torque=8.0):
    return np.clip(tau, -max_torque, max_torque)

# -------------------------------------------------
# Reset simulation
# -------------------------------------------------
mujoco.mj_resetData(model, data)

# Start near desired configuration (IMPORTANT)
data.qpos[j1_qpos] = q_des[0]
data.qpos[j2_qpos] = q_des[1]
data.qpos[j3_qpos] = q_des[2]
data.qvel[:] = 0.0

mujoco.mj_forward(model, data)

print("=" * 65)
print("JOINT IMPEDANCE CONTROL (WITH BIAS COMPENSATION)")
print("=" * 65)
print(f"Desired joint config (rad): {q_des}")
print(f"Kq diag: {np.diag(Kq)} Nm/rad")
print(f"Dq diag: {np.diag(Dq)} Nm/(rad/s)")
print("=" * 65)

# -------------------------------------------------
# Simulation loop
# -------------------------------------------------
dt = model.opt.timestep
sim_time = 0.0
max_sim_time = 15.0

with mujoco.viewer.launch_passive(model, data) as viewer:
    step_count = 0

    while viewer.is_running() and sim_time < max_sim_time:

        q = np.array([
            data.qpos[j1_qpos],
            data.qpos[j2_qpos],
            data.qpos[j3_qpos]
        ])

        qdot = np.array([
            data.qvel[j1_qvel],
            data.qvel[j2_qvel],
            data.qvel[j3_qvel]
        ])

        # -------------------------------
        # Joint Impedance Control
        # -------------------------------

        # Applies joint impedance control on joint 3 as well

        # q_error = q_des - q
        # tau_impedance = Kq @ q_error - Dq @ qdot

        # # Bias compensation (gravity + Coriolis + friction)
        # tau_bias = data.qfrc_bias[:model.nv].copy()

        # # tau = tau_impedance + tau_bias
        # tau = tau_impedance 
        # tau = clamp_torque(tau, 8.0)

        # data.ctrl[:] = tau

        # Applies joint impedance control on joint 1 and 2 only
        # -------------------------------
        # Joint-level Impedance Control
        # -------------------------------
        q_error = q_des - q

        tau = np.zeros(3)

        # Joint 1 & 2: true impedance (spring + damper)
        tau[0] = Kq[0, 0] * q_error[0] - Dq[0, 0] * qdot[0]
        tau[1] = Kq[1, 1] * q_error[1] - Dq[1, 1] * qdot[1]

        # Joint 3: low stiffness + higher damping (for low-inertia joint)
        tau[2] = Kq[2, 2] * q_error[2] - Dq[2, 2] * qdot[2]

        # Torque safety limits (lower limit for low-inertia joint 3)
        tau[:2] = np.clip(tau[:2], -8.0, 8.0)
        tau[2] = np.clip(tau[2], -1.0, 1.0)

        data.ctrl[:] = tau


        mujoco.mj_step(model, data)
        sim_time += dt
        step_count += 1

        if step_count % int(0.5 / dt) == 0:
            q_deg = np.rad2deg(q)
            q_err_norm = np.linalg.norm(q_error)
            print(f"t={sim_time:.2f}s | |q_err|={q_err_norm:.4f} rad | "
                  f"q=({q_deg[0]:.1f}, {q_deg[1]:.1f}, {q_deg[2]:.1f}) deg")

        viewer.sync()
        time.sleep(dt)

# -------------------------------------------------
# Final report
# -------------------------------------------------
q_final = np.array([
    data.qpos[j1_qpos],
    data.qpos[j2_qpos],
    data.qpos[j3_qpos]
])

print("\n" + "=" * 65)
print("SIMULATION COMPLETE")
print("=" * 65)
print("Final joint configuration (rad):", q_final)
print("Desired joint configuration (rad):", q_des)
print(f"Final joint error norm: {np.linalg.norm(q_des - q_final):.6f} rad")
print("=" * 65)
