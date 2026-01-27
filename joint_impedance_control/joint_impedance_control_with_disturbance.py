"""
Joint Impedance Control with External Disturbance

Based on joint_impedance_control_v1.py with added external disturbance capability
similar to cartesian_impedance_control_v3_disturbance.py

Control Law:
    tau = Kq (q_des - q) - Dq qdot + bias(q, qdot)

External Disturbance:
    - Can be applied as torque disturbance to any joint
    - Can also be applied as Cartesian force to end-effector body
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

# End-effector body (for Cartesian disturbance option)
try:
    EE_SITE = model.site("ee_site").id
    EE_BODY = model.site("ee_site").bodyid
    HAS_EE_SITE = True
except:
    HAS_EE_SITE = False
    print("Warning: ee_site not found")

# Body IDs for each link (for applying Cartesian force to specific joints)
LINK_BODIES = {
    0: model.body("robot_link1").id,   # Joint 1 body
    1: model.body("robot_link2").id,   # Joint 2 body
    2: model.body("gripper_link").id,  # Joint 3 body (gripper)
}
print(f"Link body IDs: {LINK_BODIES}")

# -------------------------------------------------
# Desired joint configuration (ABSOLUTE)
# -------------------------------------------------
q_des = np.array([0.6, 0.6, 0.3])  # radians

# -------------------------------------------------
# Joint impedance gains
# -------------------------------------------------
Kq = np.diag([8.0, 8.0, 0.0])
Dq = np.diag([0.1, 0.1, 0.1])

# -------------------------------------------------
# EXTERNAL DISTURBANCE PARAMETERS
# -------------------------------------------------
# Disturbance time window
DIST_START = 5.0    # Time to start disturbance (s)
DIST_END = 7.0      # Time to end disturbance (s)

# Choose disturbance type: "joint_torque" or "cartesian"
DISTURBANCE_TYPE = "cartesian"

# Joint torque disturbance (Nm) - applied directly to joints
# Set which joint to disturb (0, 1, or 2) and the torque magnitude
DISTURB_JOINT = 1           # Joint index (0=joint1, 1=joint2, 2=joint3)
TAU_DISTURBANCE = 0.5       # Torque disturbance magnitude (Nm)

# Cartesian force disturbance (N) - applied to specified joint's body
DISTURB_LINK = 1            # Link index for Cartesian force (0=link1, 1=link2, 2=gripper)
F_EXT = np.array([15.0, 0.0, 0.0])  # External force [Fx, Fy, Fz] along X direction

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
print("JOINT IMPEDANCE CONTROL WITH EXTERNAL DISTURBANCE")
print("=" * 65)
print(f"Desired joint config (rad): {q_des}")
print(f"Desired joint config (deg): {np.rad2deg(q_des)}")
print(f"Kq diag: {np.diag(Kq)} Nm/rad")
print(f"Dq diag: {np.diag(Dq)} Nm/(rad/s)")
print("-" * 65)
print(f"Disturbance type: {DISTURBANCE_TYPE}")
print(f"Disturbance window: {DIST_START}s to {DIST_END}s")
if DISTURBANCE_TYPE == "joint_torque":
    print(f"Disturbed joint: Joint {DISTURB_JOINT + 1}")
    print(f"Torque disturbance: {TAU_DISTURBANCE} Nm")
else:
    print(f"Disturbed link: Link {DISTURB_LINK + 1} (Joint {DISTURB_LINK + 1} body)")
    print(f"Cartesian force (along X): {F_EXT} N")
print("=" * 65)

# -------------------------------------------------
# Simulation loop
# -------------------------------------------------
dt = model.opt.timestep
sim_time = 0.0
max_sim_time = 15.0

# Track disturbance status
disturbance_active = False

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
        # Joint-level Impedance Control
        # -------------------------------
        q_error = q_des - q

        tau = np.zeros(3)

        # Joint 1 & 2: true impedance (spring + damper)
        tau[0] = Kq[0, 0] * q_error[0] - Dq[0, 0] * qdot[0]
        tau[1] = Kq[1, 1] * q_error[1] - Dq[1, 1] * qdot[1]

        # Joint 3: damping only (NO stiffness)
        tau[2] = -Dq[2, 2] * qdot[2]

        # -------------------------------
        # Apply External Disturbance
        # -------------------------------
        disturbance_now = DIST_START <= sim_time <= DIST_END

        if DISTURBANCE_TYPE == "joint_torque":
            # Apply torque disturbance directly to specified joint
            if disturbance_now:
                tau[DISTURB_JOINT] += TAU_DISTURBANCE
                if not disturbance_active:
                    print(f"\n>>> DISTURBANCE STARTED at t={sim_time:.2f}s "
                          f"(Joint {DISTURB_JOINT + 1}, τ={TAU_DISTURBANCE} Nm)")
                    disturbance_active = True
            else:
                if disturbance_active and sim_time > DIST_END:
                    print(f">>> DISTURBANCE ENDED at t={sim_time:.2f}s\n")
                    disturbance_active = False

        elif DISTURBANCE_TYPE == "cartesian":
            # Apply Cartesian force to specified link's body (along X)
            disturb_body = LINK_BODIES[DISTURB_LINK]
            data.xfrc_applied[disturb_body][:] = 0.0  # Clear previous forces

            if disturbance_now:
                data.xfrc_applied[disturb_body, 0] = F_EXT[0]  # Fx (along X)
                data.xfrc_applied[disturb_body, 1] = F_EXT[1]  # Fy
                data.xfrc_applied[disturb_body, 2] = F_EXT[2]  # Fz
                data.xfrc_applied[disturb_body, 3:6] = 0.0     # No torque
                if not disturbance_active:
                    print(f"\n>>> DISTURBANCE STARTED at t={sim_time:.2f}s "
                          f"(Link {DISTURB_LINK + 1}, F_x={F_EXT[0]} N)")
                    disturbance_active = True
            else:
                if disturbance_active and sim_time > DIST_END:
                    print(f">>> DISTURBANCE ENDED at t={sim_time:.2f}s\n")
                    disturbance_active = False

        # -------------------------------
        # Torque safety limits
        # -------------------------------
        tau = clamp_torque(tau, 8.0)

        data.ctrl[:] = tau

        mujoco.mj_step(model, data)
        sim_time += dt
        step_count += 1

        if step_count % int(0.5 / dt) == 0:
            q_deg = np.rad2deg(q)
            q_err_norm = np.linalg.norm(q_error)
            dist_str = " [DISTURBANCE]" if disturbance_now else ""
            print(f"t={sim_time:.2f}s | |q_err|={q_err_norm:.4f} rad | "
                  f"q=({q_deg[0]:.1f}, {q_deg[1]:.1f}, {q_deg[2]:.1f}) deg{dist_str}")

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
print("Final joint configuration (deg):", np.rad2deg(q_final))
print("Desired joint configuration (rad):", q_des)
print("Desired joint configuration (deg):", np.rad2deg(q_des))
print(f"Final joint error norm: {np.linalg.norm(q_des - q_final):.6f} rad")
print(f"Final joint error norm: {np.rad2deg(np.linalg.norm(q_des - q_final)):.4f} deg")
print("=" * 65)
