"""
Cartesian Impedance Control with External Disturbance

Based on cartesian_impedance_control_corrected.py with added disturbance capability.

Uses scene.xml (with robot.xml) which has:
- Torque limits (ctrlrange)
- Joint damping
- Friction

External Disturbance Types:
1. "cartesian" - Apply force to end-effector body (Fx, Fy, Fz)
2. "joint_torque" - Apply torque disturbance to specific joint

Impedance Control Law:
    F = K * (x_des - x) - D * xdot
    tau = J^T * F + gravity_comp
"""

import numpy as np
import mujoco
import mujoco.viewer
import os
import time

# -------------------------------------------------
# Load model (with limited actuators + damping)
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(BASE_DIR, "..", "xmls", "scene.xml")

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

print(f"Number of actuators: {model.nu}")
print(f"Number of joints (nv): {model.nv}")
print(f"Timestep: {model.opt.timestep} s")

# -------------------------------------------------
# Joint addresses
# -------------------------------------------------
j1_qpos = model.joint("robot_joint1").qposadr[0]
j2_qpos = model.joint("robot_joint2").qposadr[0]
j3_qpos = model.joint("robot_joint3").qposadr[0]

j1_qvel = model.joint("robot_joint1").dofadr[0]
j2_qvel = model.joint("robot_joint2").dofadr[0]
j3_qvel = model.joint("robot_joint3").dofadr[0]

# -------------------------------------------------
# End-effector body
# -------------------------------------------------
EE_BODY = model.body("gripper_link").id

# Body IDs for each link (for applying forces to specific links)
LINK_BODIES = {
    0: model.body("robot_link1").id,   # Link 1
    1: model.body("robot_link2").id,   # Link 2
    2: model.body("gripper_link").id,  # Link 3 (gripper/end-effector)
}

# -------------------------------------------------
# Target Cartesian position (meters)
# -------------------------------------------------
x_des = np.array([0.05, 0.10, 0.0])  # 3D target (X, Y, Z) - reachable from initial config

# -------------------------------------------------
# Impedance Control Parameters
# -------------------------------------------------
K_val = 200.0    # Stiffness (N/m) - increased for better tracking
D_val = 20.0     # Damping (Ns/m) - increased proportionally

K = np.diag([K_val, K_val, 0.0])    # Stiffness matrix (XY plane only)
D = np.diag([D_val, D_val, 0.0])    # Damping matrix

# Additional joint-space damping
D_joint = np.array([0.02, 0.02, 0.02])

# -------------------------------------------------
# EXTERNAL DISTURBANCE PARAMETERS
# -------------------------------------------------
# Disturbance time window
DIST_START = 1.5    # Time to start disturbance (s)
DIST_END = 4.0      # Time to end disturbance (s)

# Choose disturbance type: "cartesian" or "joint_torque"
DISTURBANCE_TYPE = "cartesian"

# --- Cartesian force disturbance ---
# Applied to end-effector body
F_EXT = np.array([1.0, 0.0, 0.0])  # External force [Fx, Fy, Fz] in Newtons (increased)

# --- Joint torque disturbance ---
DISTURB_JOINT = 1           # Joint index (0=joint1, 1=joint2, 2=joint3)
TAU_DISTURBANCE = 0.1       # Torque disturbance magnitude (Nm)
                            # Note: Keep small due to ctrlrange limits

# -------------------------------------------------
# Helper: Get Jacobian
# -------------------------------------------------
def get_jacobian(model, data, body_id):
    """Compute positional Jacobian for a body."""
    Jp = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, Jp, None, body_id)
    return Jp

# -------------------------------------------------
# Reset simulation
# -------------------------------------------------
mujoco.mj_resetData(model, data)
data.qpos[:] = 0.0
data.qvel[:] = 0.0
mujoco.mj_forward(model, data)

# Get initial position
x_init = data.xpos[EE_BODY].copy()

print("=" * 70)
print("CARTESIAN IMPEDANCE CONTROL WITH EXTERNAL DISTURBANCE")
print("=" * 70)
print(f"Initial EE position: ({x_init[0]:.4f}, {x_init[1]:.4f}, {x_init[2]:.4f}) m")
print(f"Target EE position:  ({x_des[0]:.4f}, {x_des[1]:.4f}, {x_des[2]:.4f}) m")
print(f"Distance to target:  {np.linalg.norm((x_des - x_init)[:2]):.4f} m")
print("-" * 70)
print(f"Stiffness K: {K_val} N/m")
print(f"Damping D:   {D_val} Ns/m")
print("-" * 70)
print(f"Disturbance type:   {DISTURBANCE_TYPE}")
print(f"Disturbance window: {DIST_START}s to {DIST_END}s")
if DISTURBANCE_TYPE == "cartesian":
    print(f"Cartesian force:    F = {F_EXT} N (applied to end-effector)")
else:
    print(f"Joint torque:       τ = {TAU_DISTURBANCE} Nm (applied to joint {DISTURB_JOINT + 1})")
print("=" * 70)

# -------------------------------------------------
# Simulation
# -------------------------------------------------
dt = model.opt.timestep
sim_time = 0.0
max_sim_time = 15.0

# Convergence tracking
error_threshold = 0.005  # 5mm
converged = False
converge_time = None

with mujoco.viewer.launch_passive(model, data) as viewer:

    step_count = 0

    while viewer.is_running() and sim_time < max_sim_time:

        # --- Current end-effector position ---
        x = data.xpos[EE_BODY].copy()

        # --- Current joint velocities ---
        qdot = np.array([data.qvel[j1_qvel], data.qvel[j2_qvel], data.qvel[j3_qvel]])

        # --- Compute Jacobian ---
        J = get_jacobian(model, data, EE_BODY)

        # --- End-effector velocity: xdot = J * qdot ---
        xdot = J @ data.qvel[:model.nv]

        # --- Position error ---
        error = x_des - x
        error_norm = np.linalg.norm(error[:2])

        # --- Impedance control: F = K * error - D * xdot ---
        F = K @ error - D @ xdot

        # --- Joint torques: tau = J^T * F ---
        tau_impedance = J.T @ F

        # --- Gravity compensation ---
        tau_gravity = data.qfrc_bias[:model.nv].copy()

        # --- Extra joint damping ---
        tau_joint_damping = -D_joint * qdot

        # --- Total torque ---
        tau = tau_impedance + tau_gravity + tau_joint_damping

        # --- Apply torques via actuators ---
        data.ctrl[0] = tau[0]
        data.ctrl[1] = tau[1]
        data.ctrl[2] = tau[2]

        # -------------------------------------------------
        # Apply external disturbance at end-effector
        # (Applied AFTER ctrl, BEFORE mj_step - same as v3)
        # -------------------------------------------------
        data.xfrc_applied[EE_BODY][:] = 0.0   # clear previous forces

        disturbance_now = DIST_START <= sim_time <= DIST_END
        if disturbance_now:
            data.xfrc_applied[EE_BODY, 0] = F_EXT[0]  # Fx
            data.xfrc_applied[EE_BODY, 1] = F_EXT[1]  # Fy
            data.xfrc_applied[EE_BODY, 2] = F_EXT[2]  # Fz
            data.xfrc_applied[EE_BODY, 3] = 0.0       # Tx
            data.xfrc_applied[EE_BODY, 4] = 0.0       # Ty
            data.xfrc_applied[EE_BODY, 5] = 0.0       # Tz

        # --- Step simulation ---
        mujoco.mj_step(model, data)
        sim_time += dt
        step_count += 1

        # --- Check convergence (only when no disturbance) ---
        if not disturbance_now:
            if not converged and error_norm < error_threshold:
                converged = True
                converge_time = sim_time
                print(f"\n*** CONVERGED at t={converge_time:.2f}s "
                      f"(error < {error_threshold*1000:.1f}mm) ***\n")

        # --- Print every 0.5 seconds ---
        if step_count % int(0.5 / dt) == 0:
            q = np.array([data.qpos[j1_qpos], data.qpos[j2_qpos], data.qpos[j3_qpos]])
            q_deg = np.rad2deg(q)
            vel_norm = np.linalg.norm(xdot[:2])
            dist_str = " [DISTURBANCE]" if disturbance_now else ""

            print(f"t={sim_time:5.2f}s | "
                  f"EE=({x[0]:7.4f}, {x[1]:7.4f})m | "
                  f"err={error_norm*1000:6.2f}mm | "
                  f"vel={vel_norm:.4f}m/s{dist_str}")

        viewer.sync()
        time.sleep(dt)

# -------------------------------------------------
# Final summary
# -------------------------------------------------
print("\n" + "=" * 70)
print("SIMULATION COMPLETE")
print("=" * 70)
x_final = data.xpos[EE_BODY].copy()
final_error = np.linalg.norm((x_des - x_final)[:2])
print(f"Final EE Position: ({x_final[0]:.4f}, {x_final[1]:.4f}) m")
print(f"Target Position:   ({x_des[0]:.4f}, {x_des[1]:.4f}) m")
print(f"Final Error:       {final_error*1000:.2f} mm")
if converged:
    print(f"Convergence Time:  {converge_time:.2f} s")
else:
    print("Did NOT converge within simulation time.")
print("-" * 70)
print("Disturbance Summary:")
print(f"  Type: {DISTURBANCE_TYPE}")
print(f"  Window: {DIST_START}s - {DIST_END}s")
if DISTURBANCE_TYPE == "cartesian":
    print(f"  Force: {F_EXT} N")
else:
    print(f"  Joint {DISTURB_JOINT + 1} torque: {TAU_DISTURBANCE} Nm")
print("=" * 70)
