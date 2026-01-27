"""
Cartesian Impedance Control for 3-DOF Planar Robot (CORRECTED VERSION)

FIXES APPLIED:
1. Gravity direction fixed in XML (now -Z instead of X)
2. End-effector site position corrected to actual gripper tip
3. Added gravity compensation using MuJoCo's qfrc_bias
4. Removed meaningless null-space control (3-DOF robot with 3D target = no redundancy)
5. Using 2D (XY) Jacobian for planar robot control
6. Proper handling of the robot's actual configuration
7. Added workspace reachability info display

Impedance Control Law:
    F = K * (x_des - x) - D * xdot
    tau = J^T * F + g(q)   (gravity compensation added)

"""

import numpy as np
import mujoco
import mujoco.viewer
import os
import time

# -------------------------------------------------
# Load model (with actuators)
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(BASE_DIR, "..", "xmls", "scene_with_actuators.xml")

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

print(f"Number of actuators: {model.nu}")
print(f"Number of joints (nv): {model.nv}")
print(f"Timestep: {model.opt.timestep} s")

# -------------------------------------------------
# Joint addresses
# -------------------------------------------------
j1_qpos = model.joint("robot_joint1").qposadr.item()
j2_qpos = model.joint("robot_joint2").qposadr.item()
j3_qpos = model.joint("gripper_joint").qposadr.item()

j1_qvel = model.joint("robot_joint1").dofadr.item()
j2_qvel = model.joint("robot_joint2").dofadr.item()
j3_qvel = model.joint("gripper_joint").dofadr.item()

# -------------------------------------------------
# End-effector site
# -------------------------------------------------
EE_SITE = model.site("ee_site").id
EE_BODY = model.site("ee_site").bodyid

# -------------------------------------------------
# Target Cartesian position (meters) - in XY plane
# User confirmed this is reachable
# -------------------------------------------------
x_des = np.array([0.15, 0.15])  # 2D target (X, Y) in meters

# -------------------------------------------------
# Impedance Control Parameters (2D - XY plane)
# -------------------------------------------------
# For this lightweight robot (~0.15 kg), we need lower gains
# Critical damping: D = 2 * sqrt(K * m)
# For K=50 N/m, m=0.15 kg: D_critical = 2*sqrt(50*0.15) = 5.5 Ns/m
# We use slightly overdamped (ζ > 1) for stability

# K_val = 50.0     # Stiffness (N/m) - reduced for lightweight robot
# D_val = 15.0     # Damping (Ns/m) - overdamped for stability
K_val = 55.0     # Stiffness (N/m) - adjusted for better performance
D_val = 10.0     # Damping (Ns/m) - overdamped for

K = np.diag([K_val, K_val])    # Stiffness matrix (N/m)
D = np.diag([D_val, D_val])    # Damping matrix (Ns/m)

# -------------------------------------------------
# External disturbance parameters (Cartesian)
# -------------------------------------------------
DIST_START=5.0    # Time to start disturbance (s)
DIST_END=7.0      # Time to end disturbance (s)
F_EXT = np.array([8.0, 0.0, 0.0])  # External force disturbance (N) in XY


# -------------------------------------------------
# Joint limits for safety (from robot_parameters.json)
# -------------------------------------------------
JOINT_LIMITS = {
    'q1': (-3.14, 3.14),   # Joint 1: +/-180 deg
    'q2': (-1.57, 1.57),   # Joint 2: +/-90 deg
    'q3': (-1.57, 1.57),   # Joint 3: +/-90 deg
}

# -------------------------------------------------
# Helper function: Get 2D Jacobian (XY components only)
# -------------------------------------------------
def get_jacobian_2d(model, data, site_id):
    """
    Compute the 2D positional Jacobian (XY plane only).
    Returns shape (2, nv) where nv is number of joint velocities.
    """
    Jp_full = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, Jp_full, None, site_id)
    # Extract only X and Y rows for planar control
    return Jp_full[:2, :]

# -------------------------------------------------
# Helper function: Clamp torques for safety
# -------------------------------------------------
def clamp_torque(tau, max_torque=10.0):
    """Clamp torques to prevent excessive values."""
    return np.clip(tau, -max_torque, max_torque)

# -------------------------------------------------
# Reset simulation - Start from initial configuration
# -------------------------------------------------
mujoco.mj_resetData(model, data)

# Set initial joint positions (qpos = 0 means joints at their reference angles)
# The reference angles in XML are: q1_ref=2.35, q2_ref=-0.03, q3_ref=-0.08
# Setting qpos=0 puts joints AT those reference angles, not at "straight"
data.qpos[:] = 0.0
data.qvel[:] = 0.0
mujoco.mj_forward(model, data)

# Get initial EE position
x_init = data.site_xpos[EE_SITE].copy()
x_init_2d = x_init[:2]

print("=" * 65)
print("CARTESIAN IMPEDANCE CONTROL (CORRECTED VERSION)")
print("=" * 65)
print(f"Initial EE Position: ({x_init[0]:.4f}, {x_init[1]:.4f}, {x_init[2]:.4f}) m")
print(f"Target EE Position:  ({x_des[0]:.4f}, {x_des[1]:.4f}) m (XY plane)")
print(f"Distance to target:  {np.linalg.norm(x_des - x_init_2d):.4f} m")
print("-" * 65)
print(f"Stiffness K: {K[0,0]:.1f} N/m")
print(f"Damping D:   {D[0,0]:.1f} Ns/m")
print("-" * 65)
print("Initial joint angles (qpos=0 means at XML reference angles):")
q_init = np.array([data.qpos[j1_qpos], data.qpos[j2_qpos], data.qpos[j3_qpos]])
print(f"  q1 = {np.rad2deg(q_init[0]):.2f} deg")
print(f"  q2 = {np.rad2deg(q_init[1]):.2f} deg")
print(f"  q3 = {np.rad2deg(q_init[2]):.2f} deg")
print("=" * 65)

# -------------------------------------------------
# Simulation
# -------------------------------------------------
dt = model.opt.timestep
sim_time = 0.0
max_sim_time = 30.0  # Maximum simulation time

# For convergence detection
error_threshold = 0.002  # 2mm
converged = False
converge_time = None

with mujoco.viewer.launch_passive(model, data) as viewer:

    step_count = 0

    while viewer.is_running() and sim_time < max_sim_time:

        # --- Current end-effector position (3D from MuJoCo) ---
        x_full = data.site_xpos[EE_SITE].copy()
        x = x_full[:2]  # Extract XY for 2D control

        # --- Current joint positions and velocities ---
        q = np.array([data.qpos[j1_qpos], data.qpos[j2_qpos], data.qpos[j3_qpos]])
        qdot = np.array([data.qvel[j1_qvel], data.qvel[j2_qvel], data.qvel[j3_qvel]])

        # --- Compute 2D Jacobian (XY plane) ---
        J = get_jacobian_2d(model, data, EE_SITE)  # Shape: (2, 3)

        # --- End-effector velocity: xdot = J * qdot ---
        xdot = J @ qdot  # Shape: (2,)

        # --- Position error ---
        error = x_des - x  # Shape: (2,)
        error_norm = np.linalg.norm(error)

        # --- Impedance force: F = K * error - D * xdot ---
        # This creates a spring-damper behavior in Cartesian space
        F = K @ error - D @ xdot  # Shape: (2,)

        # --- Joint torques from Cartesian impedance: tau = J^T * F ---
        tau_impedance = J.T @ F  # Shape: (3,)

        # --- Gravity compensation ---
        # qfrc_bias contains Coriolis, centrifugal, and gravitational forces
        # Adding this compensates for gravity, making the robot "weightless"
        tau_gravity = data.qfrc_bias[:model.nv].copy()

        # --- Total torque ---
        tau = tau_impedance + tau_gravity

        # --- Clamp torques for safety ---
        tau = clamp_torque(tau, max_torque=5.0)

        # --- Apply torques via actuators ---
        data.ctrl[0] = tau[0]
        data.ctrl[1] = tau[1]
        data.ctrl[2] = tau[2]

        # -------------------------------------------------
        # Apply external disturbance at end-effector
        # -------------------------------------------------
        data.xfrc_applied[EE_BODY][:] = 0.0   # clear previous forces

        if DIST_START <= sim_time <= DIST_END:
            data.xfrc_applied[EE_BODY, 0] = F_EXT[0]  # Fx
            data.xfrc_applied[EE_BODY, 1] = F_EXT[1]  # Fy
            data.xfrc_applied[EE_BODY, 2] = F_EXT[2]  # Fz
            data.xfrc_applied[EE_BODY, 3] = 0.0       # Tx
            data.xfrc_applied[EE_BODY, 4] = 0.0       # Ty
            data.xfrc_applied[EE_BODY, 5] = 0.0       # Tz
            # data.xfrc_applied[EE_BODY, :] = 0.0


        # --- Step simulation ---
        mujoco.mj_step(model, data)
        sim_time += dt
        step_count += 1

        # --- Check convergence ---
        if not converged and error_norm < error_threshold:
            converged = True
            converge_time = sim_time
            print(f"\n*** CONVERGED at t={converge_time:.2f}s (error < {error_threshold*1000:.1f}mm) ***\n")

        # --- Print every 0.5 seconds ---
        if step_count % int(0.5 / dt) == 0:
            q_deg = np.rad2deg(q)
            vel_norm = np.linalg.norm(xdot)

            print(f"t={sim_time:5.2f}s | "
                  f"EE=({x[0]:7.4f}, {x[1]:7.4f})m | "
                  f"err={error_norm*1000:6.2f}mm | "
                  f"vel={vel_norm:.4f}m/s | "
                  f"q=({q_deg[0]:6.1f}, {q_deg[1]:6.1f}, {q_deg[2]:6.1f})deg")

        viewer.sync()
        time.sleep(dt)

# -------------------------------------------------
# Final summary
# -------------------------------------------------
print("\n" + "=" * 65)
print("SIMULATION COMPLETE")
print("=" * 65)
x_final = data.site_xpos[EE_SITE][:2]
final_error = np.linalg.norm(x_des - x_final)
print(f"Final EE Position: ({x_final[0]:.4f}, {x_final[1]:.4f}) m")
print(f"Target Position:   ({x_des[0]:.4f}, {x_des[1]:.4f}) m")
print(f"Final Error:       {final_error*1000:.2f} mm")
if converged:
    print(f"Convergence Time:  {converge_time:.2f} s")
else:
    print("Did NOT converge within simulation time.")
print("=" * 65)
