"""
Cartesian Impedance Control - Corrected Version

Uses scene.xml (with robot.xml) which has:
- Torque limits (ctrlrange)
- Joint damping
- Friction

This makes the control more stable and easier to tune.

Impedance Control Law:
    F = K * (x_des - x) - D * xdot           (Cartesian spring-damper)
    tau = J^T * F + gravity_comp             (Joint torques)
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
# End-effector body (robot.xml doesn't have ee_site)
# -------------------------------------------------
EE_BODY = model.body("gripper_link").id

# -------------------------------------------------
# Target Cartesian position (meters)
# -------------------------------------------------
x_des = np.array([-0.15, 0.10, 0.0])  # 3D target (X, Y, Z)

# -------------------------------------------------
# Impedance Control Parameters
# -------------------------------------------------
# Since actuators have small limits (±0.2 Nm), use moderate gains
# The XML already has joint damping, so we need less control damping

K_val = 50.0     # Stiffness (N/m)
D_val = 5.0      # Damping (Ns/m)

K = np.diag([K_val, K_val, 0.0])    # Stiffness matrix (XY plane only)
D = np.diag([D_val, D_val, 0.0])    # Damping matrix

# -------------------------------------------------
# Additional joint-space damping (optional, XML already has some)
# -------------------------------------------------
D_joint = np.array([0.02, 0.02, 0.02])  # Small extra damping

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

print("=" * 65)
print("CARTESIAN IMPEDANCE CONTROL (Corrected - uses scene.xml)")
print("=" * 65)
print(f"Initial EE position: ({x_init[0]:.4f}, {x_init[1]:.4f}, {x_init[2]:.4f}) m")
print(f"Target EE position:  ({x_des[0]:.4f}, {x_des[1]:.4f}, {x_des[2]:.4f}) m")
print(f"Distance to target:  {np.linalg.norm((x_des - x_init)[:2]):.4f} m")
print("-" * 65)
print(f"Stiffness K: {K_val} N/m")
print(f"Damping D:   {D_val} Ns/m")
print(f"Joint damping: {D_joint}")
print("-" * 65)
print("Note: XML has ctrlrange limits and joint damping built-in")
print("=" * 65)

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
        J = get_jacobian(model, data, EE_BODY)  # Shape: (3, nv)

        # --- End-effector velocity: xdot = J * qdot ---
        xdot = J @ data.qvel[:model.nv]

        # --- Position error ---
        error = x_des - x
        error_norm = np.linalg.norm(error[:2])  # XY error

        # --- Impedance control: F = K * error - D * xdot ---
        F = K @ error - D @ xdot

        # --- Joint torques: tau = J^T * F ---
        tau_impedance = J.T @ F

        # --- Gravity compensation ---
        # qfrc_bias contains gravity + Coriolis forces
        tau_gravity = data.qfrc_bias[:model.nv].copy()

        # --- Extra joint damping ---
        tau_joint_damping = -D_joint * qdot

        # --- Total torque ---
        tau = tau_impedance + tau_gravity + tau_joint_damping

        # --- Apply torques (XML will clip to ctrlrange automatically) ---
        for i in range(model.nu):
            data.ctrl[i] = tau[i]

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
            q = np.array([data.qpos[j1_qpos], data.qpos[j2_qpos], data.qpos[j3_qpos]])
            q_deg = np.rad2deg(q)
            vel_norm = np.linalg.norm(xdot[:2])

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
x_final = data.xpos[EE_BODY].copy()
final_error = np.linalg.norm((x_des - x_final)[:2])
print(f"Final EE Position: ({x_final[0]:.4f}, {x_final[1]:.4f}) m")
print(f"Target Position:   ({x_des[0]:.4f}, {x_des[1]:.4f}) m")
print(f"Final Error:       {final_error*1000:.2f} mm")
if converged:
    print(f"Convergence Time:  {converge_time:.2f} s")
else:
    print("Did NOT converge within simulation time.")
print("=" * 65)
