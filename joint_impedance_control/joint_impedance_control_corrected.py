"""
Joint Impedance Control (Corrected Version)

Uses scene.xml (with robot.xml) which has:
- Joint damping (0.05)
- Joint friction (0.1)
- Actuator torque limits (±5 Nm)

Control Law:
    tau = Kq (q_des - q) - Dq qdot + bias(q, qdot)

The robot will hold/track the desired joint configuration.
"""

import numpy as np
import mujoco
import mujoco.viewer
import os
import time

# -------------------------------------------------
# Load MuJoCo model (with damping + limits)
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(BASE_DIR, "..", "xmls", "scene.xml")

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

print(f"Number of actuators: {model.nu}")
print(f"Number of joints (nv): {model.nv}")
print(f"Timestep: {model.opt.timestep} s")

# -------------------------------------------------
# Joint indices (robot.xml uses robot_joint3, not gripper_joint)
# -------------------------------------------------
j1_qpos = model.joint("robot_joint1").qposadr[0]
j2_qpos = model.joint("robot_joint2").qposadr[0]
j3_qpos = model.joint("robot_joint3").qposadr[0]

j1_qvel = model.joint("robot_joint1").dofadr[0]
j2_qvel = model.joint("robot_joint2").dofadr[0]
j3_qvel = model.joint("robot_joint3").dofadr[0]

# -------------------------------------------------
# Desired joint configuration
# -------------------------------------------------
q_des_deg = np.array([45.0, 30.0, 15.0])  # degrees
q_des = np.deg2rad(q_des_deg)

# -------------------------------------------------
# Joint impedance gains
# -------------------------------------------------
# With XML damping (0.05) and friction (0.1), we can use higher gains
# These gains create a spring-damper behavior in joint space

Kq = np.array([50.0, 50.0, 20.0])    # Joint stiffness (Nm/rad)
Dq = np.array([5.0, 5.0, 2.0])       # Joint damping (Nm*s/rad)

print("=" * 65)
print("JOINT IMPEDANCE CONTROL (Corrected Version)")
print("=" * 65)
print(f"Desired joint config (deg): {q_des_deg}")
print(f"Desired joint config (rad): {q_des}")
print(f"Kq: {Kq} Nm/rad")
print(f"Dq: {Dq} Nm*s/rad")
print("-" * 65)
print("XML has built-in joint damping (0.05) and friction (0.1)")
print("Actuator limits: ±5.0 Nm (joints 1,2), ±2.0 Nm (joint 3)")
print("=" * 65)

# -------------------------------------------------
# Reset simulation and set initial pose
# -------------------------------------------------
mujoco.mj_resetData(model, data)

# Start at desired configuration
data.qpos[j1_qpos] = q_des[0]
data.qpos[j2_qpos] = q_des[1]
data.qpos[j3_qpos] = q_des[2]
data.qvel[:] = 0.0

mujoco.mj_forward(model, data)

# -------------------------------------------------
# Simulation parameters
# -------------------------------------------------
dt = model.opt.timestep
sim_time = 0.0
max_sim_time = 10.0

# Convergence tracking
converged = False
converge_time = None
error_threshold = 0.01  # 0.01 rad ≈ 0.57 degrees

# -------------------------------------------------
# Simulation loop
# -------------------------------------------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    step_count = 0

    while viewer.is_running() and sim_time < max_sim_time:

        # Get current joint positions and velocities
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

        # -------------------------------------------------
        # Joint Impedance Control
        # tau = Kq * (q_des - q) - Dq * qdot + gravity_comp
        # -------------------------------------------------
        q_error = q_des - q

        # Impedance torque (spring + damper)
        tau_impedance = Kq * q_error - Dq * qdot

        # Gravity/bias compensation
        tau_bias = data.qfrc_bias[:model.nv].copy()

        # Total torque
        tau = tau_impedance + tau_bias

        # Apply torques (XML will clip to ctrlrange)
        data.ctrl[0] = tau[0]
        data.ctrl[1] = tau[1]
        data.ctrl[2] = tau[2]

        # Step simulation
        mujoco.mj_step(model, data)
        sim_time += dt
        step_count += 1

        # Check convergence
        error_norm = np.linalg.norm(q_error)
        if not converged and error_norm < error_threshold:
            converged = True
            converge_time = sim_time
            print(f"\n*** HOLDING POSITION (error < {np.rad2deg(error_threshold):.1f} deg) "
                  f"at t={converge_time:.2f}s ***\n")

        # Print status every 0.5 seconds
        if step_count % int(0.5 / dt) == 0:
            q_deg = np.rad2deg(q)
            print(f"t={sim_time:.2f}s | error={np.rad2deg(error_norm):.2f} deg | "
                  f"q=({q_deg[0]:.1f}, {q_deg[1]:.1f}, {q_deg[2]:.1f}) deg | "
                  f"tau=({tau[0]:.2f}, {tau[1]:.2f}, {tau[2]:.2f}) Nm")

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

final_error = np.linalg.norm(q_des - q_final)

print("\n" + "=" * 65)
print("SIMULATION COMPLETE")
print("=" * 65)
print(f"Desired configuration (deg): {q_des_deg}")
print(f"Final configuration (deg):   {np.rad2deg(q_final)}")
print(f"Final error: {np.rad2deg(final_error):.2f} degrees")
if converged:
    print(f"Status: HOLDING POSITION (converged at t={converge_time:.2f}s)")
else:
    print("Status: Did NOT converge - check gains or target")
print("=" * 65)
