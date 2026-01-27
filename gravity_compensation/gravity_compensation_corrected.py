"""
Gravity Compensation + PD Control (Corrected Version)

Uses scene_vertical.xml with updated robot_vertical.xml that has:
- Joint damping (0.05)
- Joint friction (0.1)
- Actuator torque limits (±5 Nm)

Control Law:
    tau = qfrc_bias + Kp*(q_des - q) - Kd*qdot

The vertical robot will hold its position against gravity.
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import os

# -------------------------------------------------
# Load vertical robot model
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(BASE_DIR, "..", "xmls", "scene_vertical.xml")

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

print("=" * 65)
print("GRAVITY COMPENSATION + PD (Corrected Version)")
print("=" * 65)
print(f"Gravity: {model.opt.gravity}")
print(f"Number of actuators: {model.nu}")
print(f"Number of DOFs: {model.nv}")

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
# Desired configuration (where robot will hold)
# -------------------------------------------------
q_des_deg = np.array([45.0, 30.0, 15.0])  # degrees
q_des = np.deg2rad(q_des_deg)

# -------------------------------------------------
# PD Gains (tuned for robot with XML damping)
# -------------------------------------------------
# With XML damping (0.05) and friction (0.1), we can use higher Kp
# Rule of thumb: Kp should be high enough to overcome gravity
# For a vertical robot, gravity torques can be significant

Kp = np.array([50.0, 50.0, 20.0])    # Higher stiffness for position holding
Kd = np.array([5.0, 5.0, 2.0])       # Moderate damping (XML adds more)

print(f"\nDesired configuration (deg): {q_des_deg}")
print(f"Desired configuration (rad): {q_des}")
print(f"Kp: {Kp}")
print(f"Kd: {Kd}")
print("\nRobot should HOLD this position against gravity.")
print("XML has built-in joint damping (0.05) and friction (0.1)")
print("=" * 65)

# -------------------------------------------------
# Reset simulation and set initial pose
# -------------------------------------------------
mujoco.mj_resetData(model, data)
data.qpos[j1_qpos] = q_des[0]
data.qpos[j2_qpos] = q_des[1]
data.qpos[j3_qpos] = q_des[2]
data.qvel[:] = 0.0
mujoco.mj_forward(model, data)

# -------------------------------------------------
# Simulation parameters
# -------------------------------------------------
dt = float(model.opt.timestep)
sim_time = 0.0
max_sim_time = 10.0

# Convergence tracking
converged = False
converge_time = None
drift_threshold = 0.01  # 0.01 rad ≈ 0.57 degrees

# -------------------------------------------------
# Viewer + control loop
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
        # Gravity Compensation + PD Control
        # tau = gravity_comp + Kp*(q_des - q) - Kd*qdot
        # -------------------------------------------------
        tau_gravity = data.qfrc_bias[:model.nv].copy()

        q_error = q_des - q
        tau_pd = Kp * q_error - Kd * qdot

        tau = tau_gravity[:model.nu] + tau_pd

        # Apply torques (XML will clip to ctrlrange)
        data.ctrl[:model.nu] = tau

        # Step simulation
        mujoco.mj_step(model, data)
        sim_time += dt
        step_count += 1

        # Check convergence
        drift = np.linalg.norm(q - q_des)
        if not converged and drift < drift_threshold:
            converged = True
            converge_time = sim_time
            print(f"\n*** HOLDING POSITION (drift < {np.rad2deg(drift_threshold):.1f} deg) "
                  f"at t={converge_time:.2f}s ***\n")

        # Print status every 1 second
        if step_count % int(1.0 / dt) == 0:
            q_deg = np.rad2deg(q)
            print(f"t={sim_time:.1f}s | drift={np.rad2deg(drift):.2f} deg | "
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

final_drift = np.linalg.norm(q_final - q_des)

print("\n" + "=" * 65)
print("SIMULATION COMPLETE")
print("=" * 65)
print(f"Desired configuration (deg): {q_des_deg}")
print(f"Final configuration (deg):   {np.rad2deg(q_final)}")
print(f"Final drift: {np.rad2deg(final_drift):.2f} degrees")
if converged:
    print(f"Status: HOLDING POSITION (converged at t={converge_time:.2f}s)")
else:
    print("Status: Did NOT converge - robot drifted")
print("=" * 65)
