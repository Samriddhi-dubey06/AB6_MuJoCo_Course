# AB6 MuJoCo Course Simulation

A comprehensive robotics learning project featuring a 3-DOF planar robot manipulator. This repository demonstrates various robot control techniques, kinematics, and dynamics using the MuJoCo physics simulation engine.

## Overview

This project serves as an educational platform for learning fundamental and advanced robotics concepts through practical implementations:

- Forward and Inverse Kinematics
- Gravity Compensation
- Joint and Cartesian Impedance Control
- PID Control
- Trajectory Planning and Tracking
- Workspace Analysis
- Teleoperation

## Project Structure

```
AB6_MuJoCo_Course_Simulation/
├── forward_kinematics/          # FK computation methods
├── inverse_kinematics/          # IK computation methods
├── gravity_compensation/        # Gravity compensation control
├── gravity_comp/                # Alternative gravity compensation
├── joint_impedance_control/     # Joint-space impedance control
├── cartesian_impedance_control/ # Cartesian-space impedance control
├── cartesian_teleop/            # Cartesian teleoperation
├── pid control/                 # PID-based control (joint & cartesian)
├── velocity_control/            # Task-space velocity control
├── trajectory_tracking/         # Trajectory planning & tracking
├── workspace_identification/    # Workspace analysis
├── robox_parameters/            # Robot configuration & model
├── xmls/                        # MuJoCo model files (XML & meshes)
└── utils/                       # Utility functions
```

## Requirements

### Core Dependencies

```bash
pip install mujoco numpy
```

### Optional Dependencies

```bash
pip install PyKDL    # Advanced kinematics library
pip install scipy    # Trajectory planning
pip install matplotlib  # Visualization
```

## Getting Started

### 1. Clone the Repository

```bash
git clone <repository-url>
cd AB6_MuJoCo_Course_Simulation
```

### 2. Install Dependencies

```bash
pip install mujoco numpy
```

### 3. Run a Simple Test

```bash
python test_vertical_robot.py
```

## Usage Examples

### Forward Kinematics

```bash
python forward_kinematics/FK_maths.py      # Mathematical derivation
python forward_kinematics/FK_kdl.py        # Using PyKDL library
python forward_kinematics/FK_mujoco_inbuilt.py  # MuJoCo built-in
```

### Inverse Kinematics

```bash
python inverse_kinematics/IK_maths.py      # Mathematical derivation
python inverse_kinematics/IK_kdl.py        # Using PyKDL library
```

### Control Techniques

```bash
# Gravity Compensation
python gravity_compensation/gravity_compensation_corrected.py

# Joint Impedance Control
python joint_impedance_control/joint_impedance_control.py

# Cartesian Impedance Control
python cartesian_impedance_control/cartesian_impedance_control_v2.py

# PID Control
python "pid control/joint/joint_pid_control.py"
python "pid control/cartesian/cartesian_pid_control.py"
```

### Trajectory Tracking

```bash
python trajectory_tracking/quintic_trajectory_straight.py
python trajectory_tracking/quintic_trajectory_square.py
```

### Workspace Analysis

```bash
python workspace_identification/workspace_identification.py
```

## Robot Specifications

The 3-DOF planar robot manipulator has the following characteristics:

| Parameter | Value |
|-----------|-------|
| Degrees of Freedom | 3 |
| Link Masses | 1.0 - 1.5 kg |
| Joint Limits | ±π radians |
| Velocity Limits | ±2π rad/s |
| Actuator Torque Limits | ±5 Nm |

### Control Parameters (Default)

```python
Kp = [50.0, 50.0, 20.0]  # Position gains
Kd = [5.0, 5.0, 2.0]     # Damping gains
```

## MuJoCo Models

The `xmls/` directory contains several robot configurations:

- `robot.xml` - Base planar 3-DOF robot
- `robot_vertical.xml` - Vertical configuration (for gravity effects)
- `robot_with_actuators.xml` - Includes motor/actuator models
- `scene.xml` - Complete scene with robot and environment

## Features

| Feature | Description |
|---------|-------------|
| Forward Kinematics | 4 methods (math, custom, KDL, MuJoCo) |
| Inverse Kinematics | 3 methods (math, custom, KDL) |
| Gravity Compensation | PD control with gravity bias |
| Joint Impedance | Spring-damper in joint space |
| Cartesian Impedance | Spring-damper in task space |
| PID Control | Joint and Cartesian variants |
| Trajectory Planning | Quintic polynomial trajectories |
| Workspace Analysis | Sampling-based identification |
| Disturbance Testing | Controller robustness evaluation |

## Simulation Controls

When running scripts with MuJoCo's viewer:

- **Mouse drag** - Rotate camera view
- **Scroll** - Zoom in/out
- **Space** - Pause/resume simulation
- **Close window** - Exit gracefully

## Code Structure

All control scripts follow a common pattern:

```python
import mujoco
import numpy as np

# Load model
model = mujoco.MjModel.from_xml_path("xmls/scene.xml")
data = mujoco.MjData(model)

# Run simulation with viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # Compute control law
        tau = compute_control(data)
        data.ctrl[:] = tau

        # Step simulation
        mujoco.mj_step(model, data)
        viewer.sync()
```

## Configuration

Robot parameters are stored in `robox_parameters/robot_parameters.json`:

- DH parameters for each link
- Physical properties (mass, inertia, center of mass)
- Joint limits (position and velocity)
- Friction coefficients (Coulomb, viscous, stiction)

## License

This project is for educational purposes.

## Acknowledgments

- [MuJoCo](https://mujoco.org/) - Physics simulation engine
- [PyKDL](https://github.com/orocos/orocos_kinematics_dynamics) - Kinematics library
