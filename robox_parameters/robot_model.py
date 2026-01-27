import json
import numpy as np
# scipy is imported lazily in inverse_kinematics to avoid version conflicts
# from utils import get_config_path
import utils
from typing import Tuple, Optional, List, Dict, Callable
import os

class Link:
    """
    Represents a robot link with DH parameters and physical properties.
    All units are SI: meters, radians, kilograms, kg⋅m².
    """
    def __init__(self, joint_type, d, a, alpha, mass, inertia_tensor, 
                 center_of_mass, theta_offset=0, actuator_id=None, joint_limits=None):
        """
        Initialize a robot link.
        
        :param joint_type: Type of joint ('revolute' or 'prismatic')
        :param d: Link offset [m]
        :param a: Link length [m]
        :param alpha: Link twist [rad]
        :param mass: Link mass [kg]
        :param inertia_tensor: 3x3 inertia tensor [kg⋅m²]
        :param center_of_mass: Center of mass position [m]
        :param theta_offset: Joint offset angle [rad]
        :param actuator_id: ID of associated actuator (if any)
        """
        self.joint_type = joint_type
        self.d = float(d)
        self.a = float(a)
        self.alpha = float(alpha)
        self.theta_offset = float(theta_offset)
        self.mass = float(mass)
        self.inertia_tensor = np.array(inertia_tensor, dtype=float)
        self.center_of_mass = np.array(center_of_mass, dtype=float)
        self.actuator_id = actuator_id
        self.joint_limits = joint_limits
        
        # State variables (updated during motion)
        self.current_position = 0.0    # [rad] or [m]
        self.current_velocity = 0.0    # [rad/s] or [m/s]
        self.current_acceleration = 0.0 # [rad/s²] or [m/s²]
        self.current_torque = 0.0      # [Nm]

class Robot:
    """
    Represents a robot manipulator with multiple links.
    Handles kinematics, dynamics, and actuator integration.
    All units are SI: meters, radians, kilograms, Newtons.
    """
    def __init__(self, name):
        """
        Initialize a robot.
        
        :param name: Robot name
        """
        self.name = name
        self.links = []
        self.coulomb_coeff = []
        self.viscous_coeff = []
        self.stiction_coeff = []
        self.stiction_velocity_threshold = []
        self.gravity_vector = np.array([0, 0, -9.81])
        self.joint_limits = []

    @classmethod
    def from_config(cls, config_file):
        """
        Create a robot instance from a configuration file.
        
        :param config_file: Path to robot configuration JSON file
        :return: Robot instance
        """
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', config_file)
        # config_path = utils.get_config_path(config_file)
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        robot = cls(config['robot_name'])
        
        # Load links
        for link_params in config['links']:
            robot.add_link(
                joint_type=link_params['joint_type'],
                d=link_params['d'],
                a=link_params['a'],
                alpha=link_params['alpha'],
                mass=link_params['mass'],
                inertia_tensor=link_params['inertia_tensor'],
                center_of_mass=link_params['center_of_mass'],
                theta_offset=link_params.get('theta_offset', 0),
                actuator_id=link_params.get('actuator_id', None),
                joint_limits=link_params.get('joint_limits')
            )
            

            #TODO:IMPORTANT Load joint limits through robot_parameter file, modify that file for this
            if 'joint_limits' in link_params:
                robot.joint_limits.append(link_params['joint_limits'])
            else:
                robot.joint_limits.append({
                    'position': {'min': -np.pi/2, 'max': np.pi/2},
                    'velocity': {'min': -np.pi, 'max': np.pi},
                    'acceleration': {'min': -10.0, 'max': 10.0},
                    'torque': {'min': -100.0, 'max': 100.0}
                })
        
        # Load friction parameters
        if 'friction_parameters' in config:
            robot.set_friction_parameters(
                config['friction_parameters']['coulomb_coefficients'],
                config['friction_parameters']['viscous_coefficients'],
                config['friction_parameters']['stiction_coefficients'],
                config['friction_parameters']['stiction_velocity_thresholds']
            )
        
        # Load gravity vector
        if 'gravity_vector' in config:
            robot.set_gravity_vector(config['gravity_vector'])
        
        return robot

    def add_link(self, joint_type, d, a, alpha, mass, inertia_tensor, 
                 center_of_mass, theta_offset=0, actuator_id=None, joint_limits=None):
        """Add a link to the robot."""
        self.links.append(Link(
            joint_type, d, a, alpha, mass, inertia_tensor,
            center_of_mass, theta_offset, actuator_id, joint_limits
        ))

    def set_friction_parameters(self, coulomb, viscous, stiction, stiction_threshold):
        """Set friction parameters for all joints."""
        self.coulomb_coeff = np.array(coulomb, dtype=float)
        self.viscous_coeff = np.array(viscous, dtype=float)
        self.stiction_coeff = np.array(stiction, dtype=float)
        self.stiction_velocity_threshold = np.array(stiction_threshold, dtype=float)

    def set_gravity_vector(self, gravity_vector):
        """Set gravity vector in world frame [m/s²]."""
        self.gravity_vector = np.array(gravity_vector, dtype=float)

    def forward_kinematics(self, joint_angles):
        """
        Compute forward kinematics for the robot.
        
        :param joint_angles: List of joint variables [rad] for revolute, [m] for prismatic
        :return: 4x4 homogeneous transformation matrix of end-effector
        :raises ValueError: If joint_angles length doesn't match number of links
        """
        if len(joint_angles) != len(self.links):
            raise ValueError(f"Expected {len(self.links)} joint angles, got {len(joint_angles)}")
            
        T = np.eye(4)
        for link, q in zip(self.links, joint_angles):
            if link.joint_type == "revolute":
                theta = q + link.theta_offset
                d = link.d
            else:  # prismatic
                theta = link.theta_offset
                d = q + link.d
            
            T = T @ self.dh_transform(theta, d, link.a, link.alpha)
        return T

    #def inverse_kinematics(self, position, orientation, initial_guess=None, tolerance=1e-6, max_iter=100):
        """
        Compute inverse kinematics numerically.
        
        :param position: Desired end-effector position [m]
        :param orientation: Desired end-effector orientation (3x3 rotation matrix)
        :param initial_guess: Initial joint angles [rad]
        :param tolerance: Convergence tolerance
        :param max_iter: Maximum iterations
        :return: Joint angles [rad] that achieve desired pose
        """
   
    def inverse_kinematics(self, position, orientation, initial_guess=None, tolerance=1e-6, max_iter=100):
        # Lazy import scipy to avoid version conflicts when not using IK
        from scipy.optimize import least_squares

        def objective(q):
            T = self.forward_kinematics(q)
            current_position = T[:3, 3]
            current_orientation = T[:3, :3]
            position_error = position - current_position
            orientation_error = self.orientation_error(current_orientation, orientation)
            return np.concatenate([position_error, orientation_error])

        if initial_guess is None:
            initial_guess = np.zeros(len(self.links))

        # Get joint limits for each link
        lower_bounds = []
        upper_bounds = []
        for link in self.links:
            if hasattr(link, 'joint_limits') and link.joint_limits is not None:
                lower_bounds.append(link.joint_limits['position']['min'])
                upper_bounds.append(link.joint_limits['position']['max'])
            else:
                # Default limits if not specified
                lower_bounds.append(-np.pi/2)
                upper_bounds.append(np.pi/2)

        bounds = (np.array(lower_bounds), np.array(upper_bounds))

        # Solve with joint limits as bounds
        result = least_squares(
            objective,
            initial_guess,
            bounds=bounds,
            ftol=tolerance,
            max_nfev=max_iter,
            method='trf'  # Trust Region Reflective algorithm, works well with bounds
        )

        if not result.success:
            print("Warning: Inverse kinematics may not have converged")

        return result.x

    def jacobian(self, joint_angles):
        """
        Compute the geometric Jacobian of the robot.
        
        :param joint_angles: Current joint angles [rad]
        :return: 6xn Jacobian matrix [m/rad] for linear components, [1] for angular
        """
        if len(joint_angles) != len(self.links):
            raise ValueError(f"Expected {len(self.links)} joint angles, got {len(joint_angles)}")
            
        J = np.zeros((6, len(self.links)))
        T = np.eye(4)
        
        for i, (link, theta) in enumerate(zip(self.links, joint_angles)):
            if link.joint_type == "revolute":
                z = T[:3, 2]  # z-axis of current frame
                p = self.forward_kinematics(joint_angles)[:3, 3] - T[:3, 3]  # vector to end-effector
                J[:3, i] = np.cross(z, p)  # linear velocity component
                J[3:, i] = z  # angular velocity component
            else:  # prismatic
                z = T[:3, 2]
                J[:3, i] = z  # linear velocity component
                J[3:, i] = 0  # no angular velocity for prismatic joint
            
            Ti = self.dh_transform(theta, link.d, link.a, link.alpha)
            T = T @ Ti
        
        return J

    @staticmethod
    def dh_transform(theta, d, a, alpha):
        """
        Compute DH transformation matrix.
        
        :param theta: Joint angle [rad]
        :param d: Link offset [m]
        :param a: Link length [m]
        :param alpha: Link twist [rad]
        :return: 4x4 homogeneous transformation matrix
        """
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)

        return np.array([
            [ct, -st*ca, st*sa, a*ct],
            [st, ct*ca, -ct*sa, a*st],
            [0, sa, ca, d],
            [0, 0, 0, 1]
        ])

    @staticmethod
    def orientation_error(current, desired):
        """
        Compute orientation error between two rotation matrices.
        
        :param current: Current orientation (3x3 rotation matrix)
        :param desired: Desired orientation (3x3 rotation matrix)
        :return: 3D orientation error vector [rad]
        """
        error_matrix = desired @ current.T - np.eye(3)
        return np.array([error_matrix[2, 1], error_matrix[0, 2], error_matrix[1, 0]]) / 2

    def apply_joint_limits(self, joint_angles):
        """
        Apply joint limits to given joint angles.
        
        :param joint_angles: Joint angles to check [rad]
        :return: Joint angles clipped to limits [rad]
        """
        return np.array([
            np.clip(angle, limit['position']['min'], limit['position']['max'])
            for angle, limit in zip(joint_angles, self.joint_limits)
        ])

    # Actuator integration methods
    def get_actuator_ids(self):
        """Get list of actuator IDs used in the robot."""
        return [link.actuator_id for link in self.links if link.actuator_id is not None]

    def update_joint_states(self, actuator_controller):
        """
        Update joint states from actuator readings.
        
        :param actuator_controller: ActuatorController instance
        """
        for link in self.links:
            if link.actuator_id is not None:
                # Get raw values and convert to SI units
                raw_pos = actuator_controller.get_position(link.actuator_id)
                raw_vel = actuator_controller.get_velocity(link.actuator_id)
                raw_curr = actuator_controller.get_current(link.actuator_id)

                # Convert to SI units
                link.current_position = (
                    actuator_controller.convert_position_from_raw(link.actuator_id, raw_pos) - 
                    link.theta_offset
                )
                link.current_velocity = actuator_controller.convert_velocity_from_raw(
                    link.actuator_id, raw_vel
                )
                link.current_torque = actuator_controller.convert_current_to_torque(
                    link.actuator_id, raw_curr
                )

    def command_joint_positions(self, actuator_controller, joint_positions):
        """
        Command joint positions.
        
        :param actuator_controller: ActuatorController instance
        :param joint_positions: Desired joint positions [rad]
        :raises ValueError: If joint_positions length doesn't match number of links
        """
        if len(joint_positions) != len(self.links):
            raise ValueError(f"Expected {len(self.links)} joint positions, got {len(joint_positions)}")

        # Apply joint limits
        joint_positions = self.apply_joint_limits(joint_positions)

        for link, angle in zip(self.links, joint_positions):
            if link.actuator_id is not None:
                actuator_value = actuator_controller.convert_joint_angle_to_raw(
                    link.actuator_id, angle + link.theta_offset
                )
                actuator_controller.set_position(link.actuator_id, actuator_value)

    def command_joint_velocities(self, actuator_controller, joint_velocities):
        """
        Command joint velocities.
        
        :param actuator_controller: ActuatorController instance
        :param joint_velocities: Desired joint velocities [rad/s]
        """
        if len(joint_velocities) != len(self.links):
            raise ValueError(f"Expected {len(self.links)} joint velocities, got {len(joint_velocities)}")

        for link, velocity in zip(self.links, joint_velocities):
            if link.actuator_id is not None:
                actuator_value = actuator_controller.convert_velocity_to_raw(
                    link.actuator_id, velocity
                )
                actuator_controller.set_velocity(link.actuator_id, actuator_value)

    def command_joint_torques(self, actuator_controller, joint_torques):
        """
        Command joint torques.
        
        :param actuator_controller: ActuatorController instance
        :param joint_torques: Desired joint torques [Nm]
        """
        if len(joint_torques) != len(self.links):
            raise ValueError(f"Expected {len(self.links)} joint torques, got {len(joint_torques)}")

        for link, torque in zip(self.links, joint_torques):
            if link.actuator_id is not None:
                current = actuator_controller.convert_torque_to_current(
                    link.actuator_id, torque
                )
                actuator_controller.set_current(link.actuator_id, current)

    def get_joint_states(self, actuator_controller):
        """
        Get current joint states.
        
        :param actuator_controller: ActuatorController instance
        :return: Dictionary with positions [rad], velocities [rad/s], and torques [Nm]
        """
        self.update_joint_states(actuator_controller)
        
        return {
            'positions': [link.current_position for link in self.links],
            'velocities': [link.current_velocity for link in self.links],
            'torques': [link.current_torque for link in self.links]
        }
    
# =============================================================================
# KEYBOARD TELEOPERATION HELPER FUNCTION
# =============================================================================

def keyboard_teleop(
    send_command: Callable[[Dict], None],
    repeat_interval: float = 0.08,
    mapping: Optional[Dict[str, Dict]] = None,
):
    """
    Keyboard teleoperation helper (single-function API).

    Calling this function RETURNS two callables:
        start() -> start keyboard teleop
        stop()  -> stop keyboard teleop

    Usage:
        start, stop = keyboard_teleop(send_command)
        start()
        ...
        stop()

    Args:
        send_command: Callable that receives a command dict when keys are pressed
        repeat_interval: Seconds between repeated calls while key is held (default 0.08)
        mapping: Optional dict mapping key names to command dicts

    Returns:
        (start_func, stop_func): Tuple of functions to start/stop keyboard listening

    Example:
        def my_command_handler(cmd):
            print(f"Received: {cmd}")

        start, stop = keyboard_teleop(my_command_handler)
        start()  # Begin listening for keyboard
        # ... do work ...
        stop()   # Stop listening
    """
    import threading
    import time

    try:
        from pynput import keyboard
    except Exception as e:
        raise RuntimeError("pynput is required for keyboard teleop") from e

    DEFAULT_MAPPING = {
        'w': {'type': 'delta', 'joint': 'y', 'value': 0.01},
        's': {'type': 'delta', 'joint': 'y', 'value': -0.01},
        'a': {'type': 'delta', 'joint': 'x', 'value': -0.01},
        'd': {'type': 'delta', 'joint': 'x', 'value': 0.01},
        'Key.up': {'type': 'delta', 'joint': 'y', 'value': 0.01},
        'Key.down': {'type': 'delta', 'joint': 'y', 'value': -0.01},
        'Key.left': {'type': 'delta', 'joint': 'x', 'value': -0.01},
        'Key.right': {'type': 'delta', 'joint': 'x', 'value': 0.01},
        'q': {'type': 'stop'},
    }

    key_mapping = mapping or dict(DEFAULT_MAPPING)

    running = False
    pressed = set()
    threads = {}
    lock = threading.Lock()
    listener = None

    def _key_id(key) -> str:
        try:
            if hasattr(key, 'char') and key.char is not None:
                return key.char
        except Exception:
            pass
        return f'Key.{key.name}' if hasattr(key, 'name') else str(key)

    def _loop(kid: str):
        while True:
            with lock:
                if not running or kid not in pressed:
                    break
                cmd = key_mapping.get(kid)
            if cmd:
                try:
                    send_command(cmd)
                except Exception:
                    pass
            time.sleep(repeat_interval)

    def _on_press(key):
        kid = _key_id(key)
        with lock:
            if kid in pressed or not running:
                return
            pressed.add(kid)
            t = threading.Thread(target=_loop, args=(kid,), daemon=True)
            threads[kid] = t
            t.start()

    def _on_release(key):
        kid = _key_id(key)
        with lock:
            pressed.discard(kid)

    def start():
        nonlocal running, listener
        if running:
            return
        running = True
        listener = keyboard.Listener(
            on_press=_on_press,
            on_release=_on_release
        )
        listener.start()

    def stop():
        nonlocal running, listener
        with lock:
            running = False
            pressed.clear()
        if listener:
            try:
                listener.stop()
            except Exception:
                pass
            listener = None

    return start, stop

# Example usage
if __name__ == "__main__":

    # Create a sample configuration file
    sample_config = {
        "robot_name": "My Robot",
        "links": [
            {"theta": 0, "d": 0, "a": 0, "alpha": np.pi/2, "mass": 1.0, "inertia_tensor": [[1,0,0],[0,1,0],[0,0,1]], "center_of_mass": [0,0,0]},
            {"theta": 0, "d": 0, "a": 0.5, "alpha": 0, "mass": 1.0, "inertia_tensor": [[1,0,0],[0,1,0],[0,0,1]], "center_of_mass": [0,0,0]},
            {"theta": 0, "d": 0, "a": 0.5, "alpha": 0, "mass": 1.0, "inertia_tensor": [[1,0,0],[0,1,0],[0,0,1]], "center_of_mass": [0,0,0]}
        ]
    }

    # Save the sample configuration to a file
    with open('sample_robot_config.json', 'w') as f:
        json.dump(sample_config, f)

    # Create the robot from the configuration file
    robot = Robot.from_config('sample_robot_config.json')

    # Test forward kinematics
    joint_angles = [0, np.pi/4, -np.pi/4]
    end_effector_pose = robot.forward_kinematics(joint_angles)
    print("End effector pose:")
    print(end_effector_pose)

    # Test inverse kinematics
    desired_position = end_effector_pose[:3, 3]
    desired_orientation = end_effector_pose[:3, :3]
    calculated_joint_angles = robot.inverse_kinematics(desired_position, desired_orientation)
    print("\nCalculated joint angles:")
    print(calculated_joint_angles)

    # Test Jacobian
    J = robot.jacobian(joint_angles)
    print("\nJacobian:")
    print(J)
