from dynamixel_sdk import *
from esp_bridge import ESPActuatorController as ActuatorController
import numpy as np
import time

class RobotControlUtility:
    """
    Utility class for controlling 3R planar robot and calculating its kinematics.
    All angles in radians, distances in meters.
    """
    
    def __init__(self, actuator_config_file):
        self.actuator_controller = ActuatorController(actuator_config_file)
        self.l1 = 0.1
        self.l2 = 0.07
        self.l3 = 0.05
        self.actuator_ids = list(self.actuator_controller.actuators.keys())
        
    def enable_torque(self):
        for actuator_id in self.actuator_ids:
            self.actuator_controller.enable_torque(actuator_id)
            
    def disable_torque(self):
        for actuator_id in self.actuator_ids:
            self.actuator_controller.disable_torque(actuator_id)

    def set_joint_angles(self, angles):
        if len(angles) != 3:
            print("Must provide exactly 3 angles")
            return False
            
        success = True
        for actuator_id, angle in zip(self.actuator_ids, angles):
            time.sleep(0.2)
            target_raw = self.actuator_controller.relative_joint_angle_to_raw(
                actuator_id, angle
            )
            print(angle)
            if not self.actuator_controller.set_position(actuator_id, target_raw):
                time.sleep(0.2)
                success = False
                print(f"Failed to set position for joint {actuator_id}")
        return success

    def get_joint_angles(self):
        angles = []
        for actuator_id in self.actuator_ids:
            angle = self.actuator_controller.relative_joint_angle(actuator_id)
            angles.append(angle)
        return angles

    def forward_kinematics(self, angles=None):
        if angles is None:
            angles = self.get_joint_angles()
            
        theta1, theta2, theta3 = angles
        theta12 = theta1 + theta2
        theta123 = theta12 + theta3

        x = (
            self.l1 * np.cos(theta1)
            + self.l2 * np.cos(theta12)
            + self.l3 * np.cos(theta123)
        )
        y = (
            self.l1 * np.sin(theta1)
            + self.l2 * np.sin(theta12)
            + self.l3 * np.sin(theta123)
        )
        z = 0
        
        position = np.array([x, y, z])
        
        orientation = np.array(
            [
                [np.cos(theta123), -np.sin(theta123), 0],
                [np.sin(theta123), np.cos(theta123), 0],
                [0, 0, 1],
            ]
        )
        
        return position, orientation

    def print_status(self):
        current_angles = self.get_joint_angles()
        position, orientation = self.forward_kinematics(current_angles)
        
        print("\nCurrent Robot Status:")
        print("---------------------")
        print("Joint Angles (degrees):")
        for i, angle in enumerate(current_angles, 1):
            print(f"Joint {i}: {np.degrees(angle):.2f}°")
            
        print("\nEnd-Effector Position (meters):")
        print(f"X: {position[0]:.4f}")
        print(f"Y: {position[1]:.4f}")
        print(f"Z: {position[2]:.4f}")
        
        print("\nEnd-Effector Orientation:")
        print("Rotation Matrix:")
        for row in orientation:
            print(f"[{row[0]:7.4f} {row[1]:7.4f} {row[2]:7.4f}]")
            
        final_angle = np.degrees(np.arctan2(orientation[1, 0], orientation[0, 0]))
        print(f"\nFinal Orientation Angle: {final_angle:.2f}°")

def main():
    robot = RobotControlUtility("actuator_config.json")
    
    try:
        robot.enable_torque()
        
        while True:
            print("\nRobot Control Menu:")
            print("1. Set joint angles")
            print("2. Get current status")
            print("3. Move to home position")
            print("4. Exit")
            
            choice = input("\nEnter choice (1-4): ")
            
            if choice == "1":
                try:
                    angles = []
                    for i in range(3):
                        angle_deg = float(input(f"Enter angle for joint {i+1} (degrees): "))
                        angles.append(np.radians(angle_deg))
                    if robot.set_joint_angles(angles):
                        print("Moving to requested position...")
                        time.sleep(2)
                        robot.print_status()
                    else:
                        print("Failed to set joint angles")
                except ValueError:
                    print("Invalid angle input. Please enter numbers only.")
                    
            elif choice == "2":
                robot.print_status()
                
            elif choice == "3":
                home_angles = [1.57, 0, 0]
                if robot.set_joint_angles(home_angles):
                    print("Moving to home position...")
                    time.sleep(2)
                    robot.print_status()
                else:
                    print("Failed to move to home position")
                    
            elif choice == "4":
                break
            else:
                print("Invalid choice. Please enter 1-4.")
                
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    finally:
        robot.disable_torque()
        print("Program terminated safely")

if __name__ == "__main__":
    main()
