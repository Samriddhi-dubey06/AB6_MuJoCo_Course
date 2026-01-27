import os
import sys
import time
import logging
import subprocess
import json
from typing import Optional

def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_config_path(config_file):
    return os.path.join(get_project_root(), "config", config_file)

def get_control_table_path(model):
    return os.path.join(get_project_root(), "control_tables", f"{model}.json")

def ensure_directory_exists(path):
    os.makedirs(path, exist_ok=True)

def load_json_file(file_path):
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file: {file_path}")
        return None

def check_pd_gains(actuator_id, actuator_controller):
    try:
        addr_p_gain = actuator_controller.get_control_table_addr(actuator_id, "ADDR_POSITION_P_GAIN")
        addr_d_gain = actuator_controller.get_control_table_addr(actuator_id, "ADDR_POSITION_D_GAIN")

        p_gain, p_result, p_error = actuator_controller.packetHandler.read2byteTxRx(
            actuator_controller.portHandler,
            actuator_id,
            addr_p_gain,
        )

        d_gain, d_result, d_error = actuator_controller.packetHandler.read2byteTxRx(
            actuator_controller.portHandler,
            actuator_id,
            addr_d_gain,
        )

        if p_result != 0 or p_error != 0:
            print(
                f"Error reading P gain: "
                f"{actuator_controller.packetHandler.getTxRxResult(p_result)} "
                f"{actuator_controller.packetHandler.getRxPacketError(p_error)}"
            )
            return None

        if d_result != 0 or d_error != 0:
            print(
                f"Error reading D gain: "
                f"{actuator_controller.packetHandler.getTxRxResult(d_result)} "
                f"{actuator_controller.packetHandler.getRxPacketError(d_error)}"
            )
            return None

        model = actuator_controller.get_actuator_model(actuator_id)

        gains = {
            "actuator_id": actuator_id,
            "model": model,
            "position_p_gain": p_gain,
            "position_d_gain": d_gain,
        }

        print(f"\nGain values for {model} (ID: {actuator_id}):")
        print(f"Position P Gain: {p_gain}")
        print(f"Position D Gain: {d_gain}")

        return gains

    except Exception as e:
        print(f"Error checking gains: {str(e)}")
        return None

def check_all_actuator_gains(actuator_controller):
    all_gains = {}

    print("\nChecking current PD gains for all actuators...")

    for actuator_id in actuator_controller.actuators:
        gains = check_pd_gains(actuator_id, actuator_controller)
        if gains:
            all_gains[actuator_id] = gains

    return all_gains

def force_release_com_port(port: str = "COM4") -> bool:
    logging.info(f"Attempting to force release {port}")
    try:
        import serial
        try:
            ser = serial.Serial(port, 115200, timeout=0.1)
            ser.close()
            logging.info(f"{port} was available and has been safely closed")
            return True
        except Exception as e:
            logging.info(f"Could not open {port} directly: {str(e)}")
            pass

        if sys.platform == "win32":
            try:
                from serial.tools import list_ports
                ports = list(list_ports.comports())
                for p in ports:
                    if p.device == port:
                        logging.info(f"{port} is currently connected to: {p.description}")
                time.sleep(0.5)
                return True
            except Exception as e:
                logging.warning(f"Error checking port status: {str(e)}")
        return False
    except Exception as e:
        logging.error(f"Error during port release: {str(e)}")
        return False

def cleanup_hardware_resources():
    force_release_com_port("COM4")
    time.sleep(0.5)
    logging.info("Hardware resources cleanup completed")

if __name__ == "__main__":
    from esp_bridge import ESPActuatorController as ActuatorController

    actuator_controller = ActuatorController("actuator_config.json")
    gains = check_all_actuator_gains(actuator_controller)
    actuator_controller.portHandler.closePort()
