import time
import logging
import threading
from typing import Optional, Dict, Any

class SerialConnectionManager:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, "initialized"):
            self.active_connections = {}
            self.connection_lock = threading.Lock()
            self.current_owner = None
            self.logger = logging.getLogger("SerialConnectionManager")
            self.initialized = True
    
    def request_connection(self, owner_id: str, actuator_controller) -> bool:
        with self.connection_lock:
            if self.current_owner is None:
                self.current_owner = owner_id
                self.active_connections[owner_id] = actuator_controller
                self.logger.info(f"Connection granted to {owner_id}")
                return True
            if self.current_owner == owner_id:
                self.active_connections[owner_id] = actuator_controller
                return True
            self.logger.warning(f"Connection denied to {owner_id}, currently owned by {self.current_owner}")
            return False
    
    def release_connection(self, owner_id: str) -> bool:
        with self.connection_lock:
            if self.current_owner == owner_id:
                if owner_id in self.active_connections:
                    controller = self.active_connections[owner_id]
                    self._safe_cleanup_controller(controller, owner_id)
                    del self.active_connections[owner_id]
                self.current_owner = None
                self.logger.info(f"Connection released by {owner_id}")
                return True
            else:
                self.logger.warning(f"Cannot release connection: {owner_id} is not the current owner")
                return False
    
    def force_release_all(self) -> None:
        with self.connection_lock:
            self.logger.info("Force releasing all connections")
            for owner_id, controller in self.active_connections.items():
                self._safe_cleanup_controller(controller, owner_id)
            self.active_connections.clear()
            self.current_owner = None
    
    def refresh_connection(self, owner_id: str) -> bool:
        with self.connection_lock:
            if self.current_owner != owner_id:
                self.logger.error(f"Refresh denied: {owner_id} is not the current owner")
                return False
            if owner_id not in self.active_connections:
                self.logger.error(f"No active connection found for {owner_id}")
                return False
            controller = self.active_connections[owner_id]
            return self._refresh_controller_connection(controller, owner_id)
    
    def _refresh_controller_connection(self, controller, owner_id: str) -> bool:
        try:
            self.logger.info(f"Refreshing connection for {owner_id}")
            if hasattr(controller, "restart_connection"):
                controller.restart_connection()
                self.logger.info(f"Connection restarted using restart_connection() for {owner_id}")
                return True
            if hasattr(controller, "close") and hasattr(controller, "connect"):
                controller.close()
                time.sleep(0.5)
                controller.connect()
                self.logger.info(f"Connection refreshed using close/connect for {owner_id}")
                return True
            if hasattr(controller, "serial") and controller.serial:
                port = controller.serial.port
                baudrate = controller.serial.baudrate
                controller.serial.close()
                time.sleep(0.5)
                controller.serial.open()
                self.logger.info(f"Serial connection refreshed for {owner_id}")
                return True
            if hasattr(controller, "__init__"):
                config_file = getattr(controller, "config_file", "actuator_config.json")
                controller.__init__(config_file)
                self.logger.info(f"Controller reinitialized for {owner_id}")
                return True
            self.logger.warning(f"No refresh method available for {owner_id}")
            return False
        except Exception as e:
            self.logger.error(f"Error refreshing connection for {owner_id}: {str(e)}")
            return False
    
    def _safe_cleanup_controller(self, controller, owner_id: str) -> None:
        try:
            self.logger.info(f"Cleaning up controller for {owner_id}")
            if hasattr(controller, "actuators"):
                for actuator_id in controller.actuators:
                    try:
                        controller.disable_torque(actuator_id)
                    except:
                        pass
            else:
                for motor_id in [1, 2, 3]:
                    try:
                        controller.disable_torque(motor_id)
                    except:
                        pass
            if hasattr(controller, "close"):
                controller.close()
            elif hasattr(controller, "serial") and controller.serial:
                controller.serial.close()
                controller.serial = None
            self.logger.info(f"Controller cleanup completed for {owner_id}")
        except Exception as e:
            self.logger.error(f"Error during controller cleanup for {owner_id}: {str(e)}")
    
    def get_current_owner(self) -> Optional[str]:
        return self.current_owner
    
    def is_connection_available(self) -> bool:
        return self.current_owner is None
    
    def get_connection_status(self) -> Dict[str, Any]:
        with self.connection_lock:
            return {
                "current_owner": self.current_owner,
                "active_connections": list(self.active_connections.keys()),
                "is_available": self.current_owner is None,
            }

def get_connection_manager() -> SerialConnectionManager:
    return SerialConnectionManager()

def request_robot_access(component_name: str, controller) -> bool:
    manager = get_connection_manager()
    return manager.request_connection(component_name, controller)

def release_robot_access(component_name: str) -> bool:
    manager = get_connection_manager()
    return manager.release_connection(component_name)

def refresh_robot_connection(component_name: str) -> bool:
    manager = get_connection_manager()
    return manager.refresh_connection(component_name)

def is_robot_available() -> bool:
    manager = get_connection_manager()
    return manager.is_connection_available()

def get_robot_owner() -> Optional[str]:
    manager = get_connection_manager()
    return manager.get_current_owner()

def force_release_robot() -> None:
    manager = get_connection_manager()
    manager.force_release_all()