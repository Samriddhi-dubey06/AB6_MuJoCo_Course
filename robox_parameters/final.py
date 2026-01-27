import mujoco
import mujoco.viewer
import time
import os

# Absolute path to scene.xml
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(BASE_DIR, "scene.xml")

# Load MuJoCo model
model = mujoco.MjModel.from_xml_path(XML_PATH)

# Create simulation data
data = mujoco.MjData(model)

print("MuJoCo scene loaded successfully!")
print("Press ESC in viewer to close.")

# Launch viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)
