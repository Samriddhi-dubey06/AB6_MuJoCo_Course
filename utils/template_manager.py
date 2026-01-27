import os
import shutil
import subprocess
import tempfile
import logging
from typing import Optional, Tuple

class TemplateManager:
    def __init__(self, template_dir="src/utils"):
        self.template_dir = template_dir
        self.workspace_dir = tempfile.mkdtemp(prefix="challenge_workspace_")
        logging.basicConfig(
            filename="template_manager.log",
            level=logging.INFO,
            format="%(asctime)s:%(levelname)s:%(message)s",
        )

    def setup_challenge_workspace(self, challenge_id: str) -> Tuple[bool, str]:
        try:
            challenge_workspace = os.path.join(self.workspace_dir, challenge_id)
            os.makedirs(challenge_workspace, exist_ok=True)
            template_path = os.path.join(self.template_dir, challenge_id.lower())
            if not os.path.exists(template_path):
                logging.error(f"Template directory not found: {template_path}")
                return False, ""
            for file in os.listdir(template_path):
                src = os.path.join(template_path, file)
                dst = os.path.join(challenge_workspace, file)
                shutil.copy2(src, dst)
            logging.info(f"Challenge workspace setup at: {challenge_workspace}")
            return True, challenge_workspace
        except Exception as e:
            logging.error(f"Error setting up workspace: {str(e)}")
            return False, ""

    def get_editor_command(self) -> str:
        if "VISUAL" in os.environ:
            return os.environ["VISUAL"]
        elif "EDITOR" in os.environ:
            return os.environ["EDITOR"]
        elif os.name == "nt":
            return "notepad"
        else:
            return "nano"

    def open_template_in_editor(self, template_path: str) -> bool:
        editor = self.get_editor_command()
        try:
            subprocess.run([editor, template_path], check=True)
            logging.info(f"Template opened in editor: {editor}")
            return True
        except subprocess.SubprocessError as e:
            logging.error(f"Error opening editor: {str(e)}")
            return False

    def validate_solution(self, solution_path: str, challenge_id: str) -> bool:
        try:
            if not os.path.exists(solution_path):
                return False
            with open(solution_path, "r") as f:
                content = f.read().strip()
                if not content:
                    return False
            try:
                compile(content, solution_path, "exec")
            except SyntaxError:
                return False
            return True
        except Exception as e:
            logging.error(f"Error validating solution: {str(e)}")
            return False

    def cleanup(self, workspace_path: Optional[str] = None):
        try:
            if workspace_path and os.path.exists(workspace_path):
                shutil.rmtree(workspace_path, ignore_errors=True)
            elif not workspace_path:
                shutil.rmtree(self.workspace_dir, ignore_errors=True)
            logging.info("Workspace cleanup completed")
        except Exception as e:
            logging.error(f"Error during cleanup: {str(e)}")
