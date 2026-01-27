import os
import json
import hashlib
import hmac
from datetime import datetime
from typing import Optional, Dict, Any
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import base64

class EncryptedStorage:
    def __init__(self, user_email: str, app_data_dir: Optional[str] = None):
        self.user_email = user_email.lower().strip()
        if app_data_dir is None:
            app_data_dir = os.path.join(os.path.expanduser("~"), ".robox_data")
        self.storage_dir = app_data_dir
        self.data_file = os.path.join(self.storage_dir, f"{self.user_email}.encrypted")
        self.key_file = os.path.join(self.storage_dir, f"{self.user_email}.key")
        os.makedirs(self.storage_dir, exist_ok=True)
        self._cipher = None
        self._load_or_create_key()

    def _load_or_create_key(self):
        if os.path.exists(self.key_file):
            with open(self.key_file, "rb") as f:
                key = f.read()
        else:
            key_material = f"{self.user_email}robox_secret_salt_2024".encode()
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b"robox_salt_2024",
                iterations=100000,
                backend=default_backend(),
            )
            key = base64.urlsafe_b64encode(kdf.derive(key_material))
            with open(self.key_file, "wb") as f:
                f.write(key)
        self._cipher = Fernet(key)

    def _calculate_checksum(self, data: Dict) -> str:
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()

    def _calculate_hmac(self, data: str, checksum: str) -> str:
        secret = f"{self.user_email}robox_hmac_secret".encode()
        message = f"{data}{checksum}".encode()
        return hmac.new(secret, message, hashlib.sha256).hexdigest()

    def save(self, data: Dict[str, Any]) -> bool:
        try:
            data_with_meta = {
                "email": self.user_email,
                "data": data,
                "saved_at": datetime.utcnow().isoformat(),
                "version": "1.0",
            }
            checksum = self._calculate_checksum(data_with_meta["data"])
            data_json = json.dumps(data_with_meta, default=str)
            hmac_signature = self._calculate_hmac(data_json, checksum)
            protected_data = {
                "checksum": checksum,
                "hmac": hmac_signature,
                "encrypted_data": self._cipher.encrypt(data_json.encode()).decode(),
            }
            with open(self.data_file, "w") as f:
                json.dump(protected_data, f)
            return True
        except Exception:
            return False

    def load(self) -> Optional[Dict[str, Any]]:
        if not os.path.exists(self.data_file):
            return None
        try:
            with open(self.data_file, "r") as f:
                protected_data = json.load(f)
            checksum = protected_data.get("checksum")
            hmac_signature = protected_data.get("hmac")
            encrypted_data = protected_data.get("encrypted_data")
            if not checksum or not hmac_signature or not encrypted_data:
                return None
            decrypted = self._cipher.decrypt(encrypted_data.encode()).decode()
            calculated_hmac = self._calculate_hmac(decrypted, checksum)
            if not hmac.compare_digest(calculated_hmac, hmac_signature):
                return None
            data_with_meta = json.loads(decrypted)
            data = data_with_meta.get("data")
            if data is None:
                return None
            calculated_checksum = self._calculate_checksum(data)
            if calculated_checksum != checksum:
                return None
            return data
        except Exception:
            return None

    def delete(self) -> bool:
        try:
            if os.path.exists(self.data_file):
                os.remove(self.data_file)
            if os.path.exists(self.key_file):
                os.remove(self.key_file)
            return True
        except Exception:
            return False

    def exists(self) -> bool:
        return os.path.exists(self.data_file)