"""
Encrypted file storage for code submissions.
Stores user code files in encrypted format to prevent tampering.
"""
import os
import base64
import json
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from datetime import datetime
from typing import Optional, Dict, List

class EncryptedFileStorage:
    """Manages encrypted storage of code submission files."""
    
    def __init__(self, user_email: str, base_dir: str = None):
        self.user_email = user_email
        if base_dir is None:
            base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'user_submission')
        self.base_dir = base_dir
        self.storage_dir = os.path.join(base_dir, '.encrypted_submissions')
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Generate encryption key from user email
        self.cipher = self._get_cipher()
    
    def _get_cipher(self) -> Fernet:
        """Generate Fernet cipher from user email."""
        from cryptography.hazmat.backends import default_backend
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'robox_salt_2024',
            iterations=100000,
            backend=default_backend()
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.user_email.encode()))
        return Fernet(key)
    
    def save_code_file(self, challenge_id: str, code: str, filename: str) -> Dict:
        """
        Save code file in encrypted format.
        
        Args:
            challenge_id: Challenge identifier (e.g., "8", "9")
            code: Source code content
            filename: Original filename
            
        Returns:
            Dict with file metadata
        """
        timestamp = datetime.utcnow().isoformat()
        
        # Create metadata
        metadata = {
            'challenge_id': challenge_id,
            'filename': filename,
            'timestamp': timestamp,
            'user_email': self.user_email,
            'size': len(code)
        }
        
        # Encrypt code
        encrypted_code = self.cipher.encrypt(code.encode())
        
        # Save encrypted file
        encrypted_filename = f"{challenge_id}_{filename}.enc"
        encrypted_path = os.path.join(self.storage_dir, encrypted_filename)
        
        with open(encrypted_path, 'wb') as f:
            f.write(encrypted_code)
        
        # Save metadata
        metadata_path = os.path.join(self.storage_dir, f"{challenge_id}_{filename}.meta")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return metadata
    
    def load_code_file(self, challenge_id: str, filename: str) -> Optional[str]:
        """
        Load and decrypt code file.
        
        Args:
            challenge_id: Challenge identifier
            filename: Original filename
            
        Returns:
            Decrypted code content or None if not found
        """
        encrypted_filename = f"{challenge_id}_{filename}.enc"
        encrypted_path = os.path.join(self.storage_dir, encrypted_filename)
        
        if not os.path.exists(encrypted_path):
            return None
        
        try:
            with open(encrypted_path, 'rb') as f:
                encrypted_code = f.read()
            
            decrypted_code = self.cipher.decrypt(encrypted_code)
            return decrypted_code.decode()
        except Exception as e:
            print(f"Error decrypting file: {e}")
            return None
    
    def get_file_metadata(self, challenge_id: str, filename: str) -> Optional[Dict]:
        """Get metadata for a code file."""
        metadata_path = os.path.join(self.storage_dir, f"{challenge_id}_{filename}.meta")
        
        if not os.path.exists(metadata_path):
            return None
        
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading metadata: {e}")
            return None
    
    def list_challenge_files(self, challenge_id: str) -> List[Dict]:
        """List all files for a specific challenge."""
        files = []
        prefix = f"{challenge_id}_"
        
        for filename in os.listdir(self.storage_dir):
            if filename.startswith(prefix) and filename.endswith('.meta'):
                metadata = self.get_file_metadata(
                    challenge_id,
                    filename[len(prefix):-5]  # Remove prefix and .meta
                )
                if metadata:
                    files.append(metadata)
        
        return files
    
    def delete_file(self, challenge_id: str, filename: str) -> bool:
        """Delete encrypted file and metadata."""
        encrypted_filename = f"{challenge_id}_{filename}.enc"
        metadata_filename = f"{challenge_id}_{filename}.meta"
        
        encrypted_path = os.path.join(self.storage_dir, encrypted_filename)
        metadata_path = os.path.join(self.storage_dir, metadata_filename)
        
        success = True
        if os.path.exists(encrypted_path):
            try:
                os.remove(encrypted_path)
            except Exception as e:
                print(f"Error deleting encrypted file: {e}")
                success = False
        
        if os.path.exists(metadata_path):
            try:
                os.remove(metadata_path)
            except Exception as e:
                print(f"Error deleting metadata: {e}")
                success = False
        
        return success
    
    def clear_all(self) -> bool:
        """Clear all encrypted files for this user."""
        try:
            for filename in os.listdir(self.storage_dir):
                filepath = os.path.join(self.storage_dir, filename)
                if os.path.isfile(filepath):
                    os.remove(filepath)
            return True
        except Exception as e:
            print(f"Error clearing storage: {e}")
            return False
