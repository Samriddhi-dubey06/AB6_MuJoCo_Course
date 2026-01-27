import copy
import time
import threading
from typing import Optional, Dict, Any, Tuple, List
from src.utils.encrypted_storage import EncryptedStorage
from src.utils.encrypted_file_storage import EncryptedFileStorage
from backend_api_client import BackendAPIClient

class ProgressSyncManager:
    def __init__(self, user_email: str, api_client: BackendAPIClient):
        self.user_email = user_email
        self.api_client = api_client
        self.local_storage = EncryptedStorage(user_email)
        self.file_storage = EncryptedFileStorage(user_email)
        self.last_sync_time: Optional[float] = None
        self.pending_progress: Optional[Dict[str, Any]] = None
        self.pending_files: List[Dict[str, str]] = []  # Track pending file uploads
        # Cache for is_online() to avoid hitting API too often
        # Increased cache duration to reduce API calls under high traffic
        self._online_cache: Optional[bool] = None
        self._online_cache_time: float = 0
        self._online_cache_duration: float = 600  # Cache for 10 minutes (reduced from 60s for better performance)
        # Debouncing for save operations - batch rapid saves
        # Increased debounce delay to reduce API calls under high traffic
        self._save_debounce_timer: Optional[float] = None
        self._debounced_progress: Optional[Dict[str, Any]] = None
        self._debounce_delay: float = 5.0  # Wait 5 seconds before saving to batch rapid updates (increased from 2s)
        self._save_in_progress: bool = False
        self._sync_lock = threading.Lock()  # Thread safety

    def _clone(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return copy.deepcopy(payload)

    def _update_snapshot(self, progress_data: Dict[str, Any]):
        self.local_storage.save(self._clone(progress_data))

    def _store_pending(self, progress_data: Dict[str, Any], message: str) -> Tuple[bool, str]:
        self.pending_progress = self._clone(progress_data)
        return True, message

    def _load_snapshot(self) -> Optional[Dict[str, Any]]:
        try:
            return self.local_storage.load()
        except Exception:
            return None

    def is_online(self) -> bool:
        """Check if online with caching to avoid frequent API calls.
        Uses token validation instead of full /users/me endpoint for better performance."""
        try:
            # Check if authenticated first (this uses cached token validation)
            if not self.api_client.is_authenticated():
                self._online_cache = False
                return False
            
            # Check cache first - avoid hitting API too often
            current_time = time.time()
            if (self._online_cache is not None and 
                current_time - self._online_cache_time < self._online_cache_duration):
                return self._online_cache
            
            # Since token is valid (from is_authenticated), we assume online
            # Only invalidate if we actually get a 401 on a real API call
            # This avoids hitting /users/me endpoint which does DB lookup on every call
            self._online_cache = True
            self._online_cache_time = current_time
            return True
        except Exception:
            self._online_cache = False
            self._online_cache_time = time.time()
            return False
    
    def invalidate_online_cache(self):
        """Invalidate the online cache (call when connection status might have changed)."""
        self._online_cache = None
        self._online_cache_time = 0

    def has_pending_local_data(self) -> bool:
        return self.pending_progress is not None

    def save_progress(self, progress_data: Dict[str, Any], force_cloud: bool = False, debounce: bool = True) -> Tuple[bool, str]:
        """
        Save progress with debouncing to batch rapid saves (important for multiple concurrent users).
        When online: Save directly to cloud, otherwise store locally.
        """
        with self._sync_lock:  # Thread safety
            # Store the latest progress data for debouncing
            self._debounced_progress = self._clone(progress_data)
            
            # Always update local snapshot immediately for responsiveness
            self._update_snapshot(progress_data)
            
            # If debouncing is disabled or force_cloud, save immediately
            if not debounce or force_cloud:
                self._save_debounce_timer = None  # Clear debounce timer
                return self._do_save_progress(progress_data, force_cloud)
            
            # Debounce: Check if enough time has passed since last save
            current_time = time.time()
            if self._save_debounce_timer is None:
                # First save in sequence - start timer and save immediately
                self._save_debounce_timer = current_time
                return self._do_save_progress(progress_data, force_cloud)
            
            # Check if debounce period has passed
            time_since_last = current_time - self._save_debounce_timer
            if time_since_last >= self._debounce_delay:
                # Enough time passed, save now and reset timer
                self._save_debounce_timer = current_time
                return self._do_save_progress(self._debounced_progress, force_cloud)
            else:
                # Too soon - queue for later (will be saved by sync timer or flush)
                return True, f"Progress queued (saving in {self._debounce_delay - time_since_last:.1f}s)"
    
    def _do_save_progress(self, progress_data: Dict[str, Any], force_cloud: bool = False) -> Tuple[bool, str]:
        """Internal method to actually save progress (called after debouncing)."""
        # Prevent concurrent saves
        if self._save_in_progress:
            return True, "Save already in progress, queued"
        
        self._save_in_progress = True
        try:
            if self.is_online() or force_cloud:
                try:
                    # Save directly to cloud first when online
                    success, message = self.api_client.save_progress(progress_data)
                    if success:
                        self.pending_progress = None
                        self.last_sync_time = time.time()
                        # Invalidate online cache after successful save
                        self.invalidate_online_cache()
                        return True, "Progress saved to cloud"
                    # If cloud save failed, save locally as backup
                    return self._store_pending(progress_data, message or "Cloud rejected progress, saved locally")
                except Exception as exc:
                    # Exception during cloud save, save locally as backup
                    return self._store_pending(progress_data, f"Offline mode ({str(exc)}), saved locally")
            else:
                # Offline - save locally only
                return self._store_pending(progress_data, "No internet connection, saved locally")
        finally:
            self._save_in_progress = False
    
    def flush_pending_save(self) -> Tuple[bool, str]:
        """Force save any pending debounced progress immediately."""
        if self._debounced_progress:
            self._save_debounce_timer = None
            return self._do_save_progress(self._debounced_progress)
        return True, "No pending save"

    def load_progress(self) -> Optional[Dict[str, Any]]:
        """
        Load progress - prefer local first, only sync from cloud if needed.
        Avoids unnecessary cloud fetches that cause overriding issues.
        """
        print(f"[ProgressSync] Loading progress for {self.user_email}")
        
        local_data = self._load_snapshot()
        
        # If not authenticated, use local only
        if not self.api_client.is_authenticated():
            print(f"[ProgressSync] Not authenticated, using local storage")
            return local_data

        # If online and we have pending data, sync it first
        if self.is_online() and self.pending_progress:
            try:
                success, _ = self.api_client.save_progress(self.pending_progress)
                if success:
                    self.pending_progress = None
                    self.last_sync_time = time.time()
            except Exception as e:
                print(f"[ProgressSync] Error syncing pending data: {e}")

        # If online and no local data, fetch from cloud
        if self.is_online() and not local_data:
            print(f"[ProgressSync] Online but no local data - fetching from cloud...")
            try:
                cloud_data = self.api_client.get_progress()
                if cloud_data:
                    print(f"[ProgressSync] Using cloud data")
                    self._update_snapshot(cloud_data)
                    self.last_sync_time = time.time()
                    return cloud_data
            except Exception as e:
                print(f"[ProgressSync] Error fetching from cloud: {e}")
        
        # Use local data (preferred to avoid unnecessary overrides)
        # Only merge if we have both local and cloud and they differ significantly
        if self.is_online() and local_data:
            try:
                cloud_data = self.api_client.get_progress()
                if cloud_data:
                    # Only merge if there's a significant difference
                    local_score = local_data.get("total_score", 0)
                    cloud_score = cloud_data.get("total_score", 0)
                    local_completed = local_data.get("challenges_completed", 0)
                    cloud_completed = cloud_data.get("challenges_completed", 0)
                    
                    # If scores differ significantly, merge
                    if abs(local_score - cloud_score) > 0.1 or local_completed != cloud_completed:
                        merged_data = self._merge_progress(local_data, cloud_data)
                        print(f"[ProgressSync] Merged local and cloud data")
                        self._update_snapshot(merged_data)
                        self.last_sync_time = time.time()
                        return merged_data
            except Exception as e:
                print(f"[ProgressSync] Error comparing with cloud: {e}")
        
        # Use local data (default - avoids unnecessary overrides)
        print(f"[ProgressSync] Using local storage")
        return local_data
    
    def _merge_progress(self, local_data: Dict[str, Any], cloud_data: Dict[str, Any]) -> Dict[str, Any]:
        """Merge local and cloud data, keeping the most attempts and best scores."""
        merged = cloud_data.copy()
        
        local_scores = local_data.get("challenge_scores", {})
        cloud_scores = cloud_data.get("challenge_scores", {})
        merged_scores = {}
        
        # Merge challenge scores - keep all attempts from both
        all_challenge_ids = set(local_scores.keys()) | set(cloud_scores.keys())
        for challenge_id in all_challenge_ids:
            local_attempts = local_scores.get(challenge_id, [])
            cloud_attempts = cloud_scores.get(challenge_id, [])
            
            # If local has more attempts, use local
            if len(local_attempts) > len(cloud_attempts):
                merged_scores[challenge_id] = local_attempts
            else:
                merged_scores[challenge_id] = cloud_attempts
        
        merged["challenge_scores"] = merged_scores
        
        # Recalculate totals
        total_score = 0
        for scores in merged_scores.values():
            if scores:
                best = max(s.get("score", 0) for s in scores if isinstance(s, dict))
                total_score += best
        
        merged["total_score"] = total_score
        merged["challenges_completed"] = max(
            local_data.get("challenges_completed", 0),
            cloud_data.get("challenges_completed", 0)
        )
        
        # Merge challenges list - prefer local if it has more completed
        local_challenges = local_data.get("challenges", [])
        cloud_challenges = cloud_data.get("challenges", [])
        local_completed = sum(1 for c in local_challenges if c.get("completed", False))
        cloud_completed = sum(1 for c in cloud_challenges if c.get("completed", False))
        
        merged["challenges"] = local_challenges if local_completed >= cloud_completed else cloud_challenges
        
        return merged

    def sync_to_cloud(self) -> Tuple[bool, str]:
        """Sync pending data to cloud without fetching back (avoids override issues)."""
        # Flush any debounced saves first (check if debounce period has passed)
        if self._debounced_progress:
            current_time = time.time()
            if (self._save_debounce_timer is None or 
                current_time - self._save_debounce_timer >= self._debounce_delay):
                self.flush_pending_save()
        
        if not self.pending_progress and not self.pending_files:
            return True, "No local data to sync"

        if not self.is_online():
            return False, "No internet connection"

        messages = []
        overall_success = True
        
        # Sync progress data
        if self.pending_progress:
            try:
                success, message = self.api_client.save_progress(self.pending_progress)
                if success:
                    self.last_sync_time = time.time()
                    # Update local snapshot after successful sync (don't fetch back)
                    self._update_snapshot(self.pending_progress)
                    self.pending_progress = None
                    messages.append("Progress synced")
                else:
                    messages.append(f"Progress sync failed: {message}")
                    overall_success = False
            except Exception as exc:
                messages.append(f"Progress sync error: {str(exc)}")
                overall_success = False
        
        # Sync pending files
        if self.pending_files:
            file_success, file_message = self.sync_pending_files()
            messages.append(file_message)
            if not file_success:
                overall_success = False
        
        return overall_success, "; ".join(messages) if messages else "Sync completed"

    def clear_local_cache(self) -> None:
        self.local_storage.delete()

    def clear_all_storage(self) -> None:
        import shutil
        self.pending_progress = None
        self.pending_files = []
        self.local_storage.delete()
        self.file_storage.clear_all()
        
        # Delete user_data folder
        try:
            import os
            user_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'user_data')
            if os.path.exists(user_data_dir):
                shutil.rmtree(user_data_dir)
                print(f"Deleted user_data folder: {user_data_dir}")
        except Exception as e:
            print(f"Error deleting user_data: {e}")
        
        # Delete submission_data folder
        try:
            submission_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'user_submission')
            if os.path.exists(submission_data_dir):
                shutil.rmtree(submission_data_dir)
                print(f"Deleted submission_data folder: {submission_data_dir}")
        except Exception as e:
            print(f"Error deleting submission_data: {e}")

    def get_sync_status(self) -> Dict[str, Any]:
        return {
            "online": self.is_online(),
            "has_local_data": self.has_pending_local_data(),
            "last_sync": self.last_sync_time,
            "pending_files": len(self.pending_files)
        }
    
    def save_code_file(self, challenge_id: str, code: str, filename: str, score: Optional[float] = None, feedback: Optional[str] = None) -> Tuple[bool, str]:
        """
        Save code file with submission metadata: Upload to MongoDB if online, save locally as backup.
        Avoids unnecessary fetch-after-upload to prevent override issues.
        
        Args:
            challenge_id: Challenge identifier
            code: Source code
            filename: Filename
            score: Score from mentor (only on final submission)
            feedback: Mentor feedback (only on final submission)
        """
        # Upload to MongoDB first if online (avoid unnecessary local save/fetch cycle)
        if self.is_online():
            try:
                success, data, error = self.api_client.upload_code_submission(
                    challenge_id, code, filename, score, feedback
                )
                if success:
                    # Save locally after successful upload (no fetch needed)
                    try:
                        self.file_storage.save_code_file(challenge_id, code, filename)
                    except Exception as e:
                        print(f"Local save failed after upload: {e}")
                    return True, "Uploaded and saved locally"
                else:
                    # Upload failed, save locally as backup
                    try:
                        self.file_storage.save_code_file(challenge_id, code, filename)
                        self.pending_files.append({'challenge_id': challenge_id, 'filename': filename})
                        return True, f"Saved locally, pending upload: {error}"
                    except Exception as e:
                        return False, f"Both upload and local save failed: {str(e)}"
            except Exception as e:
                # Exception during upload, save locally as backup
                try:
                    self.file_storage.save_code_file(challenge_id, code, filename)
                    self.pending_files.append({'challenge_id': challenge_id, 'filename': filename})
                    return True, f"Saved locally, pending upload: {str(e)}"
                except Exception as local_e:
                    return False, f"Both upload and local save failed: {str(e)}, {str(local_e)}"
        else:
            # Offline - save locally only
            try:
                self.file_storage.save_code_file(challenge_id, code, filename)
                self.pending_files.append({'challenge_id': challenge_id, 'filename': filename})
                return True, "Offline - will upload when online"
            except Exception as e:
                return False, f"Local save failed: {str(e)}"
    
    def load_code_file(self, challenge_id: str, filename: str) -> Optional[str]:
        """
        Load code file from local storage or cloud.
        
        Args:
            challenge_id: Challenge identifier
            filename: Original filename
            
        Returns:
            Code content or None
        """
        # Try local storage first
        code = self.file_storage.load_code_file(challenge_id, filename)
        if code:
            print(f"Code file loaded from local storage: {filename}")
            return code
        
        # Try cloud if online
        if self.is_online():
            try:
                success, data, error = self.api_client.get_code_submission(challenge_id)
                if success and data:
                    code = data.get('code')
                    if code:
                        # Save to local storage for offline access
                        self.file_storage.save_code_file(challenge_id, code, filename)
                        print(f"Code file downloaded from cloud: {filename}")
                        return code
            except Exception as e:
                print(f"Error fetching code from cloud: {e}")
        
        return None
    
    def sync_pending_files(self) -> Tuple[bool, str]:
        """
        Upload pending files to MongoDB without fetching back (avoids override issues).
        """
        if not self.pending_files:
            return True, "No pending files"

        if not self.is_online():
            return False, "Offline"

        successful = 0
        for file_info in self.pending_files[:]:
            challenge_id = file_info['challenge_id']
            filename = file_info['filename']
            
            code = self.file_storage.load_code_file(challenge_id, filename)
            if not code:
                # Remove from pending if file doesn't exist locally
                self.pending_files.remove(file_info)
                continue
            
            try:
                success, _, _ = self.api_client.upload_code_submission(challenge_id, code, filename)
                if success:
                    # Don't fetch back - local file is already correct
                    # Just remove from pending list
                    self.pending_files.remove(file_info)
                    successful += 1
            except Exception as e:
                print(f"Error syncing file {filename}: {e}")
        
        return True, f"Synced {successful} files"