"""
Offline Queue Manager for ECG Data
Handles data queuing when internet is unavailable and syncs when connection is restored
"""

import os
import json
import time
import threading
import queue
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import socket


class OfflineQueue:
    """
    Queue system for offline data storage and automatic sync when online
    
    Features:
    - Stores data locally when offline
    - Auto-detects internet connectivity
    - Syncs queued data when connection restored
    - Prevents data loss
    - Maintains data order
    """
    
    def __init__(self, queue_dir: str = "offline_queue"):
        self.queue_dir = queue_dir
        self.pending_dir = os.path.join(queue_dir, "pending")
        self.failed_dir = os.path.join(queue_dir, "failed")
        self.synced_dir = os.path.join(queue_dir, "synced")
        
        # Create directories
        for dir_path in [self.pending_dir, self.failed_dir, self.synced_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Queue for in-memory items
        self._memory_queue = queue.Queue()
        
        # Sync thread
        self._sync_thread = None
        self._sync_running = False
        self._last_connectivity_check = 0
        self._is_online = False
        
        # Stats
        self.stats = {
            "total_queued": 0,
            "total_synced": 0,
            "total_failed": 0,
            "pending_count": 0
        }
        
        # Load pending items count
        self._update_stats()
        
        # Start background sync thread
        self.start_sync_thread()
    
    def is_online(self, force_check: bool = False) -> bool:
        """
        Check if internet connection is available
        Uses cached result for 30 seconds to avoid excessive checks
        """
        current_time = time.time()
        
        # Use cached result if recent (within 30 seconds)
        if not force_check and (current_time - self._last_connectivity_check) < 30:
            return self._is_online
        
        # Perform connectivity check
        try:
            # Try to connect to Google DNS (8.8.8.8) on port 53
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            self._is_online = True
        except OSError:
            self._is_online = False
        
        self._last_connectivity_check = current_time
        return self._is_online
    
    def queue_data(self, data_type: str, data: Dict[str, Any], priority: int = 5) -> str:
        """
        Queue data for upload
        
        Args:
            data_type: Type of data (metrics, waveform, report, etc.)
            data: The data payload
            priority: Priority level (1=highest, 10=lowest)
            
        Returns:
            Queue item ID
        """
        item_id = f"{data_type}_{int(time.time() * 1000)}_{os.urandom(4).hex()}"
        
        queue_item = {
            "id": item_id,
            "type": data_type,
            "data": data,
            "priority": priority,
            "queued_at": datetime.utcnow().isoformat() + 'Z',
            "retry_count": 0,
            "status": "pending"
        }
        
        # Save to disk immediately
        self._save_to_disk(queue_item, self.pending_dir)
        
        # Add to memory queue for faster processing
        self._memory_queue.put(queue_item)
        
        # Update stats
        self.stats["total_queued"] += 1
        self.stats["pending_count"] += 1
        
        print(f"📥 Queued {data_type}: {item_id}")
        return item_id
    
    def _save_to_disk(self, item: Dict[str, Any], directory: str) -> None:
        """Save queue item to disk"""
        try:
            file_path = os.path.join(directory, f"{item['id']}.json")
            with open(file_path, 'w') as f:
                json.dump(item, f, indent=2)
        except Exception as e:
            print(f"⚠️  Failed to save queue item to disk: {e}")
    
    def _load_from_disk(self, directory: str) -> List[Dict[str, Any]]:
        """Load all queue items from disk directory"""
        items = []
        try:
            for filename in sorted(os.listdir(directory)):
                if filename.endswith('.json'):
                    file_path = os.path.join(directory, filename)
                    try:
                        with open(file_path, 'r') as f:
                            items.append(json.load(f))
                    except Exception as e:
                        print(f"⚠️  Failed to load {filename}: {e}")
        except Exception as e:
            print(f"⚠️  Failed to read queue directory: {e}")
        return items
    
    def _delete_from_disk(self, item_id: str, directory: str) -> None:
        """Delete queue item from disk"""
        try:
            file_path = os.path.join(directory, f"{item_id}.json")
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"⚠️  Failed to delete queue item from disk: {e}")
    
    def _move_item(self, item_id: str, from_dir: str, to_dir: str) -> None:
        """Move item from one directory to another"""
        try:
            from_path = os.path.join(from_dir, f"{item_id}.json")
            to_path = os.path.join(to_dir, f"{item_id}.json")
            if os.path.exists(from_path):
                os.rename(from_path, to_path)
        except Exception as e:
            print(f"⚠️  Failed to move queue item: {e}")
    
    def start_sync_thread(self) -> None:
        """Start background sync thread"""
        if self._sync_thread is None or not self._sync_thread.is_alive():
            self._sync_running = True
            self._sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
            self._sync_thread.start()
            print("🔄 Offline queue sync thread started")
    
    def stop_sync_thread(self) -> None:
        """Stop background sync thread"""
        self._sync_running = False
        if self._sync_thread:
            self._sync_thread.join(timeout=5)
            print("⏹️  Offline queue sync thread stopped")
    
    def _sync_loop(self) -> None:
        """Background sync loop"""
        while self._sync_running:
            try:
                # Check connectivity every 30 seconds
                if self.is_online(force_check=True):
                    # Load pending items from disk if memory queue is empty
                    if self._memory_queue.empty():
                        pending_items = self._load_from_disk(self.pending_dir)
                        for item in pending_items:
                            self._memory_queue.put(item)
                    
                    # Process items from memory queue
                    self._process_queue()
                else:
                    print("📴 Offline mode - data will be queued locally")
                
                # Update stats
                self._update_stats()
                
            except Exception as e:
                print(f"⚠️  Error in sync loop: {e}")
            
            # Sleep for 30 seconds before next check
            time.sleep(30)
    
    def _process_queue(self) -> None:
        """Process queued items and attempt upload"""
        processed = 0
        max_batch = 10  # Process max 10 items per cycle
        
        while not self._memory_queue.empty() and processed < max_batch:
            try:
                # Get item with timeout
                item = self._memory_queue.get(timeout=1)
                
                # Attempt to sync
                success = self._sync_item(item)
                
                if success:
                    # Move to synced directory
                    self._move_item(item['id'], self.pending_dir, self.synced_dir)
                    self.stats["total_synced"] += 1
                    self.stats["pending_count"] -= 1
                    print(f"✅ Synced {item['type']}: {item['id']}")
                    
                    # Delete old synced items (keep last 100)
                    self._cleanup_synced_items()
                else:
                    # Increment retry count
                    item['retry_count'] += 1
                    
                    if item['retry_count'] < 5:
                        # Re-queue for retry
                        self._memory_queue.put(item)
                        self._save_to_disk(item, self.pending_dir)
                        print(f"🔄 Re-queued {item['type']}: {item['id']} (retry {item['retry_count']})")
                    else:
                        # Move to failed directory after 5 retries
                        self._move_item(item['id'], self.pending_dir, self.failed_dir)
                        self.stats["total_failed"] += 1
                        self.stats["pending_count"] -= 1
                        print(f"❌ Failed {item['type']}: {item['id']} (max retries exceeded)")
                
                processed += 1
                
            except queue.Empty:
                break
            except Exception as e:
                print(f"⚠️  Error processing queue item: {e}")
    
    def _sync_item(self, item: Dict[str, Any]) -> bool:
        """
        Attempt to sync single item to backend or cloud storage
        Supports both BackendAPI and CloudUploader (S3)
        """
        # Try cloud uploader first (for S3 uploads)
        try:
            from .cloud_uploader import get_cloud_uploader
            cloud_uploader = get_cloud_uploader()
            
            # Handle cloud-specific queue items
            if item['type'] == 'cloud_report':
                # Upload report to S3
                file_path = item['data'].get('file_path')
                metadata = item['data'].get('metadata', {})
                
                if file_path and os.path.exists(file_path):
                    # Temporarily disable offline queue check to force upload
                    original_offline_queue = cloud_uploader.offline_queue
                    cloud_uploader.offline_queue = None  # Disable queue check during sync
                    
                    try:
                        result = cloud_uploader.upload_report(file_path, metadata)
                    finally:
                        cloud_uploader.offline_queue = original_offline_queue  # Restore
                    
                    if result.get('status') == 'success':
                        print(f"✅ Synced report to S3: {os.path.basename(file_path)}")
                        return True
                    elif result.get('status') == 'already_uploaded':
                        # Already uploaded, consider it success
                        return True
                    else:
                        print(f"⚠️ S3 upload failed: {result.get('message', 'Unknown error')}")
                        return False
                else:
                    print(f"⚠️ Report file not found: {file_path}")
                    return False
            
            elif item['type'] == 'cloud_user_signup':
                # Upload user signup to S3
                user_data = item['data'].get('user_data', {})
                
                # Temporarily disable offline queue check to force upload
                original_offline_queue = cloud_uploader.offline_queue
                cloud_uploader.offline_queue = None  # Disable queue check during sync
                
                try:
                    result = cloud_uploader.upload_user_signup(user_data)
                finally:
                    cloud_uploader.offline_queue = original_offline_queue  # Restore
                
                if result.get('status') == 'success':
                    print(f"✅ Synced user signup to S3: {user_data.get('username', 'unknown')}")
                    return True
                elif result.get('status') == 'already_uploaded':
                    # Already uploaded, consider it success
                    return True
                else:
                    print(f"⚠️ S3 user signup failed: {result.get('message', 'Unknown error')}")
                    return False
            
            elif item['type'] == 'cloud_complete_package':
                # Upload complete report package to S3
                pdf_path = item['data'].get('pdf_path')
                patient_data = item['data'].get('patient_data', {})
                ecg_data_file = item['data'].get('ecg_data_file')
                report_metadata = item['data'].get('report_metadata', {})
                
                if pdf_path and os.path.exists(pdf_path):
                    # Temporarily disable offline queue check to force upload
                    original_offline_queue = cloud_uploader.offline_queue
                    cloud_uploader.offline_queue = None  # Disable queue check during sync
                    
                    try:
                        result = cloud_uploader.upload_complete_report_package(
                            pdf_path, patient_data, ecg_data_file, report_metadata
                        )
                    finally:
                        cloud_uploader.offline_queue = original_offline_queue  # Restore
                    
                    if result.get('status') == 'success':
                        print(f"✅ Synced complete package to S3: {os.path.basename(pdf_path)}")
                        return True
                    else:
                        print(f"⚠️ S3 package upload failed: {result.get('message', 'Unknown error')}")
                        return False
                else:
                    print(f"⚠️ PDF file not found: {pdf_path}")
                    return False
        except ImportError:
            pass
        except Exception as e:
            print(f"⚠️ Cloud uploader error: {e}")
        
        # Fallback to BackendAPI for other data types
        try:
            from .backend_api import get_backend_api
            backend_api = get_backend_api()
            
            # Route based on data type
            if item['type'] == 'metrics':
                result = backend_api.upload_metrics(item['data'])
            elif item['type'] == 'waveform':
                result = backend_api.upload_waveform(
                    item['data'].get('leads', {}),
                    item['data'].get('sampling_rate', 80)
                )
            elif item['type'] == 'report':
                result = backend_api.upload_report(
                    item['data'].get('file_path'),
                    item['data'].get('metadata', {})
                )
            elif item['type'] == 'session_start':
                result = backend_api.start_session(
                    item['data'].get('device_serial'),
                    item['data'].get('device_info', {})
                )
            elif item['type'] == 'session_end':
                result = backend_api.end_session(item['data'].get('summary', {}))
            else:
                print(f"⚠️  Unknown data type: {item['type']}")
                return False
            
            return result.get('status') == 'success'
            
        except ImportError:
            # Neither backend API nor cloud uploader available
            print(f"ℹ️  No upload service available for {item['type']}")
            return False
        except Exception as e:
            print(f"⚠️  Sync error for {item['type']}: {e}")
            return False
    
    def _update_stats(self) -> None:
        """Update queue statistics"""
        try:
            self.stats["pending_count"] = len([f for f in os.listdir(self.pending_dir) if f.endswith('.json')])
        except Exception:
            pass
    
    def _cleanup_synced_items(self, keep_last: int = 100) -> None:
        """Clean up old synced items, keeping only the most recent ones"""
        try:
            synced_files = sorted([
                f for f in os.listdir(self.synced_dir) if f.endswith('.json')
            ])
            
            if len(synced_files) > keep_last:
                files_to_delete = synced_files[:-keep_last]
                for filename in files_to_delete:
                    os.remove(os.path.join(self.synced_dir, filename))
                print(f"🧹 Cleaned up {len(files_to_delete)} old synced items")
        except Exception as e:
            print(f"⚠️  Cleanup error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        self._update_stats()
        return {
            **self.stats,
            "is_online": self.is_online(),
            "queue_dir": self.queue_dir
        }
    
    def get_pending_items(self) -> List[Dict[str, Any]]:
        """Get all pending queue items"""
        return self._load_from_disk(self.pending_dir)
    
    def get_failed_items(self) -> List[Dict[str, Any]]:
        """Get all failed queue items"""
        return self._load_from_disk(self.failed_dir)
    
    def retry_failed_items(self) -> int:
        """Retry all failed items"""
        failed_items = self.get_failed_items()
        count = 0
        
        for item in failed_items:
            # Reset retry count
            item['retry_count'] = 0
            item['status'] = 'pending'
            
            # Move back to pending
            self._move_item(item['id'], self.failed_dir, self.pending_dir)
            self._memory_queue.put(item)
            
            self.stats["total_failed"] -= 1
            self.stats["pending_count"] += 1
            count += 1
        
        print(f"🔄 Retrying {count} failed items")
        return count
    
    def clear_synced_items(self) -> int:
        """Clear all synced items"""
        try:
            synced_files = [f for f in os.listdir(self.synced_dir) if f.endswith('.json')]
            for filename in synced_files:
                os.remove(os.path.join(self.synced_dir, filename))
            print(f"🧹 Cleared {len(synced_files)} synced items")
            return len(synced_files)
        except Exception as e:
            print(f"⚠️  Clear error: {e}")
            return 0
    
    def force_sync_now(self) -> None:
        """Force immediate sync attempt"""
        print("🔄 Forcing immediate sync...")
        if self.is_online(force_check=True):
            self._process_queue()
        else:
            print("📴 Cannot sync - no internet connection")


# Global instance
_offline_queue = None

def get_offline_queue() -> OfflineQueue:
    """Get or create global offline queue instance"""
    global _offline_queue
    if _offline_queue is None:
        _offline_queue = OfflineQueue()
    return _offline_queue

