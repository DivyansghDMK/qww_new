"""
Automatic Background Cloud Sync Service
Runs every 5 seconds to upload any new/modified files to cloud
"""

import os
import json
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Set, Dict, List


class AutoSyncService:
    """Background service that automatically syncs files to cloud every 15 seconds"""
    
    def __init__(self, interval_seconds=15):
        self.interval = interval_seconds
        self.running = False
        self.thread = None
        self.cloud_uploader = None
        self.synced_files: Set[str] = set()
        self.last_sync_time = datetime.now()
        
        # Directories to monitor
        self.project_root = Path(__file__).parent.parent.parent
        self.reports_dir = self.project_root / "reports"
        self.users_file = self.project_root / "users.json"
        
        # Track last modified times
        self.file_timestamps: Dict[str, float] = {}
        
        print(f"🔄 Auto-sync service initialized (interval: {interval_seconds}s)")
    
    def _init_cloud_uploader(self):
        """Initialize cloud uploader lazily"""
        if self.cloud_uploader is None:
            try:
                from utils.cloud_uploader import get_cloud_uploader
                self.cloud_uploader = get_cloud_uploader()
                return self.cloud_uploader.is_configured()
            except Exception as e:
                print(f"⚠️  Auto-sync: Cloud uploader not available: {e}")
                return False
        return self.cloud_uploader.is_configured()
    
    def _get_modified_files(self) -> List[Path]:
        """Get list of files that have been modified since last sync"""
        modified_files = []
        
        try:
            # Check reports directory for new/modified PDFs and JSONs
            if self.reports_dir.exists():
                # Glob for PDFs
                for file_path in self.reports_dir.glob("ECG_Report_*.pdf"):
                    if self._is_file_modified(file_path):
                        modified_files.append(file_path)
                        # Also check for JSON twin in same dir
                        json_twin = file_path.with_suffix('.json')
                        if json_twin.exists() and self._is_file_modified(json_twin):
                            modified_files.append(json_twin)
                
                # Glob for unified JSON data files in subfolder
                ecg_data_dir = self.reports_dir / "ecg_data"
                if ecg_data_dir.exists():
                    for file_path in ecg_data_dir.glob("ecg_data_*.json"):
                        if self._is_file_modified(file_path):
                            modified_files.append(file_path)
            
            # Check for generic metrics in reports root
            for metric_file in ["metrics.json", "hyper_metric.json", "hrv_metric.json"]:
                metric_path = self.reports_dir / metric_file
                if metric_path.exists() and self._is_file_modified(metric_path):
                    modified_files.append(metric_path)
            
            # Check for new user signups
            if self.users_file.exists() and self._is_file_modified(self.users_file):
                modified_files.append(self.users_file)
        
        except Exception as e:
            print(f"⚠️  Auto-sync: Error scanning files: {e}")
        
        return modified_files
    
    def _is_file_modified(self, file_path: Path) -> bool:
        """Check if file has been modified since last sync"""
        try:
            current_mtime = file_path.stat().st_mtime
            file_str = str(file_path)
            
            # If we haven't seen this file before, it's new
            if file_str not in self.file_timestamps:
                self.file_timestamps[file_str] = current_mtime
                return True
            
            # If modification time changed, file was modified
            if current_mtime > self.file_timestamps[file_str]:
                self.file_timestamps[file_str] = current_mtime
                return True
            
            return False
        except Exception:
            return False
    
    def _upload_report_files(self, file_path: Path) -> bool:
        """Upload report PDF and its JSON twin to cloud"""
        try:
            if not self.cloud_uploader:
                return False
            
            # Upload PDF
            result = self.cloud_uploader.upload_report(str(file_path))
            
            status = result.get('status')
            if status == 'success':
                print(f"✅ Auto-sync: Uploaded {file_path.name}")
                
                # Upload JSON twin if exists
                if file_path.suffix == '.pdf':
                    json_twin = file_path.with_suffix('.json')
                    if json_twin.exists():
                        json_result = self.cloud_uploader.upload_report(str(json_twin))
                        if json_result.get('status') == 'success':
                            print(f"✅ Auto-sync: Uploaded {json_twin.name}")
                
                return True
            elif status == 'already_uploaded':
                print(f"ℹ️  Auto-sync: Skipped duplicate {file_path.name}")
                return True
            else:
                print(f"⚠️  Auto-sync: Upload failed for {file_path.name}: {result.get('message', 'Unknown error')}")
                return False
                
        except Exception as e:
            print(f"❌ Auto-sync: Error uploading {file_path.name}: {e}")
            return False
    
    def _upload_user_signups(self) -> bool:
        """Upload any new user signups to cloud"""
        try:
            if not self.users_file.exists():
                return False
            
            # Read users.json
            with open(self.users_file, 'r') as f:
                users_data = json.load(f)
            
            # Upload each user's signup data individually
            upload_count = 0
            for username, user_info in users_data.items():
                if not isinstance(user_info, dict):
                    continue
                
                # Create unique identifier for this user's cloud file
                serial_id = user_info.get('serial_id', username)
                phone = user_info.get('phone', '')
                
                # Check if we've already uploaded this user
                user_key = f"{username}_{serial_id}"
                if user_key in self.synced_files:
                    continue
                
                # Prepare user data with timestamp
                user_data = {
                    "username": username,
                    "full_name": user_info.get('full_name', ''),
                    "phone": phone,
                    "serial_id": serial_id,
                    "email": user_info.get('email', ''),
                    "age": user_info.get('age', ''),
                    "gender": user_info.get('gender', ''),
                    "registration_date": user_info.get('registration_date', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                    "last_sync": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # Upload to cloud
                result = self.cloud_uploader.upload_user_signup(user_data)
                
                status = result.get('status')
                if status == 'success':
                    print(f"✅ Auto-sync: Uploaded user signup: {username}")
                    self.synced_files.add(user_key)
                    upload_count += 1
                elif status == 'already_uploaded':
                    print(f"ℹ️  Auto-sync: Skipped duplicate user signup: {username}")
                    self.synced_files.add(user_key)
                else:
                    print(f"⚠️  Auto-sync: Failed to upload user {username}: {result.get('message', 'Unknown error')}")
            
            return upload_count > 0
            
        except Exception as e:
            print(f"❌ Auto-sync: Error uploading user signups: {e}")
            return False
    
    def _sync_cycle(self):
        """Single sync cycle - upload any new/modified files"""
        try:
            # Check if cloud is configured
            if not self._init_cloud_uploader():
                # Only show message once per minute to avoid spam
                if (datetime.now() - self.last_sync_time).seconds >= 60:
                    print("ℹ️  Auto-sync: Cloud not configured (skipping sync)")
                    self.last_sync_time = datetime.now()
                return
            
            # Get modified files
            modified_files = self._get_modified_files()
            
            if not modified_files:
                return  # Nothing to sync
            
            print(f"\n🔄 Auto-sync: Found {len(modified_files)} file(s) to sync")
            
            # Upload reports
            for file_path in modified_files:
                if file_path == self.users_file:
                    # Handle user signups separately
                    self._upload_user_signups()
                else:
                    # Upload report files
                    self._upload_report_files(file_path)
            
            self.last_sync_time = datetime.now()
            print(f"✅ Auto-sync: Cycle complete at {self.last_sync_time.strftime('%H:%M:%S')}\n")
            
        except Exception as e:
            print(f"❌ Auto-sync: Error in sync cycle: {e}")
    
    def _sync_loop(self):
        """Background loop that runs sync every N seconds"""
        print(f"🚀 Auto-sync: Background service started (syncing every {self.interval}s)")
        
        while self.running:
            try:
                self._sync_cycle()
                time.sleep(self.interval)
            except Exception as e:
                print(f"❌ Auto-sync: Loop error: {e}")
                time.sleep(self.interval)
        
        print("🛑 Auto-sync: Background service stopped")
    
    def start(self):
        """Start the background sync service"""
        if self.running:
            print("⚠️  Auto-sync: Service already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._sync_loop, daemon=True)
        self.thread.start()
        
        print("=" * 70)
        print("🎉 AUTOMATIC CLOUD SYNC ENABLED!")
        print("=" * 70)
        print(f"📤 Syncing every {self.interval} seconds")
        print("📁 Monitoring:")
        print(f"   • Reports: {self.reports_dir}")
        print(f"   • Users: {self.users_file}")
        print("=" * 70)
        print()
    
    def stop(self):
        """Stop the background sync service"""
        if not self.running:
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        
        print("🛑 Auto-sync: Service stopped")
    
    def get_status(self) -> Dict:
        """Get current sync service status"""
        return {
            "running": self.running,
            "interval_seconds": self.interval,
            "last_sync": self.last_sync_time.strftime("%Y-%m-%d %H:%M:%S"),
            "synced_files_count": len(self.synced_files),
            "cloud_configured": self._init_cloud_uploader() if self.running else False
        }


# Global singleton instance
_auto_sync_service = None


def get_auto_sync_service(interval_seconds=15) -> AutoSyncService:
    """Get or create global auto-sync service instance"""
    global _auto_sync_service
    
    if _auto_sync_service is None:
        _auto_sync_service = AutoSyncService(interval_seconds=interval_seconds)
    
    return _auto_sync_service


def start_auto_sync(interval_seconds=15):
    """Start automatic background sync service"""
    service = get_auto_sync_service(interval_seconds)
    service.start()
    return service


def stop_auto_sync():
    """Stop automatic background sync service"""
    """Stop the automatic background sync service"""
    global _auto_sync_service
    
    if _auto_sync_service:
        _auto_sync_service.stop()


if __name__ == "__main__":
    # Test the auto-sync service
    print("Testing Auto-Sync Service...")
    service = start_auto_sync(interval_seconds=5)
    
    try:
        # Run for 30 seconds
        time.sleep(30)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stop_auto_sync()

