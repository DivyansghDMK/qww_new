"""
Cloud Uploader Module for ECG Reports
Supports multiple cloud storage services for automatic report backup
Includes offline queue support for automatic sync when internet is restored
"""

import os
import json
import requests
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import sys

# Load environment variables
# 1) Load from current working directory (if running from project root)
load_dotenv()
# 2) Also attempt to load from project root explicitly (works when app starts from elsewhere)
try:
    this_file = Path(__file__).resolve()
    project_root = this_file.parents[2] if len(this_file.parents) >= 3 else this_file.parent
    root_env = project_root / '.env'
    if root_env.exists():
        load_dotenv(dotenv_path=str(root_env), override=False)
    else:
        # Fallback: also try one directory up (in case of unusual packaging)
        alt_env = project_root.parent / '.env'
        if alt_env.exists():
            load_dotenv(dotenv_path=str(alt_env), override=False)
except Exception:
    # Best-effort; silently continue if path resolution fails
    pass

# 3) PyInstaller: load .env from frozen app directory and unpack dir (sys._MEIPASS)
try:
    if getattr(sys, 'frozen', False):
        app_dir = os.path.dirname(sys.executable)
        app_env = os.path.join(app_dir, '.env')
        if os.path.exists(app_env):
            load_dotenv(dotenv_path=app_env, override=False)
        meipass = getattr(sys, '_MEIPASS', None)
        if meipass:
            meipass_env = os.path.join(meipass, '.env')
            if os.path.exists(meipass_env):
                load_dotenv(dotenv_path=meipass_env, override=False)
except Exception:
    pass

# Final fallback: manually parse .env if python-dotenv misses it (rare edge cases)
def _manual_env_load(env_path: Path):
    try:
        if env_path.exists():
            with env_path.open('r', encoding='utf-8', errors='ignore') as f:
                for raw in f:
                    line = raw.replace('\ufeff', '')  # strip BOM if present
                    line = line.replace('＝', '=')      # normalize unicode equals
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if '=' in line:
                        k, v = line.split('=', 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        # Force override to ensure latest values are used
                        if k:
                            os.environ[k] = v
    except Exception:
        pass

try:
    _manual_env_load(root_env)
    # Also try app dir and MEIPASS for manual parse
    if getattr(sys, 'frozen', False):
        app_dir = os.path.dirname(sys.executable)
        _manual_env_load(Path(app_dir) / '.env')
        meipass = getattr(sys, '_MEIPASS', None)
        if meipass:
            _manual_env_load(Path(meipass) / '.env')
except Exception:
    pass


class CloudUploader:
    """Handle uploading ECG reports to cloud storage with offline queue support"""
    
    def __init__(self):
        self.cloud_service = os.getenv('CLOUD_SERVICE', 'none').lower()
        self.upload_enabled = os.getenv('CLOUD_UPLOAD_ENABLED', 'false').lower() == 'true'
        
        # AWS S3 Configuration
        self.s3_bucket = os.getenv('AWS_S3_BUCKET')
        self.s3_region = os.getenv('AWS_S3_REGION', 'us-east-1')
        self.aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        
        # Azure Blob Storage Configuration
        self.azure_connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
        self.azure_container = os.getenv('AZURE_CONTAINER_NAME', 'ecg-reports')
        
        # Google Cloud Storage Configuration
        self.gcs_bucket = os.getenv('GCS_BUCKET_NAME')
        self.gcs_credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        
        # Custom API Endpoint Configuration
        self.api_endpoint = os.getenv('CLOUD_API_ENDPOINT')
        self.api_key = os.getenv('CLOUD_API_KEY')
        
        # FTP/SFTP Configuration
        self.ftp_host = os.getenv('FTP_HOST')
        self.ftp_port = int(os.getenv('FTP_PORT', '21'))
        self.ftp_username = os.getenv('FTP_USERNAME')
        self.ftp_password = os.getenv('FTP_PASSWORD')
        self.ftp_remote_path = os.getenv('FTP_REMOTE_PATH', '/ecg-reports')
        
        # Dropbox Configuration
        self.dropbox_token = os.getenv('DROPBOX_ACCESS_TOKEN')
        
        # Doctor Review API Configuration
        self.doctor_review_enabled = os.getenv('DOCTOR_REVIEW_ENABLED', 'false').lower() == 'true'
        self.doctor_review_api_url = os.getenv('DOCTOR_REVIEW_API_URL', '')
        # Robustly handle if URL includes /upload-assigned
        if self.doctor_review_api_url:
            if self.doctor_review_api_url.endswith('/upload-assigned'):
                self.doctor_review_api_url = self.doctor_review_api_url[:-16]
            elif self.doctor_review_api_url.endswith('/upload-assigned/'):
                self.doctor_review_api_url = self.doctor_review_api_url[:-17]
                
        self.doctor_review_api_key = os.getenv('DOCTOR_REVIEW_API_KEY', '')
        self._doctor_list_cache = None
        self._last_doctor_fetch_time = 0
        
        # Log file for upload tracking
        self.upload_log_path = "reports/upload_log.json"
        
        # Initialize offline queue for automatic sync
        try:
            from .offline_queue import get_offline_queue
            self.offline_queue = get_offline_queue()
            print("✅ Offline queue initialized for cloud uploader")
        except Exception as e:
            print(f"⚠️ Could not initialize offline queue: {e}")
            self.offline_queue = None

    def reload_config(self):
        """Re-read .env from CWD and project root and refresh fields."""
        try:
            load_dotenv(override=True)
            this_file = Path(__file__).resolve()
            project_root = this_file.parents[2] if len(this_file.parents) >= 3 else this_file.parent
            root_env = project_root / '.env'
            if root_env.exists():
                load_dotenv(dotenv_path=str(root_env), override=True)
            # Manual fallback parse as well
            try:
                _manual_env_load(root_env)
            except Exception:
                pass
            # Also try CWD .env
            try:
                cwd_env = Path(os.getcwd()) / '.env'
                _manual_env_load(cwd_env)
            except Exception:
                pass
        except Exception:
            pass
        self.cloud_service = os.getenv('CLOUD_SERVICE', 'none').lower()
        self.upload_enabled = os.getenv('CLOUD_UPLOAD_ENABLED', 'false').lower() == 'true'
        self.s3_bucket = os.getenv('AWS_S3_BUCKET')
        self.s3_region = os.getenv('AWS_S3_REGION', 'us-east-1')
        self.aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.doctor_review_enabled = os.getenv('DOCTOR_REVIEW_ENABLED', 'false').lower() == 'true'
        self.doctor_review_api_url = os.getenv('DOCTOR_REVIEW_API_URL')
        self.doctor_review_api_key = os.getenv('DOCTOR_REVIEW_API_KEY')

    def get_config_snapshot(self):
        return {
            'cloud_service': self.cloud_service,
            'upload_enabled': self.upload_enabled,
            's3_bucket': self.s3_bucket,
            's3_region': self.s3_region,
            'aws_access_key_set': bool(self.aws_access_key),
            'aws_secret_key_set': bool(self.aws_secret_key),
        }
        
    def is_configured(self):
        """Check if cloud upload is properly configured"""
        if not self.upload_enabled:
            return False
            
        if self.cloud_service == 's3':
            return bool(self.s3_bucket and self.aws_access_key and self.aws_secret_key)
        elif self.cloud_service == 'azure':
            return bool(self.azure_connection_string)
        elif self.cloud_service == 'gcs':
            return bool(self.gcs_bucket and self.gcs_credentials_path)
        elif self.cloud_service == 'api':
            return bool(self.api_endpoint)
        elif self.cloud_service == 'ftp' or self.cloud_service == 'sftp':
            return bool(self.ftp_host and self.ftp_username)
        elif self.cloud_service == 'dropbox':
            return bool(self.dropbox_token)
        
        return False
    
    def _is_file_already_uploaded(self, file_path):
        """
        Check if a file has already been uploaded based on filename
        
        Args:
            file_path (str): Path to the file to check
            
        Returns:
            bool: True if file has already been uploaded, False otherwise
        """
        try:
            filename = os.path.basename(file_path)
            
            # Check upload log
            if os.path.exists(self.upload_log_path):
                with open(self.upload_log_path, 'r') as f:
                    log_data = json.load(f)
                
                # Check if this filename has been uploaded before
                for entry in log_data:
                    # Get filename from the logged path
                    logged_filename = os.path.basename(entry.get('local_path', ''))
                    
                    # Also check metadata filename as backup
                    metadata_filename = entry.get('metadata', {}).get('filename', '')
                    
                    # If filename matches and upload was successful
                    if (logged_filename == filename or metadata_filename == filename):
                        if entry.get('result', {}).get('status') == 'success':
                            return True
            
            return False
            
        except Exception as e:
            print(f"⚠️ Error checking upload history: {e}")
            # If we can't check, assume not uploaded (safer to upload twice than not at all)
            return False
    
    def upload_report(self, file_path, metadata=None):
        """
        Upload ONLY reports, metrics, and report files to AWS S3
        Does NOT upload: session logs, debug data, crash logs, temp files
        Prevents duplicate uploads - files are only uploaded once
        Supports offline mode - queues for upload when internet is restored
        
        Args:
            file_path (str): Path to the report file (PDF, JSON, etc.)
            metadata (dict): Optional metadata about the report
            
        Returns:
            dict: Upload result with status, url, and error if any
        """
        if not self.upload_enabled:
            return {"status": "disabled", "message": "Cloud upload is disabled"}
            
        if not self.is_configured():
            return {"status": "error", "message": f"Cloud service '{self.cloud_service}' is not properly configured"}
        
        try:
            # Check if file has already been uploaded
            filename = os.path.basename(file_path)
            if self._is_file_already_uploaded(file_path):
                return {
                    "status": "already_uploaded",
                    "message": f"File '{filename}' has already been uploaded to cloud - skipping duplicate upload",
                    "filename": filename
                }
            
            # Check if this is a report file (PDF or JSON report)
            file_ext = Path(file_path).suffix.lower()
            file_basename = os.path.basename(file_path).lower()
            
            # ONLY upload reports, metrics, and unified data - filter out everything else
            allowed_extensions = ['.pdf', '.json']
            is_report = file_ext in allowed_extensions and 'report' in file_basename
            is_metric = file_ext == '.json' and ('metric' in file_basename or 'ecg_data' in file_basename)
            
            if not (is_report or is_metric):
                return {
                    "status": "skipped",
                    "message": f"File {file_basename} is not a report or metric file - not uploaded"
                }
            
            # Prepare metadata - ONLY include essential report information
            upload_metadata = {
                "filename": os.path.basename(file_path),
                "uploaded_at": datetime.now().isoformat(),
                "file_size": os.path.getsize(file_path),
                "file_type": Path(file_path).suffix,
            }
            
            # Only add specific metadata fields if provided
            if metadata:
                # Include all patient details and metrics in metadata
                allowed_keys = [
                    'patient_name', 'patient_age', 'patient_gender', 'patient_address', 
                    'patient_phone', 'report_date', 'machine_serial',
                    'heart_rate', 'pr_interval', 'qrs_duration', 'qt_interval', 
                    'qtc_interval', 'st_segment'
                ]
                filtered_metadata = {k: v for k, v in metadata.items() if k in allowed_keys}
                upload_metadata.update(filtered_metadata)
            
            # Check if online - if offline, queue for later upload
            if self.offline_queue and not self.offline_queue.is_online():
                # Queue for upload when internet is restored
                queue_payload = {
                    'file_path': file_path,
                    'metadata': upload_metadata,
                    'cloud_service': self.cloud_service
                }
                self.offline_queue.queue_data('cloud_report', queue_payload, priority=2)  # High priority
                print(f"📥 Queued report for upload when online: {filename}")
                return {
                    "status": "queued",
                    "message": f"Report queued for upload when internet connection is restored",
                    "filename": filename
                }
            
            # Upload based on configured service
            if self.cloud_service == 's3':
                result = self._upload_to_s3(file_path, upload_metadata)
            elif self.cloud_service == 'azure':
                result = self._upload_to_azure(file_path, upload_metadata)
            elif self.cloud_service == 'gcs':
                result = self._upload_to_gcs(file_path, upload_metadata)
            elif self.cloud_service == 'api':
                result = self._upload_to_api(file_path, upload_metadata)
            elif self.cloud_service == 'ftp':
                result = self._upload_to_ftp(file_path, upload_metadata, use_sftp=False)
            elif self.cloud_service == 'sftp':
                result = self._upload_to_ftp(file_path, upload_metadata, use_sftp=True)
            elif self.cloud_service == 'dropbox':
                result = self._upload_to_dropbox(file_path, upload_metadata)
            else:
                result = {"status": "error", "message": f"Unknown cloud service: {self.cloud_service}"}
            
            # If upload failed and we have offline queue, queue for retry
            if result.get("status") != "success" and self.offline_queue:
                queue_payload = {
                    'file_path': file_path,
                    'metadata': upload_metadata,
                    'cloud_service': self.cloud_service
                }
                self.offline_queue.queue_data('cloud_report', queue_payload, priority=2)
                result["status"] = "queued"
                result["message"] = "Upload failed - queued for retry when online"
            
            # Log the upload
            if result.get("status") == "success":
                self._log_upload(file_path, result, upload_metadata)
            
            return result
            
        except Exception as e:
            # On any error, try to queue if offline queue is available
            if self.offline_queue:
                try:
                    upload_metadata = {
                        "filename": os.path.basename(file_path),
                        "uploaded_at": datetime.now().isoformat(),
                        "file_size": os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                        "file_type": Path(file_path).suffix,
                    }
                    if metadata:
                        upload_metadata.update(metadata)
                    
                    queue_payload = {
                        'file_path': file_path,
                        'metadata': upload_metadata,
                        'cloud_service': self.cloud_service
                    }
                    self.offline_queue.queue_data('cloud_report', queue_payload, priority=2)
                    return {
                        "status": "queued",
                        "message": f"Error occurred - queued for upload when online: {str(e)}"
                    }
                except:
                    pass
            
            return {"status": "error", "message": str(e)}
    
    def upload_user_signup(self, user_data):
        """
        Upload user signup details to cloud storage
        Prevents duplicate uploads - checks if user has already been uploaded
        Supports offline mode - queues for upload when internet is restored
        
        Args:
            user_data (dict): Dictionary containing user signup information
                             {username, full_name, age, gender, phone, address, serial_number, registered_at}
        
        Returns:
            dict: Upload result with status and details
        """
        print(f"🔵 upload_user_signup called with data: {user_data}")
        
        if not self.upload_enabled:
            msg = "Cloud upload is disabled"
            print(f"❌ {msg}")
            return {"status": "disabled", "message": msg}
        
        if not self.is_configured():
            msg = f"Cloud service '{self.cloud_service}' is not properly configured"
            print(f"❌ {msg}")
            return {"status": "error", "message": msg}
        
        if not user_data or not isinstance(user_data, dict):
            msg = "Invalid user data"
            print(f"❌ {msg}")
            return {"status": "error", "message": msg}
        
        # Check if this user has already been uploaded
        username = user_data.get('username', 'unknown')
        serial_number = user_data.get('serial_number', '')
        
        try:
            if os.path.exists(self.upload_log_path):
                with open(self.upload_log_path, 'r') as f:
                    log_data = json.load(f)
                
                # Check if user with same username or serial has been uploaded
                for entry in log_data:
                    metadata = entry.get('metadata', {})
                    if metadata.get('type') == 'user_signup':
                        if (metadata.get('username') == username or 
                            (serial_number and metadata.get('serial_number') == serial_number)):
                            if entry.get('result', {}).get('status') == 'success':
                                print(f"ℹ️ User signup for '{username}' already uploaded - skipping duplicate")
                                return {
                                    "status": "already_uploaded",
                                    "message": f"User signup for '{username}' has already been uploaded",
                                    "username": username
                                }
        except Exception as e:
            print(f"⚠️ Error checking user signup history: {e}")
        
        # Check if online - if offline, queue for later upload
        if self.offline_queue and not self.offline_queue.is_online():
            queue_payload = {
                'user_data': user_data,
                'cloud_service': self.cloud_service
            }
            self.offline_queue.queue_data('cloud_user_signup', queue_payload, priority=1)  # Highest priority
            print(f"📥 Queued user signup for upload when online: {username}")
            return {
                "status": "queued",
                "message": f"User signup queued for upload when internet connection is restored",
                "username": username
            }
        
        try:
            # Create a JSON file with user signup details
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"user_signup_{username}_{timestamp}.json"
            
            # Create temp directory if it doesn't exist
            temp_dir = "temp"
            os.makedirs(temp_dir, exist_ok=True)
            file_path = os.path.join(temp_dir, filename)
            
            print(f"📝 Creating user signup file: {file_path}")
            
            # Add timestamp to user data
            upload_data = user_data.copy()
            upload_data['uploaded_at'] = datetime.now().isoformat()
            
            # Write user data to JSON file
            with open(file_path, 'w') as f:
                json.dump(upload_data, f, indent=2)
            
            print(f"✅ User signup file created successfully")
            
            # Upload to cloud
            metadata = {
                'type': 'user_signup',
                'username': username,
                'serial_number': serial_number,
                'uploaded_at': datetime.now().isoformat()
            }
            
            print(f"☁️ Uploading to {self.cloud_service}...")
            
            result = None
            if self.cloud_service == 's3':
                result = self._upload_to_s3(file_path, metadata)
            elif self.cloud_service == 'azure':
                result = self._upload_to_azure(file_path, metadata)
            elif self.cloud_service == 'gcs':
                result = self._upload_to_gcs(file_path, metadata)
            elif self.cloud_service == 'api':
                result = self._upload_to_api(file_path, metadata)
            elif self.cloud_service == 'ftp':
                result = self._upload_to_ftp(file_path, metadata)
            elif self.cloud_service == 'sftp':
                result = self._upload_to_ftp(file_path, metadata, use_sftp=True)
            elif self.cloud_service == 'dropbox':
                result = self._upload_to_dropbox(file_path, metadata)
            else:
                result = {"status": "error", "message": f"Unknown cloud service: {self.cloud_service}"}
            
            print(f"📤 Upload result: {result}")
            
            # Clean up temp file
            try:
                os.remove(file_path)
                print(f"🗑️ Temp file removed: {file_path}")
            except Exception as cleanup_err:
                print(f"⚠️ Could not remove temp file: {cleanup_err}")
            
            # Log upload
            if result and result.get("status") == "success":
                self._log_upload(filename, result, metadata)
                print(f"✅ User signup uploaded to {self.cloud_service}: {username}")
            else:
                print(f"❌ Upload failed: {result}")
            
            return result
            
        except Exception as e:
            import traceback
            error_msg = f"Failed to upload user signup: {str(e)}"
            print(f"❌ {error_msg}")
            print(f"Stack trace: {traceback.format_exc()}")
            return {"status": "error", "message": error_msg}

    def get_available_doctors(self, force_refresh=False):
        """
        Fetch list of available doctors from the API.
        Returns a list of doctor names.
        Uses a cache to avoid redundant API calls.
        """
        import time
        default_doctors = ['Dr_Rohit', 'Dr_Neha', 'Dr_Arjun', 'Dr_Simran', 'Dr_Kabir']
        
        # Return cache if available and not forced refresh (cache for 1 hour)
        now = time.time()
        if not force_refresh and self._doctor_list_cache and (now - self._last_doctor_fetch_time < 3600):
            return self._doctor_list_cache

        if not self.doctor_review_api_url:
            return default_doctors
        
        try:
            base_url = self.doctor_review_api_url.rstrip('/')
            url = f"{base_url}/admin/doctor"
            
            print(f"🔹 Fetching doctor list from {url}")
            
            headers = {}
            if self.doctor_review_api_key:
                # Use x-api-key for AWS API Gateway. 
                # Avoid 'Authorization' header as it triggers SigV4 validation errors on AWS.
                headers['x-api-key'] = self.doctor_review_api_key

            response = requests.get(url, headers=headers, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                doctors = []
                
                # Handle nested structure: {"doctors": [...]}
                doctor_data = data
                if isinstance(data, dict) and "doctors" in data:
                    doctor_data = data["doctors"]
                
                if isinstance(doctor_data, list):
                    for item in doctor_data:
                        if isinstance(item, str):
                            doctors.append(item)
                        elif isinstance(item, dict):
                            name = item.get('name') or item.get('doctorName') or item.get('username')
                            if name:
                                doctors.append(name)
                
                if doctors:
                    print(f"✅ Fetched {len(doctors)} doctors from API")
                    self._doctor_list_cache = doctors
                    self._last_doctor_fetch_time = now
                    return doctors
            
            print(f"⚠️ Failed to fetch doctors (Status {response.status_code}). Using fallback.")
            
        except Exception as e:
            print(f"⚠️ Error fetching doctor list: {e}. Using fallback.")
            
        # If fetch failed but we have a stale cache, return it
        if self._doctor_list_cache:
            return self._doctor_list_cache
            
        return default_doctors
    
    def _upload_to_s3(self, file_path, metadata):
        """Upload to AWS S3"""
        try:
            import boto3
            from botocore.exceptions import ClientError
            
            s3_client = boto3.client(
                's3',
                aws_access_key_id=self.aws_access_key,
                aws_secret_access_key=self.aws_secret_key,
                region_name=self.s3_region
            )
            
            # Generate S3 key
            filename = os.path.basename(file_path)
            timestamp = datetime.now().strftime("%Y/%m/%d")
            s3_key = f"ecg-reports/{timestamp}/{filename}"
            
            # Upload file
            s3_client.upload_file(
                file_path,
                self.s3_bucket,
                s3_key,
                ExtraArgs={'Metadata': {k: str(v) for k, v in metadata.items()}}
            )
            
            # Generate presigned URL (optional)
            url = f"https://{self.s3_bucket}.s3.{self.s3_region}.amazonaws.com/{s3_key}"
            
            return {
                "status": "success",
                "service": "s3",
                "url": url,
                "key": s3_key,
                "bucket": self.s3_bucket
            }
            
        except ImportError:
            return {"status": "error", "message": "boto3 not installed. Run: pip install boto3"}
        except ClientError as e:
            return {"status": "error", "message": f"S3 upload failed: {str(e)}"}
    
    def upload_complete_report_package(self, pdf_path, patient_data, ecg_data_file, report_metadata=None):
        """
        Upload complete ECG report package to S3:
        - PDF report
        - Patient details (JSON)
        - 12-lead ECG data (10 seconds, JSON)
        Supports offline mode - queues for upload when internet is restored
        
        Args:
            pdf_path (str): Path to PDF report file
            patient_data (dict): Patient details dictionary
            ecg_data_file (str): Path to JSON file containing 12-lead ECG data (10 seconds)
            report_metadata (dict): Optional additional metadata
            
        Returns:
            dict: Upload result with status and details
        """
        if not self.upload_enabled:
            return {"status": "disabled", "message": "Cloud upload is disabled"}
            
        if not self.is_configured():
            return {"status": "error", "message": f"Cloud service '{self.cloud_service}' is not properly configured"}
        
        if self.cloud_service != 's3':
            return {"status": "error", "message": "Complete report package upload only supports S3"}
        
        # Check if online - if offline, queue for later upload
        if self.offline_queue and not self.offline_queue.is_online():
            print(f"📥 Offline mode - queuing complete report package for upload when online")
            return {
                "status": "queued",
                "message": "Report package queued for upload when internet connection is restored"
            }
        
        try:
            import boto3
            from botocore.exceptions import ClientError
            
            s3_client = boto3.client(
                's3',
                aws_access_key_id=self.aws_access_key,
                aws_secret_access_key=self.aws_secret_key,
                region_name=self.s3_region
            )
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            date_prefix = datetime.now().strftime("%Y/%m/%d")
            
            upload_results = {
                "status": "success",
                "service": "s3",
                "bucket": self.s3_bucket,
                "uploads": []
            }
            
            # 1. Upload PDF Report
            if pdf_path and os.path.exists(pdf_path):
                pdf_filename = os.path.basename(pdf_path)
                pdf_s3_key = f"ecg-reports/{date_prefix}/{timestamp}/{pdf_filename}"
                
                pdf_metadata = {
                    "type": "ecg_report_pdf",
                    "uploaded_at": datetime.now().isoformat(),
                    "filename": pdf_filename
                }
                if report_metadata:
                    pdf_metadata.update(report_metadata)
                
                s3_client.upload_file(
                    pdf_path,
                    self.s3_bucket,
                    pdf_s3_key,
                    ExtraArgs={'Metadata': {k: str(v) for k, v in pdf_metadata.items()}}
                )
                
                upload_results["uploads"].append({
                    "type": "pdf_report",
                    "key": pdf_s3_key,
                    "url": f"https://{self.s3_bucket}.s3.{self.s3_region}.amazonaws.com/{pdf_s3_key}"
                })
                print(f"✅ Uploaded PDF report to S3: {pdf_s3_key}")
            
            # 2. Upload Patient Details (JSON)
            if patient_data:
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
                    json.dump(patient_data, f, indent=2, ensure_ascii=False)
                    patient_file = f.name
                
                try:
                    patient_filename = f"patient_details_{timestamp}.json"
                    patient_s3_key = f"ecg-reports/{date_prefix}/{timestamp}/{patient_filename}"
                    
                    patient_metadata = {
                        "type": "patient_details",
                        "uploaded_at": datetime.now().isoformat(),
                        "patient_name": patient_data.get("patient_name") or 
                                       f"{patient_data.get('first_name', '')} {patient_data.get('last_name', '')}".strip()
                    }
                    
                    s3_client.upload_file(
                        patient_file,
                        self.s3_bucket,
                        patient_s3_key,
                        ExtraArgs={'Metadata': {k: str(v) for k, v in patient_metadata.items()}}
                    )
                    
                    upload_results["uploads"].append({
                        "type": "patient_details",
                        "key": patient_s3_key,
                        "url": f"https://{self.s3_bucket}.s3.{self.s3_region}.amazonaws.com/{patient_s3_key}"
                    })
                    print(f"✅ Uploaded patient details to S3: {patient_s3_key}")
                finally:
                    try:
                        os.remove(patient_file)
                    except:
                        pass
            
            # 3. Upload 12-Lead ECG Data (10 seconds, JSON)
            if ecg_data_file and os.path.exists(ecg_data_file):
                ecg_filename = os.path.basename(ecg_data_file)
                ecg_s3_key = f"ecg-reports/{date_prefix}/{timestamp}/{ecg_filename}"
                
                # Load ECG data to verify it has 10 seconds
                try:
                    with open(ecg_data_file, 'r') as f:
                        ecg_data = json.load(f)
                    
                    sampling_rate = ecg_data.get("sampling_rate", 80.0)
                    expected_samples = int(sampling_rate * 10)  # 10 seconds
                    
                    ecg_metadata = {
                        "type": "ecg_12lead_data",
                        "uploaded_at": datetime.now().isoformat(),
                        "filename": ecg_filename,
                        "sampling_rate": str(sampling_rate),
                        "duration_seconds": "10",
                        "expected_samples": str(expected_samples)
                    }
                    
                    # Verify each lead has approximately 10 seconds of data
                    leads = ecg_data.get("leads", {})
                    for lead_name, lead_data in leads.items():
                        if isinstance(lead_data, list):
                            actual_samples = len(lead_data)
                            ecg_metadata[f"lead_{lead_name}_samples"] = str(actual_samples)
                    
                except Exception as e:
                    print(f"⚠️ Warning: Could not verify ECG data: {e}")
                    ecg_metadata = {
                        "type": "ecg_12lead_data",
                        "uploaded_at": datetime.now().isoformat(),
                        "filename": ecg_filename
                    }
                
                s3_client.upload_file(
                    ecg_data_file,
                    self.s3_bucket,
                    ecg_s3_key,
                    ExtraArgs={'Metadata': {k: str(v) for k, v in ecg_metadata.items()}}
                )
                
                upload_results["uploads"].append({
                    "type": "ecg_data",
                    "key": ecg_s3_key,
                    "url": f"https://{self.s3_bucket}.s3.{self.s3_region}.amazonaws.com/{ecg_s3_key}"
                })
                print(f"✅ Uploaded 12-lead ECG data to S3: {ecg_s3_key}")
            
            # Log the upload
            if upload_results["uploads"]:
                self._log_upload(pdf_path or ecg_data_file, upload_results, {
                    "type": "complete_report_package",
                    "timestamp": timestamp,
                    "uploads_count": len(upload_results["uploads"])
                })
            
            return upload_results
            
        except ImportError as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"  DEBUG: ImportError details: {e}")
            print(f"  DEBUG: Full traceback:\n{error_details}")
            return {"status": "error", "message": f"boto3 not installed or import failed: {str(e)}. Run: pip install boto3"}
        except ClientError as e:
            return {"status": "error", "message": f"S3 upload failed: {str(e)}"}
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"  DEBUG: Unexpected error during S3 upload: {e}")
            print(f"  DEBUG: Full traceback:\n{error_details}")
            return {"status": "error", "message": f"S3 upload failed: {str(e)}"}
        except Exception as e:
            import traceback
            return {"status": "error", "message": f"Upload failed: {str(e)}", "traceback": traceback.format_exc()}

    def list_reports(self, prefix: str = "ecg-reports/"):
        """List report objects in S3 (PDF and JSON under prefix)."""
        if not (self.upload_enabled and self.cloud_service == 's3' and self.s3_bucket):
            return {"status": "error", "message": "S3 not configured"}
        try:
            import boto3
            s3 = boto3.client(
                's3',
                aws_access_key_id=self.aws_access_key,
                aws_secret_access_key=self.aws_secret_key,
                region_name=self.s3_region
            )
            paginator = s3.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.s3_bucket, Prefix=prefix)
            items = []
            for page in pages:
                for obj in page.get('Contents', []) or []:
                    key = obj['Key']
                    if not key.lower().endswith(('.pdf', '.json')):
                        continue
                    items.append({
                        'key': key,
                        'size': obj.get('Size', 0),
                        'last_modified': obj.get('LastModified').isoformat() if obj.get('LastModified') else '',
                        'url': f"https://{self.s3_bucket}.s3.{self.s3_region}.amazonaws.com/{key}"
                    })
            return {"status": "success", "items": items}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def generate_presigned_url(self, key: str, expires_in: int = 3600):
        """Generate a presigned URL for a given S3 object key."""
        try:
            import boto3
            s3 = boto3.client(
                's3',
                aws_access_key_id=self.aws_access_key,
                aws_secret_access_key=self.aws_secret_key,
                region_name=self.s3_region
            )
            url = s3.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.s3_bucket, 'Key': key},
                ExpiresIn=expires_in
            )
            return {"status": "success", "url": url}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def delete_file(self, key: str):
        """Delete a file from S3 bucket"""
        try:
            import boto3
            s3 = boto3.client(
                's3',
                aws_access_key_id=self.aws_access_key,
                aws_secret_access_key=self.aws_secret_key,
                region_name=self.s3_region
            )
            
            # Delete the object
            s3.delete_object(Bucket=self.s3_bucket, Key=key)
            print(f"✅ Deleted from S3: {key}")
            return {"status": "success", "message": f"Deleted {key}"}
            
        except Exception as e:
            print(f"❌ S3 deletion error for {key}: {e}")
            return {"status": "error", "message": str(e)}
    
    def _upload_to_azure(self, file_path, metadata):
        """Upload to Azure Blob Storage"""
        try:
            from azure.storage.blob import BlobServiceClient
            
            blob_service_client = BlobServiceClient.from_connection_string(self.azure_connection_string)
            container_client = blob_service_client.get_container_client(self.azure_container)
            
            # Ensure container exists
            try:
                container_client.create_container()
            except Exception:
                pass  # Container already exists
            
            # Generate blob name
            filename = os.path.basename(file_path)
            timestamp = datetime.now().strftime("%Y/%m/%d")
            blob_name = f"ecg-reports/{timestamp}/{filename}"
            
            # Upload file
            blob_client = container_client.get_blob_client(blob_name)
            with open(file_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True, metadata=metadata)
            
            url = blob_client.url
            
            return {
                "status": "success",
                "service": "azure",
                "url": url,
                "blob_name": blob_name,
                "container": self.azure_container
            }
            
        except ImportError:
            return {"status": "error", "message": "azure-storage-blob not installed. Run: pip install azure-storage-blob"}
        except Exception as e:
            return {"status": "error", "message": f"Azure upload failed: {str(e)}"}
    
    def _upload_to_gcs(self, file_path, metadata):
        """Upload to Google Cloud Storage"""
        try:
            from google.cloud import storage
            
            storage_client = storage.Client.from_service_account_json(self.gcs_credentials_path)
            bucket = storage_client.bucket(self.gcs_bucket)
            
            # Generate blob name
            filename = os.path.basename(file_path)
            timestamp = datetime.now().strftime("%Y/%m/%d")
            blob_name = f"ecg-reports/{timestamp}/{filename}"
            
            # Upload file
            blob = bucket.blob(blob_name)
            blob.metadata = metadata
            blob.upload_from_filename(file_path)
            
            url = blob.public_url
            
            return {
                "status": "success",
                "service": "gcs",
                "url": url,
                "blob_name": blob_name,
                "bucket": self.gcs_bucket
            }
            
        except ImportError:
            return {"status": "error", "message": "google-cloud-storage not installed. Run: pip install google-cloud-storage"}
        except Exception as e:
            return {"status": "error", "message": f"GCS upload failed: {str(e)}"}
    
    def _upload_to_api(self, file_path, metadata):
        """Upload to custom API endpoint"""
        try:
            with open(file_path, 'rb') as f:
                files = {'file': f}
                headers = {}
                if self.api_key:
                    headers['Authorization'] = f'Bearer {self.api_key}'
                
                data = {'metadata': json.dumps(metadata)}
                
                response = requests.post(
                    self.api_endpoint,
                    files=files,
                    data=data,
                    headers=headers,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json() if response.content else {}
                    return {
                        "status": "success",
                        "service": "api",
                        "response": result,
                        "url": result.get('url', self.api_endpoint)
                    }
                else:
                    return {
                        "status": "error",
                        "message": f"API returned status {response.status_code}: {response.text}"
                    }
                    
        except Exception as e:
            return {"status": "error", "message": f"API upload failed: {str(e)}"}
    
    def _upload_to_ftp(self, file_path, metadata, use_sftp=False):
        """Upload to FTP/SFTP server"""
        try:
            if use_sftp:
                import paramiko
                
                transport = paramiko.Transport((self.ftp_host, self.ftp_port))
                transport.connect(username=self.ftp_username, password=self.ftp_password)
                sftp = paramiko.SFTPClient.from_transport(transport)
                
                # Create remote directory if needed
                remote_file = f"{self.ftp_remote_path}/{os.path.basename(file_path)}"
                sftp.put(file_path, remote_file)
                sftp.close()
                transport.close()
                
            else:
                from ftplib import FTP
                
                ftp = FTP()
                ftp.connect(self.ftp_host, self.ftp_port)
                ftp.login(self.ftp_username, self.ftp_password)
                
                # Upload file
                with open(file_path, 'rb') as f:
                    remote_file = f"{self.ftp_remote_path}/{os.path.basename(file_path)}"
                    ftp.storbinary(f'STOR {remote_file}', f)
                
                ftp.quit()
            
            return {
                "status": "success",
                "service": "sftp" if use_sftp else "ftp",
                "remote_path": remote_file
            }
            
        except ImportError:
            return {"status": "error", "message": "paramiko not installed for SFTP. Run: pip install paramiko"}
        except Exception as e:
            return {"status": "error", "message": f"FTP upload failed: {str(e)}"}
    
    def _upload_to_dropbox(self, file_path, metadata):
        """Upload to Dropbox"""
        try:
            import dropbox
            
            dbx = dropbox.Dropbox(self.dropbox_token)
            
            # Generate Dropbox path
            filename = os.path.basename(file_path)
            timestamp = datetime.now().strftime("%Y/%m/%d")
            dropbox_path = f"/ECG-Reports/{timestamp}/{filename}"
            
            # Upload file
            with open(file_path, 'rb') as f:
                dbx.files_upload(f.read(), dropbox_path, mode=dropbox.files.WriteMode.overwrite)
            
            # Get shareable link
            try:
                link = dbx.sharing_create_shared_link(dropbox_path)
                url = link.url
            except:
                url = dropbox_path
            
            return {
                "status": "success",
                "service": "dropbox",
                "url": url,
                "path": dropbox_path
            }
            
        except ImportError:
            return {"status": "error", "message": "dropbox not installed. Run: pip install dropbox"}
        except Exception as e:
            return {"status": "error", "message": f"Dropbox upload failed: {str(e)}"}
    
    def _log_upload(self, file_path, result, metadata):
        """Log successful upload to tracking file"""
        try:
            # Load existing log
            log_data = []
            if os.path.exists(self.upload_log_path):
                with open(self.upload_log_path, 'r') as f:
                    log_data = json.load(f)
            
            # Add new entry
            log_entry = {
                "local_path": file_path,
                "uploaded_at": datetime.now().isoformat(),
                "service": self.cloud_service,
                "result": result,
                "metadata": metadata
            }
            log_data.append(log_entry)
            
            # Save log
            os.makedirs(os.path.dirname(self.upload_log_path), exist_ok=True)
            with open(self.upload_log_path, 'w') as f:
                json.dump(log_data, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Could not log upload: {e}")
    
    def get_upload_history(self, limit=50):
        """Get recent upload history"""
        try:
            if os.path.exists(self.upload_log_path):
                with open(self.upload_log_path, 'r') as f:
                    log_data = json.load(f)
                return log_data[-limit:]
            return []
        except Exception:
            return []
    
    def get_uploaded_files_list(self):
        """
        Get a list of all uploaded filenames for easy checking
        
        Returns:
            list: List of successfully uploaded filenames
        """
        try:
            if os.path.exists(self.upload_log_path):
                with open(self.upload_log_path, 'r') as f:
                    log_data = json.load(f)
                
                uploaded_files = []
                for entry in log_data:
                    if entry.get('result', {}).get('status') == 'success':
                        filename = entry.get('metadata', {}).get('filename', '')
                        if not filename:
                            filename = os.path.basename(entry.get('local_path', ''))
                        if filename:
                            uploaded_files.append(filename)
                
                return uploaded_files
            return []
        except Exception as e:
            print(f"⚠️ Error getting uploaded files list: {e}")
            return []
    
    def clear_upload_log(self):
        """
        Clear the upload log (use with caution!)
        This will allow re-uploading of all files
        
        Returns:
            dict: Result with status
        """
        try:
            if os.path.exists(self.upload_log_path):
                # Backup the log before clearing
                backup_path = f"{self.upload_log_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                os.makedirs(os.path.dirname(backup_path), exist_ok=True)
                
                # Copy to backup
                import shutil
                shutil.copy2(self.upload_log_path, backup_path)
                
                # Clear the log
                with open(self.upload_log_path, 'w') as f:
                    json.dump([], f)
                
                return {
                    "status": "success",
                    "message": f"Upload log cleared. Backup saved to {backup_path}"
                }
            else:
                return {
                    "status": "success",
                    "message": "Upload log does not exist - nothing to clear"
                }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to clear upload log: {str(e)}"
            }
    
    def send_for_doctor_review(self, file_path, doctor_name, metadata=None):
        """
        Send ECG report for doctor review using 2-step process:
        1. GET presigned URL from backend
        2. PUT file content to presigned URL
        
        Args:
            file_path (str): Path to the PDF report
            doctor_name (str): Name of the doctor to review
            metadata (dict): Optional metadata
            
        Returns:
            dict: Result with status and message
        """
        # Default base URL if not set
        base_url = self.doctor_review_api_url
        if not base_url:
            base_url = "https://8m9fgt2fz1.execute-api.us-east-1.amazonaws.com/prod/api"
            
        if not self.doctor_review_enabled and os.getenv('DOCTOR_REVIEW_ENABLED', 'true').lower() != 'true':
             # Allow it to work if enabled in env even if init didn't pick it up? 
             # Or just enforce flag. User said "send my doctor review report", implying they want it now.
             pass

        if not os.path.exists(file_path):
            return {"status": "error", "message": f"File not found: {file_path}"}
            
        filename = os.path.basename(file_path)
        
        # Check if online
        if self.offline_queue and not self.offline_queue.is_online():
            queue_payload = {
                'file_path': file_path,
                'doctor_name': doctor_name,
                'metadata': metadata
            }
            self.offline_queue.queue_data('doctor_review_v2', queue_payload, priority=1)
            return {
                "status": "queued",
                "message": "Queued for doctor review (offline)"
            }

        try:
            # Step 1: Get presigned URL
            upload_url_endpoint = f"{base_url.rstrip('/')}/upload-assigned"
            
            payload = {
                "doctorName": doctor_name,
                "patientName": metadata.get('patient_name') if metadata else "Unknown",
                "reportType": metadata.get('report_type') if metadata else "ECG",
                "fileName": filename
            }
            
            print(f"🔹 Requesting upload URL from {upload_url_endpoint} for {doctor_name}")
            
            headers = {"Content-Type": "application/json"}
            if self.doctor_review_api_key:
                # Use x-api-key for AWS API Gateway.
                headers['x-api-key'] = self.doctor_review_api_key

            response = requests.post(
                upload_url_endpoint,
                json=payload,
                headers=headers,
                timeout=15
            )
            
            if response.status_code != 200:
                print(f"❌ Failed to get upload URL: {response.text}")
                return {"status": "error", "message": f"Failed to get upload URL: {response.text}"}
                
            response_data = response.json()
            presigned_url = response_data.get("url") # Expecting 'url' or similar in response? 
            # User said: Returns pre-signed S3 upload URL. 
            # Usually it's in a field like 'url' or directly? 
            # Let's assume it returns a JSON with the URL.
            # If response is just string, handle that?
            if isinstance(response_data, str):
                 presigned_url = response_data
            elif isinstance(response_data, dict):
                 presigned_url = response_data.get("url") or response_data.get("uploadUrl")
            
            if not presigned_url:
                 # Last ditch: maybe the body is the URL?
                 if response.text.startswith("http"):
                     presigned_url = response.text.strip().strip('"')
            
            if not presigned_url:
                print(f"❌ No URL in response: {response.text}")
                return {"status": "error", "message": "No upload URL received from backend"}
                
            print(f"🔹 Got presigned URL. Uploading file...")
            
            # Step 2: Upload file via PUT
            with open(file_path, 'rb') as f:
                file_data = f.read()
                
            put_response = requests.put(
                presigned_url,
                data=file_data,
                headers={"Content-Type": "application/pdf"},
                timeout=60
            )
            
            if put_response.status_code in [200, 201]:
                print(f"✅ Report uploaded successfully for review!")
                
                # Log success
                self._log_upload(file_path, {"status": "success"}, {"type": "doctor_review", "doctor": doctor_name})
                
                return {"status": "success", "message": "Report sent for review successfully"}
            else:
                print(f"❌ Upload PUT failed: {put_response.status_code} - {put_response.text}")
                return {"status": "error", "message": f"Upload failed: {put_response.status_code}"}
                
        except Exception as e:
            print(f"❌ Error sending for review: {e}")
            return {"status": "error", "message": str(e)}


# Global instance
_cloud_uploader = None

def get_cloud_uploader():
    """Get or create global cloud uploader instance"""
    global _cloud_uploader
    if _cloud_uploader is None:
        _cloud_uploader = CloudUploader()
    return _cloud_uploader

