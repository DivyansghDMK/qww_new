"""
Backend API Module with Offline-First Architecture
Handles all backend communication with automatic offline queuing
"""

import requests
import json
from datetime import datetime
from typing import Dict, Any, Optional
import os
from dotenv import load_dotenv
from .offline_queue import get_offline_queue

load_dotenv()


class BackendAPI:
    """
    Handle all backend communication with offline-first approach
    
    Features:
    - Automatic offline detection
    - Queue data when offline
    - Auto-sync when connection restored
    - No data loss
    """
    
    def __init__(self):
        self.base_url = os.getenv('BACKEND_API_URL', 'http://localhost:3000/api/v1')
        self.api_key = os.getenv('BACKEND_API_KEY')
        self.enabled = os.getenv('BACKEND_UPLOAD_ENABLED', 'false').lower() == 'true'
        self.token = None
        self.session_id = None
        self.offline_queue = get_offline_queue()
        
        print(f"🔌 Backend API initialized:")
        print(f"   URL: {self.base_url}")
        print(f"   Enabled: {self.enabled}")
        print(f"   Offline queue: {self.offline_queue.queue_dir}")
    
    def is_enabled(self) -> bool:
        """Check if backend upload is enabled"""
        return self.enabled
    
    def set_token(self, token: str):
        """Set JWT token for authenticated requests"""
        self.token = token
    
    def _headers(self) -> Dict[str, str]:
        """Get request headers"""
        headers = {'Content-Type': 'application/json'}
        if self.token:
            headers['Authorization'] = f'Bearer {self.token}'
        elif self.api_key:
            headers['X-API-Key'] = self.api_key
        return headers
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        Make HTTP request with offline handling
        
        Returns response JSON or queues for later if offline
        """
        if not self.enabled:
            return {"status": "disabled", "message": "Backend upload is disabled"}
        
        url = f'{self.base_url}/{endpoint}'
        
        try:
            # Check if online
            if not self.offline_queue.is_online():
                return {"status": "queued", "message": "Offline - data queued for sync"}
            
            # Make request
            kwargs['headers'] = self._headers()
            kwargs.setdefault('timeout', 10)
            
            response = requests.request(method, url, **kwargs)
            
            if response.status_code in [200, 201]:
                return response.json() if response.content else {"status": "success"}
            else:
                return {
                    "status": "error",
                    "code": response.status_code,
                    "message": response.text
                }
                
        except requests.exceptions.ConnectionError:
            return {"status": "queued", "message": "Connection error - data queued for sync"}
        except requests.exceptions.Timeout:
            return {"status": "queued", "message": "Request timeout - data queued for sync"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def register_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Register a new user
        Note: Registration requires internet - cannot be queued
        """
        if not self.offline_queue.is_online():
            return {
                "status": "error",
                "message": "Registration requires internet connection"
            }
        
        return self._make_request('POST', 'auth/register', json=user_data)
    
    def login(self, identifier: str, password: str) -> Dict[str, Any]:
        """
        Login user
        Note: Login requires internet - cannot be queued
        """
        if not self.offline_queue.is_online():
            return {
                "status": "error",
                "message": "Login requires internet connection"
            }
        
        result = self._make_request(
            'POST',
            'auth/login',
            json={'identifier': identifier, 'password': password}
        )
        
        if result.get('status') == 'success' and 'token' in result:
            self.set_token(result['token'])
        
        return result
    
    def start_session(self, device_serial: str, device_info: Dict) -> str:
        """Start a new recording session (with offline queuing)"""
        payload = {
            'device_serial': device_serial,
            'device_info': device_info
        }
        
        result = self._make_request('POST', 'sessions/start', json=payload)
        
        if result.get('status') == 'queued':
            # Queue for later sync
            self.offline_queue.queue_data('session_start', payload, priority=1)
            # Generate local session ID
            self.session_id = f"offline_session_{int(datetime.utcnow().timestamp())}"
        elif result.get('status') == 'success':
            self.session_id = result.get('session_id')
        
        return self.session_id
    
    def upload_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Upload real-time metrics (with offline queuing)"""
        if not self.session_id:
            return {"status": "error", "message": "No active session"}
        
        payload = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'metrics': metrics,
            'session_id': self.session_id
        }
        
        result = self._make_request(
            'POST',
            f'sessions/{self.session_id}/metrics',
            json=payload
        )
        
        if result.get('status') == 'queued':
            # Queue for later sync (low priority - metrics are frequent)
            self.offline_queue.queue_data('metrics', payload, priority=7)
        
        return result
    
    def upload_waveform(self, leads_data: Dict[str, list], sampling_rate: int) -> Dict[str, Any]:
        """Upload ECG waveform data (with offline queuing)"""
        if not self.session_id:
            return {"status": "error", "message": "No active session"}
        
        payload = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'sampling_rate': sampling_rate,
            'leads': leads_data,
            'session_id': self.session_id
        }
        
        result = self._make_request(
            'POST',
            f'sessions/{self.session_id}/waveform',
            json=payload
        )
        
        if result.get('status') == 'queued':
            # Queue for later sync (medium priority)
            self.offline_queue.queue_data('waveform', payload, priority=5)
        
        return result
    
    def upload_report(self, pdf_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Upload generated PDF report (with offline queuing)"""
        if not os.path.exists(pdf_path):
            return {"status": "error", "message": "PDF file not found"}
        
        try:
            # Check if online for immediate upload
            if self.offline_queue.is_online() and self.enabled:
                with open(pdf_path, 'rb') as f:
                    files = {'file': f}
                    data = {'metadata': json.dumps(metadata)}
                    
                    headers = {}
                    if self.token:
                        headers['Authorization'] = f'Bearer {self.token}'
                    elif self.api_key:
                        headers['X-API-Key'] = self.api_key
                    
                    response = requests.post(
                        f'{self.base_url}/reports/upload',
                        files=files,
                        data=data,
                        headers=headers,
                        timeout=30
                    )
                    
                    if response.status_code in [200, 201]:
                        return response.json() if response.content else {"status": "success"}
            
            # Queue for later if offline or upload failed
            payload = {
                'file_path': pdf_path,
                'metadata': metadata,
                'session_id': self.session_id
            }
            self.offline_queue.queue_data('report', payload, priority=2)  # High priority
            
            return {
                "status": "queued",
                "message": "Report queued for upload when online"
            }
            
        except Exception as e:
            # Queue on any error
            payload = {
                'file_path': pdf_path,
                'metadata': metadata,
                'session_id': self.session_id
            }
            self.offline_queue.queue_data('report', payload, priority=2)
            
            return {
                "status": "queued",
                "message": f"Upload error - queued for retry: {str(e)}"
            }
    
    def end_session(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """End current session (with offline queuing)"""
        if not self.session_id:
            return {"status": "error", "message": "No active session"}
        
        payload = {
            'summary': summary,
            'session_id': self.session_id
        }
        
        result = self._make_request(
            'POST',
            f'sessions/{self.session_id}/end',
            json=payload
        )
        
        if result.get('status') == 'queued':
            # Queue for later sync (high priority)
            self.offline_queue.queue_data('session_end', payload, priority=3)
        
        # Clear session ID
        session_id = self.session_id
        self.session_id = None
        
        return result
    
    def get_user_sessions(self, user_id: str) -> Dict[str, Any]:
        """Get user's session history (requires internet)"""
        if not self.offline_queue.is_online():
            return {
                "status": "error",
                "message": "Cannot retrieve sessions - no internet connection"
            }
        
        return self._make_request('GET', f'users/{user_id}/sessions')
    
    def get_session_data(self, session_id: str) -> Dict[str, Any]:
        """Get session data (requires internet)"""
        if not self.offline_queue.is_online():
            return {
                "status": "error",
                "message": "Cannot retrieve session data - no internet connection"
            }
        
        return self._make_request('GET', f'sessions/{session_id}/data')
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get offline queue statistics"""
        return self.offline_queue.get_stats()
    
    def force_sync(self) -> None:
        """Force immediate sync of queued data"""
        self.offline_queue.force_sync_now()
    
    def retry_failed(self) -> int:
        """Retry all failed uploads"""
        return self.offline_queue.retry_failed_items()


# Global instance
_backend_api = None

def get_backend_api() -> BackendAPI:
    """Get or create global backend API instance"""
    global _backend_api
    if _backend_api is None:
        _backend_api = BackendAPI()
    return _backend_api

