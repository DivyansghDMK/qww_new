import os
import json
from pathlib import Path
from datetime import datetime
import sys

# Ensure we can import from src/
sys.path.append(os.path.join(os.getcwd(), "src"))

try:
    from utils.cloud_uploader import get_cloud_uploader
    print("✅ Successfully imported cloud_uploader")
except ImportError as e:
    print(f"❌ Could not import cloud_uploader: {e}")
    sys.exit(1)

def upload_test_file():
    # 1. Create a dummy test JSON file
    test_data = {
        "test_id": "diagnostic_upload_test",
        "timestamp": datetime.now().isoformat(),
        "status": "verifying_s3_connection",
        "message": "This is a test file to verify JSON uploads are working correctly."
    }
    
    # We name it with 'metric' in the name so the uploader's current filter accepts it
    test_filename = f"diagnostic_metric_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    test_path = os.path.join(os.getcwd(), test_filename)
    
    try:
        with open(test_path, 'w') as f:
            json.dump(test_data, f, indent=2)
        print(f"📝 Created test file: {test_filename}")
        
        # 2. Get uploader and check config
        uploader = get_cloud_uploader()
        if not uploader.is_configured():
            print("❌ Cloud uploader is NOT configured. Check your .env file.")
            return

        print(f"☁️ Attempting upload to {uploader.cloud_service} ({uploader.s3_bucket})...")
        
        # 3. Perform upload
        result = uploader.upload_report(test_path, metadata={"patient_name": "TEST_DIAGNOSTIC"})
        
        if result.get('status') == 'success':
            print(f"🚀 SUCCESS! File uploaded successfully.")
            print(f"🔗 URL: {result.get('url')}")
        else:
            print(f"❌ UPLOAD FAILED: {result.get('message', 'Unknown error')}")
            
    finally:
        # 4. Cleanup
        if os.path.exists(test_path):
            os.remove(test_path)
            print(f"🗑️ Cleaned up local test file.")

if __name__ == "__main__":
    upload_test_file()