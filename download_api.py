import os
import json
import time
import urllib.request
import urllib.error

API_URL = "https://deckmount.in/ankur_bhaiya.php?id="
OUTPUT_DIR = "api_data"
MAX_RECORDS = 299

def download_api_data():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Beginning download of {MAX_RECORDS} records to '{OUTPUT_DIR}/' ...")
    
    success_count = 0
    fail_count = 0
    
    for i in range(1, MAX_RECORDS + 1):
        url = f"{API_URL}{i}"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status == 200:
                    raw_data = response.read().decode('utf-8')
                    try:
                        data = json.loads(raw_data)
                    except json.JSONDecodeError:
                        print(f"[{i}/{MAX_RECORDS}] ❌ Invalid JSON payload.")
                        fail_count += 1
                        continue
                        
                    if data.get("status"):
                        filename = f"record_{i}.json"
                        filepath = os.path.join(OUTPUT_DIR, filename)
                        with open(filepath, "w") as f:
                            json.dump(data, f, indent=2)
                        success_count += 1
                        
                        patient_name = data.get("data", {}).get("name", "Unknown")
                        print(f"[{i}/{MAX_RECORDS}] ✅ Saved {filename} - Patient: {patient_name}")
                    else:
                        print(f"[{i}/{MAX_RECORDS}] ⚠️ No valid data found in response.")
                        fail_count += 1
                else:
                    print(f"[{i}/{MAX_RECORDS}] ❌ HTTP Error {response.status}")
                    fail_count += 1
                    
        except urllib.error.URLError as e:
            print(f"[{i}/{MAX_RECORDS}] ❌ URLError: {e.reason}")
            fail_count += 1
        except Exception as e:
            print(f"[{i}/{MAX_RECORDS}] ❌ Exception: {e}")
            fail_count += 1
            
        time.sleep(0.1)

    print("\nDownload Complete!")
    print(f"Successfully downloaded: {success_count}")
    print(f"Failed/Missing: {fail_count}")

if __name__ == "__main__":
    download_api_data()
