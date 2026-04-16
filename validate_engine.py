import os
import json
import numpy as np
from src.ecg.arrhythmia_detector import ArrhythmiaDetector

DATA_DIR = "api_data"

# Map API keys to internal lead names
LEAD_MAP = {
    "lead1_reading": "I",
    "lead2_reading": "II",
    "lead3_reading": "III",
    "leadV1_reading": "V1",
    "leadV2_reading": "V2",
    "leadV3_reading": "V3",
    "leadV4_reading": "V4",
    "leadV5_reading": "V5",
    "leadV6_reading": "V6",
    "leadavl_reading": "aVL",
    "leadavr_reading": "aVR",
    "leadavf_reading": "aVF"
}

def load_and_test():
    if not os.path.exists(DATA_DIR):
        print(f"Directory {DATA_DIR} not found!")
        return

    detector = ArrhythmiaDetector(sampling_rate=500.0, counts_per_mv=1.0)
    
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json")]
    files.sort(key=lambda x: int(x.split('_')[1].split('.')[0])) # sort by ID

    print(f"Found {len(files)} records. Starting validation suite...\n")
    print("-" * 60)
    
    match_count = 0
    processed_count = 0

    for filename in files:
        filepath = os.path.join(DATA_DIR, filename)
        with open(filepath, "r") as f:
            payload = json.load(f)
            
        data = payload.get("data", {})
        if not data:
            continue
            
        patient_name = data.get("name", "Unknown")
        expected = data.get("conclusion", "[]")
        api_id = data.get("id", "Unknown")
        
        # Build lead dict
        signal_dict = {}
        for api_key, internal_name in LEAD_MAP.items():
            reading_str = data.get(api_key, "")
            if reading_str:
                # Convert comma string to float array
                try:
                    signal_dict[internal_name] = np.array([float(x) for x in reading_str.split(",") if x.strip()], dtype=float)
                except ValueError:
                    pass
                    
        if not signal_dict:
            continue

        try:
            # Run our new engine
            # Note: primary_signal 'signal' is required as first arg but unused if lead_signals overrides
            primary = signal_dict.get("II", np.array([]))
            results = detector.detect_arrhythmias(
                signal=primary, 
                analysis={}, 
                lead_signals=signal_dict
            )
            
            print(f"ID {api_id} | Patient: {patient_name}")
            print(f"  EXPECTED (API):   {expected}")
            print(f"  DETECTED (ENGINE): {results}")
            
            # Simple heuristic check if the expected patient name/conclusion is inside our detected strings
            # This is loose because API 'conclusion' field formatting varies
            expected_lower = str(expected).lower()
            engine_str = str(results).lower()
            
            # Very loose matching for terminal reporting
            if any(label.lower() in expected_lower for label in results) or any(expect.strip(' \'\"[]').lower() in engine_str for expect in expected.split(',')):
                 match_count += 1
            elif patient_name.lower().strip() in engine_str:
                 # Usually patient_name is the actual arrhythmia name in test sets!
                 match_count += 1

            processed_count += 1
            print("-" * 60)
            
        except Exception as e:
            print(f"ID {api_id} | Exception during processing: {e}")
            print("-" * 60)

    print(f"\nValidation Complete!")
    print(f"Processed: {processed_count} files.")
    
if __name__ == "__main__":
    load_and_test()
