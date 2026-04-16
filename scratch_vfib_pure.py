import json
import math

def get_stats(record_id):
    with open(f'api_data/record_{record_id}.json') as f:
        data = json.load(f)['data']
    str_data = data.get('lead1_reading') or data.get('lead2_reading')
    sig = [float(x) for x in str_data.split(',') if x.strip()]
    
    mean = sum(sig)/len(sig)
    std = math.sqrt(sum((x - mean)**2 for x in sig)/len(sig))
    
    diff = [abs(sig[i] - sig[i-1]) for i in range(1, len(sig))]
    # flat ratio is percentage of diffs that are tiny (e.g. less than 10 counts)
    flat_ratio_10 = sum(1 for x in diff if x < 10.0) / len(diff)
    
    # or less than 0.1 * std (which is ~20 to 30)
    thresh = 0.1 * std
    flat_ratio_std = sum(1 for x in diff if x < thresh) / len(diff)
    
    print(f"Record {record_id} -> Flat(10): {flat_ratio_10:.3f}, Flat(Std): {flat_ratio_std:.3f}")

get_stats(297) # VFib
get_stats(167) # VFib 1
get_stats(168) # VFib 2
get_stats(100) # Normal
get_stats(280) # Normal 2
