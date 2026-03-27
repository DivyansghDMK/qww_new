"""ECG constants and configuration"""

# History and buffer configuration
HISTORY_LENGTH = 10000
NORMAL_HR_MIN, NORMAL_HR_MAX = 60, 100

# Lead labels
LEAD_LABELS = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
]

# Lead colors for visualization
LEAD_COLORS = {
    "I": "#00ff99",
    "II": "#ff0055", 
    "III": "#0099ff",
    "aVR": "#ff9900",
    "aVL": "#cc00ff",
    "aVF": "#00ccff",
    "V1": "#ffcc00",
    "V2": "#00ffcc",
    "V3": "#ff6600",
    "V4": "#6600ff",
    "V5": "#00b894",
    "V6": "#ff0066"
}

# Leads mapping for different test types
LEADS_MAP = {
    "Lead II ECG Test": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
    "Lead III ECG Test": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
    "7 Lead ECG Test": ["V1", "V2", "V3", "V4", "V5", "V6", "II"],
    "12 Lead ECG Test": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
    "ECG Live Monitoring": ["II"]
}
