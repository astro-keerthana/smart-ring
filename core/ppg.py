import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

def process_ppg(hr_bpm, spo2, rmssd, sdnn, lf_hf):
    """
    Accepts slider inputs (simulating ring BLE output).
    Returns structured vitals dict with validity flags.
    """
    hr_valid   = 40 <= hr_bpm <= 180
    spo2_valid = 70 <= spo2   <= 100
    temp_valid = True

    hr_status   = "Normal"   if 60 <= hr_bpm <= 100 else \
                  "Elevated" if hr_bpm > 100 else "Low"
    spo2_status = "Healthy"  if spo2 >= 95 else \
                  "Low"      if spo2 >= 90 else "Critical"
    hrv_status  = "Good"     if rmssd >= 40 else \
                  "Reduced"  if rmssd >= 25 else "Suppressed"
    lfhf_status = "Balanced" if lf_hf <= 2.5 else "Elevated"

    return {
        "hr_bpm"       : hr_bpm,
        "spo2_pct"     : spo2,
        "rmssd_ms"     : rmssd,
        "sdnn_ms"      : sdnn,
        "lf_hf_ratio"  : lf_hf,
        "hr_status"    : hr_status,
        "spo2_status"  : spo2_status,
        "hrv_status"   : hrv_status,
        "lfhf_status"  : lfhf_status,
        "hr_valid"     : hr_valid,
        "spo2_valid"   : spo2_valid,
    }
