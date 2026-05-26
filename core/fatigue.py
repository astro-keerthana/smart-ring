def detect_fatigue(rmssd_now, rmssd_baseline, hr_now, hr_baseline):
    """
    HRV trend-based fatigue detection.
    Baseline: PhysioNet / wearable-hrv population thresholds.
    """
    decline_pct = ((rmssd_baseline - rmssd_now) / rmssd_baseline) * 100
    hr_rise_pct = ((hr_now - hr_baseline) / hr_baseline) * 100

    if   rmssd_now > 45 and decline_pct < 10:
        level, score, severity = "Alert",             0.10, "Normal"
    elif rmssd_now > 35 and decline_pct < 25:
        level, score, severity = "Mild Fatigue",      0.38, "Mild"
    elif rmssd_now > 25 and decline_pct < 40:
        level, score, severity = "Moderate Fatigue",  0.65, "Moderate"
    else:
        level, score, severity = "High Fatigue",      0.88, "High"

    return {
        "label"        : level,
        "score"        : score,
        "severity"     : severity,
        "decline_pct"  : round(decline_pct, 1),
        "hr_rise_pct"  : round(hr_rise_pct, 1),
        "rmssd_now"    : rmssd_now,
        "rmssd_baseline": rmssd_baseline,
    }
