from datetime import datetime, timedelta

def detect_cycle(cycle_day, cycle_len, period_len,
                 temp_now, temp_base,
                 hr_now,   hr_base,
                 hrv_now,  hrv_base):
    """
    Cosinor + threshold model.
    Source: cilab-ufersa/period_cycle_prediction + Oura Research
    """
    temp_rise      = round(temp_now - temp_base, 2)
    temp_shifted   = temp_rise > 0.2
    hr_elevated    = (hr_now  - hr_base)  > 3
    hrv_suppressed = (hrv_base - hrv_now) > 10
    days_left      = cycle_len - cycle_day
    next_period    = datetime.today() + timedelta(days=max(days_left, 0))

    if   cycle_day <= period_len:
        phase, conf, desc = "Menstrual",  92, "Rest and gentle movement recommended"
    elif cycle_day <= 13 and not temp_shifted:
        phase, conf, desc = "Follicular", 85, "Rising energy — good for new activities"
    elif cycle_day in range(13, 16) and not temp_shifted:
        phase, conf, desc = "Ovulatory",  78, "Peak energy and focus phase"
    elif temp_shifted and hr_elevated and hrv_suppressed:
        phase, conf, desc = "Luteal",     88, "Expect lower energy and mood sensitivity"
    else:
        phase, conf, desc = "Follicular", 72, "Rising energy — good for new activities"

    return {
        "phase"       : phase,
        "confidence"  : conf,
        "description" : desc,
        "cycle_day"   : cycle_day,
        "cycle_len"   : cycle_len,
        "days_left"   : max(days_left, 0),
        "next_period" : next_period.strftime("%b %d, %Y"),
        "temp_rise"   : temp_rise,
        "temp_shifted": temp_shifted,
        "hr_elevated" : hr_elevated,
        "hrv_suppressed": hrv_suppressed,
    }
