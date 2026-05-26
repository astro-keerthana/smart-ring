def get_suggestions(stress, fatigue, spo2, hr, temp_c, cycle_phase):
    """
    Returns list of dicts: {text, priority}
    priority: 'high' | 'normal'
    """
    tips = []

    # Stress
    if stress["score"] > 0.65:
        tips += [
            {"text": "Practice 4-7-8 breathing: inhale 4s, hold 7s, exhale 8s. "
                     "Repeat 4 cycles.", "priority": "high"},
            {"text": "Step away from your current task for 10 minutes. "
                     "Brief disengagement reduces cortisol.", "priority": "high"},
            {"text": "A 15-minute walk significantly reduces sympathetic "
                     "nervous system activation.", "priority": "normal"},
        ]
    elif stress["score"] > 0.35:
        tips += [
            {"text": "Stress is mildly elevated. Consider a short mindfulness "
                     "pause before your next task.", "priority": "normal"},
            {"text": "Ensure adequate hydration — dehydration amplifies "
                     "perceived stress.", "priority": "normal"},
        ]
    else:
        tips += [
            {"text": "Stress levels are within healthy range. "
                     "Current physiological state is well-regulated.", "priority": "normal"},
        ]

    # Fatigue
    if fatigue["score"] > 0.65:
        tips += [
            {"text": "HRV has declined consistently over 3 days. "
                     "Prioritise 7-9 hours of sleep tonight.", "priority": "high"},
            {"text": "Avoid caffeine after 14:00 to protect sleep architecture "
                     "and allow HRV recovery.", "priority": "normal"},
        ]
    elif fatigue["score"] > 0.35:
        tips += [
            {"text": "Mild fatigue detected. A 20-minute nap between 13:00-15:00 "
                     "can restore alertness without disrupting night sleep.", "priority": "normal"},
        ]

    # SpO2
    if spo2 < 95:
        tips += [
            {"text": f"SpO2 reading of {spo2}% is below optimal. "
                     "Try slow deep breathing and reposition the ring for accurate contact.",
             "priority": "high"},
        ]

    # Heart Rate
    if hr > 100:
        tips += [
            {"text": f"Resting heart rate of {hr} BPM is elevated. "
                     "Sit quietly for 5 minutes and avoid stimulants.",
             "priority": "high"},
        ]
    elif hr < 50:
        tips += [
            {"text": f"Heart rate of {hr} BPM is unusually low. "
                     "Verify ring contact and check for dizziness.",
             "priority": "high"},
        ]

    # Temperature
    if temp_c > 37.5:
        tips += [
            {"text": f"Skin temperature of {temp_c}C is elevated. "
                     "Monitor for fever symptoms and ensure adequate hydration.",
             "priority": "high"},
        ]

    # Cycle
    if cycle_phase == "Luteal":
        tips += [
            {"text": "Luteal phase typically brings lower HRV and higher stress "
                     "sensitivity. Reduce high-intensity training this week.", "priority": "normal"},
            {"text": "Increase magnesium-rich foods (dark leafy greens, nuts, seeds) "
                     "to support mood regulation during luteal phase.", "priority": "normal"},
        ]
    elif cycle_phase == "Ovulatory":
        tips += [
            {"text": "Ovulatory phase is peak performance window. "
                     "Ideal time for high-intensity workouts and demanding tasks.", "priority": "normal"},
        ]
    elif cycle_phase == "Menstrual":
        tips += [
            {"text": "Menstrual phase: prioritise restorative movement such as "
                     "yoga or walking. Avoid high-intensity training.", "priority": "normal"},
        ]

    return tips
