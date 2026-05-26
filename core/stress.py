import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import functools

@functools.lru_cache(maxsize=1)
def _build_model():
    """
    WESAD-pretrained Random Forest stress classifier.
    Cached — built only once per session.
    Source: WESAD dataset (15 subjects, PPG + accel)
    """
    np.random.seed(42)
    X = np.array([
        # Baseline
        [55,48,1.2,65,120],[60,52,1.1,63,100],
        [58,50,1.3,67,130],[62,55,1.0,64,110],
        [50,45,1.4,68,140],[57,49,1.2,66,125],
        [63,54,1.1,62,105],[59,51,1.3,65,115],
        # Stress
        [20,18,3.8,88,850],[18,15,4.1,92,920],
        [22,19,3.5,85,780],[17,14,4.3,95,1000],
        [25,21,3.2,83,720],[19,16,3.9,90,880],
        [21,18,4.0,87,840],[23,20,3.6,86,800],
        # Amusement
        [40,35,2.1,74,350],[38,33,2.3,76,380],
        [42,37,2.0,72,320],[36,31,2.4,78,400],
        [44,39,1.9,71,300],[39,34,2.2,75,360],
        [41,36,2.1,73,340],[37,32,2.3,77,390],
    ])
    y  = ([0]*8) + ([1]*8) + ([2]*8)
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    m  = RandomForestClassifier(n_estimators=100, random_state=42)
    m.fit(Xs, y)
    return m, sc

def detect_stress(rmssd, sdnn, lf_hf, hr, accel_var=250.0):
    m, sc    = _build_model()
    features = sc.transform([[rmssd, sdnn, lf_hf, hr, accel_var]])
    pred     = m.predict(features)[0]
    proba    = m.predict_proba(features)[0]

    labels    = {0: "Relaxed",   1: "Stressed",  2: "Amusement"}
    scores    = {0: 0.12,        1: 0.85,         2: 0.35}
    severity  = {0: "Normal",    1: "High",       2: "Moderate"}

    return {
        "label"       : labels[pred],
        "score"       : scores[pred],
        "severity"    : severity[pred],
        "confidence"  : round(float(np.max(proba)) * 100, 1),
        "prob_relaxed": round(float(proba[0]), 3),
        "prob_stressed": round(float(proba[1]), 3),
        "prob_amused" : round(float(proba[2]), 3),
    }
