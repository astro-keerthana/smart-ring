import numpy as np
import librosa
import threading

_lock   = threading.Lock()
_latest = {"label": "Not Recorded", "score": 0.05,
           "rms": 0.0, "zcr": 0.0, "centroid": 0.0}


def analyse(y: np.ndarray, sr: int = 16000):
    global _latest

    if y is None or len(y) < 100:
        return

    # Normalise
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    rms       = float(np.sqrt(np.mean(y ** 2)))
    zcr       = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    centroid  = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    mfccs     = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = float(np.mean(mfccs[1]))

    if rms < 0.005:
        label, score = "No Voice", 0.05
    elif rms < 0.02:
        label, score = "Calm", 0.05
    elif rms < 0.08 and zcr < 0.05:
        label, score = "Neutral", 0.10
    elif rms > 0.35 and centroid > 3000:
        label, score = "Angry", 0.65
    elif rms > 0.25 and zcr > 0.10:
        label, score = "Stressed", 0.55
    elif mfcc_mean < -20:
        label, score = "Sad", 0.45
    elif rms > 0.15 and centroid > 2000:
        label, score = "Happy", 0.20
    else:
        label, score = "Neutral", 0.10

    rms_intensity = min(rms / 0.4, 1.0)
    distress      = round(min(score + (score * rms_intensity * 0.4), 1.0), 3)

    with _lock:
        _latest = {
            "label"   : label,
            "score"   : distress,
            "rms"     : round(rms, 4),
            "zcr"     : round(zcr, 4),
            "centroid": round(centroid, 1),
        }


def get_result():
    with _lock:
        return dict(_latest)
