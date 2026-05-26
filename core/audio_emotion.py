import numpy as np
import librosa
import threading

_result = {"label": "Neutral", "score": 0.10, "rms": 0.0}
_buffer = []
_lock   = threading.Lock()

SAMPLE_RATE = 16000

def get_result():
    return _result

def process_audio_frame(audio: np.ndarray):
    """Called from webrtc callback in app.py"""
    global _buffer, _result

    with _lock:
        _buffer.extend(audio.tolist())

        if len(_buffer) >= SAMPLE_RATE * 5:
            y = np.array(_buffer[:SAMPLE_RATE * 5], dtype=np.float32)
            _buffer = []

            rms      = float(np.sqrt(np.mean(y ** 2)))
            zcr      = float(np.mean(librosa.feature.zero_crossing_rate(y)))
            centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=SAMPLE_RATE)))
            mfccs    = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=13)
            mfcc_mean = float(np.mean(mfccs[1]))

            if rms < 0.01:
                label, score = "Calm", 0.05
            elif rms < 0.03 and zcr < 0.05:
                label, score = "Neutral", 0.10
            elif rms > 0.10 and centroid > 3000:
                label, score = "Angry", 0.65
            elif rms > 0.07 and zcr > 0.10:
                label, score = "Stressed", 0.55
            elif mfcc_mean < -20:
                label, score = "Sad", 0.45
            elif rms > 0.05 and centroid > 2000:
                label, score = "Happy", 0.20
            else:
                label, score = "Neutral", 0.10

            rms_intensity = min(rms / 0.12, 1.0)
            distress = round(min(score + (score * rms_intensity * 0.4), 1.0), 3)
            _result  = {"label": label, "score": distress, "rms": rms}
