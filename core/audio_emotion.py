import numpy as np
import sounddevice as sd
import librosa
import threading

SAMPLE_RATE = 22050
DURATION    = 5

# ── Shared state ────────────────────────────────────────────
_recording_buffer = None
_is_recording     = False
_record_thread    = None


def _record_worker():
    global _recording_buffer, _is_recording
    _is_recording     = True
    _recording_buffer = None
    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate = SAMPLE_RATE,
        channels   = 1,
        dtype      = "float32"
    )
    sd.wait()
    _recording_buffer = audio.flatten()
    _is_recording     = False


def start_recording():
    """Launch recording in background thread. Non-blocking."""
    global _record_thread, _is_recording, _recording_buffer
    _recording_buffer = None
    _is_recording     = True
    _record_thread    = threading.Thread(target=_record_worker, daemon=True)
    _record_thread.start()


def is_recording():
    return _is_recording


def get_result():
    """Returns analyzed result if recording is done, else None."""
    if _recording_buffer is not None and not _is_recording:
        return analyze_emotion(_recording_buffer)
    return None


def analyze_emotion(audio: np.ndarray):
    rms      = float(np.sqrt(np.mean(audio ** 2)))
    zcr      = float(np.mean(librosa.feature.zero_crossing_rate(audio)))
    centroid = float(np.mean(
        librosa.feature.spectral_centroid(y=audio, sr=SAMPLE_RATE)
    ))
    mfccs    = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=13)
    mfcc_mean = float(np.mean(mfccs[1]))

    envelope    = np.abs(audio)
    peaks       = np.sum(np.diff((envelope > np.mean(envelope)).astype(int)) > 0)
    speech_rate = peaks / DURATION

    if rms < 0.005:
        label, score, severity = "No Voice", 0.05, "None"
    elif rms > 0.08 and zcr > 0.12 and centroid > 2500:
        label, score, severity = "Angry",    0.90, "High"
    elif rms > 0.06 and zcr > 0.10:
        label, score, severity = "Fearful",  0.75, "High"
    elif rms > 0.05 and zcr > 0.09 and centroid > 2000:
        label, score, severity = "Happy",    0.30, "Low"
    elif rms < 0.02 and zcr < 0.06:
        label, score, severity = "Sad",      0.65, "Moderate"
    elif rms < 0.03 and zcr < 0.08:
        label, score, severity = "Calm",     0.05, "Low"
    else:
        label, score, severity = "Neutral",  0.10, "Normal"

    return {
        "label"      : label,
        "score"      : score,
        "severity"   : severity,
        "rms"        : round(rms, 5),
        "zcr"        : round(zcr, 4),
        "centroid"   : round(centroid, 1),
        "speech_rate": round(speech_rate, 1),
        "mfcc_mean"  : round(mfcc_mean, 3),
    }
