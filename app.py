# ============================================================
#  SMART RING  — Professional Health Dashboard
#  Run: streamlit run app.py
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import av

from core.ppg         import process_ppg
from core.stress      import detect_stress
from core.fatigue     import detect_fatigue
from core.cycle       import detect_cycle
from core.suggestions import get_suggestions
from core.audio_emotion import get_result, process_audio_frame
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title            = "Smart Ring",
    page_icon             = "assets/favicon.ico" if Path("assets/favicon.ico").exists() else None,
    layout                = "wide",
    initial_sidebar_state = "expanded"
)

# Load CSS
css_path = Path("assets/style.css")
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────────────────────

CHART_DEFAULTS = dict(
    paper_bgcolor = "rgba(0,0,0,0)",
    plot_bgcolor  = "rgba(0,0,0,0)",
    margin        = dict(l=10, r=10, t=10, b=10),
    font          = dict(family="Inter", color="#94a3b8", size=11),
)

def make_gauge(value, label, color, max_val=100):
    fig = go.Figure(go.Indicator(
        mode   = "gauge+number",
        value  = value,
        title  = {"text": label,
                  "font": {"color": "#94a3b8", "size": 13,
                           "family": "Inter"}},
        number = {"font": {"color": color, "size": 30,
                           "family": "Inter"},
                  "suffix": ""},
        gauge  = {
            "axis"       : {"range": [0, max_val],
                            "tickcolor": "#1e2d45",
                            "tickfont" : {"color": "#334155", "size": 9},
                            "nticks"   : 5},
            "bar"        : {"color": color, "thickness": 0.22},
            "bgcolor"    : "#0c1220",
            "borderwidth": 0,
            "steps"      : [
                {"range": [0,            max_val*0.35], "color": "#0d1f12"},
                {"range": [max_val*0.35, max_val*0.65], "color": "#1f1a0d"},
                {"range": [max_val*0.65, max_val],      "color": "#1f0d0d"},
            ],
            "threshold"  : {
                "line"     : {"color": color, "width": 2},
                "thickness": 0.75,
                "value"    : value
            }
        }
    ))
    fig.update_layout(height=200, margin=dict(t=30, b=0, l=10, r=10),
                      **{k: v for k, v in CHART_DEFAULTS.items() if k != "margin"})
    return fig


def make_trend(history, key, color):
    times  = [h["label"] for h in history]
    values = [h[key]     for h in history]

    r = int(color[1:3], 16)
    g = int(color[3:5], 16)
    b = int(color[5:7], 16)
    fill_rgba = f"rgba({r},{g},{b},0.08)"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x         = times,
        y         = values,
        mode      = "lines+markers",
        line      = dict(color=color, width=1.8, shape="spline"),
        marker    = dict(size=5, color=color,
                         line=dict(color="#0f1829", width=1.5)),
        fill      = "tozeroy",
        fillcolor = fill_rgba,
    ))
    fig.update_layout(
        height = 140,
        xaxis  = dict(showgrid=False, tickfont=dict(size=9)),
        yaxis  = dict(gridcolor="#1e2d45", tickfont=dict(size=9)),
        **CHART_DEFAULTS
    )
    return fig


def make_bar_probs(relaxed, stressed, amused):
    fig = go.Figure(go.Bar(
        x                 = ["Relaxed", "Stressed", "Amusement"],
        y                 = [relaxed*100, stressed*100, amused*100],
        marker_color      = ["#10b981", "#ef4444", "#f59e0b"],
        marker_line_width = 0,
        text              = [f"{relaxed*100:.0f}%",
                             f"{stressed*100:.0f}%",
                             f"{amused*100:.0f}%"],
        textposition      = "outside",
        textfont          = dict(size=10, color="#94a3b8"),
    ))
    fig.update_layout(
        height = 170,
        xaxis  = dict(showgrid=False, tickfont=dict(size=10)),
        yaxis  = dict(gridcolor="#1e2d45", tickfont=dict(size=9),
                      range=[0, 115]),
        **CHART_DEFAULTS
    )
    return fig


def make_cycle_donut(day, total, color):
    remaining = max(total - day, 0)
    fig = go.Figure(go.Pie(
        values        = [day, remaining],
        labels        = ["Elapsed", "Remaining"],
        hole          = 0.72,
        marker_colors = [color, "#1e2d45"],
        textinfo      = "none",
        hoverinfo     = "label+value",
        direction     = "clockwise",
        sort          = False,
    ))
    fig.add_annotation(
        text      = f"<b>Day {day}</b><br><span style='font-size:10px'>"
                    f"of {total}</span>",
        x=0.5, y=0.5,
        font      = dict(size=16, color="#f1f5f9", family="Inter"),
        showarrow = False,
        align     = "center"
    )
    fig.update_layout(
        height     = 200,
        showlegend = False,
        **CHART_DEFAULTS
    )
    return fig


# ─────────────────────────────────────────────────────────────
# COLOR HELPERS
# ─────────────────────────────────────────────────────────────

def score_color(score):
    if score < 0.35:   return "#10b981"
    elif score < 0.65: return "#f59e0b"
    else:              return "#ef4444"

def score_class(score):
    if score < 0.35:   return "status-normal"
    elif score < 0.65: return "status-warning"
    else:              return "status-critical"

def score_label(score):
    if score < 0.35:   return "Normal"
    elif score < 0.65: return "Moderate"
    else:              return "High"


# ─────────────────────────────────────────────────────────────
# SESSION STATE DEFAULTS
# ─────────────────────────────────────────────────────────────

if "audio_result" not in st.session_state:
    st.session_state.audio_result = {
        "label"   : "Not Recorded",
        "score"   : 0.05,
        "rms"     : 0.0,
        "zcr"     : 0.0,
        "centroid": 0.0,
    }


# ─────────────────────────────────────────────────────────────
# WEBRTC AUDIO CALLBACK
# ─────────────────────────────────────────────────────────────

def audio_frame_callback(frame: av.AudioFrame):
    audio = frame.to_ndarray().flatten().astype(np.float32)
    if len(audio) > 0 and np.max(np.abs(audio)) > 1.0:
        audio = audio / 32768.0
    process_audio_frame(audio)

    # Pull latest result into session state
    result = get_result()
    if result is not None:
        st.session_state.audio_result = result

    return frame


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='padding:8px 0 16px 0;'>
        <div style='font-size:1.0rem; font-weight:700;
                    color:#f1f5f9; letter-spacing:-0.02em;'>
            Smart Ring
        </div>
        <div style='font-size:0.72rem; color:#475569; margin-top:2px;'>
            Sensor Input Simulator
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("VITALS")
    hr     = st.slider("Heart Rate (BPM)",  40,   160,  72)
    spo2   = st.slider("SpO2 (%)",          85,   100,  98)
    rmssd  = st.slider("HRV RMSSD (ms)",     5,    80,  38)
    sdnn   = st.slider("HRV SDNN (ms)",      5,    70,  32)
    lf_hf  = st.slider("LF/HF Ratio",      0.5,   5.0, 2.1, step=0.1)
    temp_c = st.slider("Skin Temp (C)",    35.0,  40.0, 36.7, step=0.1)

    st.markdown("<hr style='border-color:#1e2d45; margin:12px 0;'>",
                unsafe_allow_html=True)
    st.markdown("FATIGUE BASELINE (3 DAYS AGO)")
    rmssd_3d = st.slider("RMSSD Baseline (ms)", 20,  80, 52)
    hr_3d    = st.slider("HR Baseline (BPM)",   50, 110, 68)

    st.markdown("<hr style='border-color:#1e2d45; margin:12px 0;'>",
                unsafe_allow_html=True)
    st.markdown("CYCLE PARAMETERS")
    cycle_day  = st.slider("Current Cycle Day",      1,   35, 29)
    cycle_len  = st.slider("Average Cycle Length",  21,   35, 28)
    period_len = st.slider("Period Duration (days)", 2,    8,  5)
    temp_base  = st.slider("Follicular Temp (C)",  35.5, 37.0, 36.4, step=0.1)
    hr_base    = st.slider("Follicular HR (BPM)",   50,   90, 71)
    hrv_base   = st.slider("Follicular HRV (ms)",   20,   80, 52)

    st.markdown("<hr style='border-color:#1e2d45; margin:12px 0;'>",
                unsafe_allow_html=True)
    st.markdown("🎙️ **VOICE EMOTION**")
    st.caption("Click Start → Allow microphone → speak naturally")

    webrtc_streamer(
        key                        = "audio-emotion",
        mode                       = WebRtcMode.SENDONLY,
        audio_frame_callback       = audio_frame_callback,
        media_stream_constraints   = {"audio": True, "video": False},
        async_processing           = True,
    )

    ar = st.session_state.audio_result
    st.markdown(f"""
    <div style='background:#0f1829; border:1px solid #1e2d45;
                border-radius:8px; padding:12px; margin-top:8px;'>
        <div style='font-size:0.68rem; color:#475569;
                    text-transform:uppercase; letter-spacing:0.08em;'>
            Detected
        </div>
        <div style='font-size:1.1rem; font-weight:700;
                    color:#f1f5f9; margin-top:4px;'>
            {ar["label"]}
        </div>
        <div style='font-size:0.72rem; color:#64748b; margin-top:4px;'>
            RMS: {ar["rms"]} &nbsp;|&nbsp; ZCR: {ar.get("zcr", 0.0)}
        </div>
    </div>
    """, unsafe_allow_html=True)

    emotion_map = {
        "Not Recorded": 0.05, "No Voice": 0.05,
        "Calm"        : 0.05, "Neutral" : 0.10,
        "Happy"       : 0.05, "Sad"     : 0.65,
        "Fearful"     : 0.75, "Angry"   : 0.90,
    }
    distress_score = emotion_map.get(ar["label"], 0.05)
    emotion_sel    = ar["label"]


# ─────────────────────────────────────────────────────────────
# COMPUTE
# ─────────────────────────────────────────────────────────────

vitals  = process_ppg(hr, spo2, rmssd, sdnn, lf_hf)
stress  = detect_stress(rmssd, sdnn, lf_hf, hr)
fatigue = detect_fatigue(rmssd, rmssd_3d, hr, hr_3d)
cycle   = detect_cycle(
    cycle_day, cycle_len, period_len,
    temp_c, temp_base, hr, hr_base, rmssd, hrv_base
)
suggestions = get_suggestions(
    stress, fatigue, spo2, hr, temp_c, cycle["phase"]
)

hrv_history = [
    {"label": "3d ago AM", "rmssd": rmssd_3d,                  "hr": hr_3d},
    {"label": "3d ago PM", "rmssd": round(rmssd_3d * 0.92, 1), "hr": hr_3d + 3},
    {"label": "2d ago AM", "rmssd": round(rmssd_3d * 0.84, 1), "hr": hr_3d + 6},
    {"label": "2d ago PM", "rmssd": round(rmssd_3d * 0.76, 1), "hr": hr_3d + 9},
    {"label": "Today",     "rmssd": rmssd,                      "hr": hr},
]

overall = (stress["score"] + fatigue["score"] + distress_score) / 3


# ─────────────────────────────────────────────────────────────
# RENDER
# ─────────────────────────────────────────────────────────────

# ── Header ──────────────────────────────────────────────────
st.markdown("""
<div class='page-header'>
    <div class='page-title'>Smart Ring — Health Dashboard</div>
    <div class='page-subtitle'>
        Real-time physiological monitoring  |  AI-assisted analysis
    </div>
</div>
""", unsafe_allow_html=True)

# ── Status Banner ────────────────────────────────────────────
if overall > 0.65:
    bc, bd, bt, bs = (
        "banner-critical", "banner-dot-critical",
        "Attention Required",
        "Elevated stress and fatigue detected. "
        "Immediate rest and recovery is recommended."
    )
elif overall > 0.35:
    bc, bd, bt, bs = (
        "banner-warning", "banner-dot-warning",
        "Moderate Alert",
        "Physiological indicators suggest mild stress accumulation. "
        "Monitor and take short recovery breaks."
    )
else:
    bc, bd, bt, bs = (
        "banner-normal", "banner-dot-normal",
        "All Systems Normal",
        "All physiological indicators are within healthy ranges. "
        "Current state is well-regulated."
    )

st.markdown(f"""
<div class='banner {bc}'>
    <div class='banner-dot {bd}'></div>
    <div>
        <div class='banner-text-title'>{bt}</div>
        <div class='banner-text-sub'>{bs}</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Section: Vitals ──────────────────────────────────────────
st.markdown("<div class='section-label'>Vitals</div>",
            unsafe_allow_html=True)

v1, v2, v3, v4, v5 = st.columns(5)

def vital_card(col, label, value, unit, status_text, accent):
    col.markdown(f"""
    <div class='metric-card metric-card-{"green" if accent=="#10b981"
        else "yellow" if accent=="#f59e0b" else "red"}'>
        <div class='metric-card-label'>{label}</div>
        <div class='metric-card-value' style='color:{accent};'>{value}</div>
        <div class='metric-card-unit'>{unit}</div>
        <div class='metric-card-status {"status-normal" if accent=="#10b981"
            else "status-warning" if accent=="#f59e0b"
            else "status-critical"}'>{status_text}</div>
    </div>
    """, unsafe_allow_html=True)

vital_card(v1, "Heart Rate",
           hr, "BPM",
           vitals["hr_status"],
           "#10b981" if vitals["hr_status"] == "Normal" else "#f59e0b")

vital_card(v2, "SpO2",
           f"{spo2}%", "Blood Oxygen",
           vitals["spo2_status"],
           "#10b981" if vitals["spo2_status"] == "Healthy" else "#ef4444")

vital_card(v3, "Skin Temperature",
           f"{temp_c}", "Celsius",
           "Normal" if 36.0 <= temp_c <= 37.5 else "Elevated",
           "#10b981" if 36.0 <= temp_c <= 37.5 else "#f59e0b")

vital_card(v4, "HRV RMSSD",
           rmssd, "Milliseconds",
           vitals["hrv_status"],
           "#10b981" if rmssd >= 40 else
           "#f59e0b" if rmssd >= 25 else "#ef4444")

vital_card(v5, "LF/HF Ratio",
           lf_hf, "Autonomic Balance",
           vitals["lfhf_status"],
           "#10b981" if lf_hf <= 2.5 else "#f59e0b")

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)


# ── Section: AI Analysis ─────────────────────────────────────
st.markdown("<div class='section-label'>AI Analysis</div>",
            unsafe_allow_html=True)

a1, a2, a3, a4 = st.columns(4)

with a1:
    sc = score_color(stress["score"])
    st.plotly_chart(
        make_gauge(int(stress["score"] * 100), "Stress Level", sc),
        use_container_width=True
    )
    st.markdown(f"""
    <div class='analysis-card'>
        <div class='analysis-card-title'>Stress Detection</div>
        <div class='analysis-card-result' style='color:{sc};'>
            {stress["label"]}
        </div>
        <div class='analysis-card-sub'>
            Confidence: {stress["confidence"]}% &nbsp;|&nbsp;
            WESAD Random Forest
        </div>
    </div>""", unsafe_allow_html=True)

with a2:
    fc = score_color(fatigue["score"])
    st.plotly_chart(
        make_gauge(int(fatigue["score"] * 100), "Fatigue Level", fc),
        use_container_width=True
    )
    st.markdown(f"""
    <div class='analysis-card'>
        <div class='analysis-card-title'>Fatigue Detection</div>
        <div class='analysis-card-result' style='color:{fc};'>
            {fatigue["label"]}
        </div>
        <div class='analysis-card-sub'>
            HRV decline: {fatigue["decline_pct"]}% over 3 days
        </div>
    </div>""", unsafe_allow_html=True)

with a3:
    dc = score_color(distress_score)
    st.plotly_chart(
        make_gauge(int(distress_score * 100), "Voice Distress", dc),
        use_container_width=True
    )
    st.markdown(f"""
    <div class='analysis-card'>
        <div class='analysis-card-title'>Audio Emotion</div>
        <div class='analysis-card-result' style='color:{dc};'>
            {emotion_sel}
        </div>
        <div class='analysis-card-sub'>
            librosa &nbsp;|&nbsp; RMS + ZCR + MFCC features
        </div>
    </div>""", unsafe_allow_html=True)

with a4:
    st.plotly_chart(
        make_bar_probs(
            stress["prob_relaxed"],
            stress["prob_stressed"],
            stress["prob_amused"]
        ),
        use_container_width=True
    )
    st.markdown(f"""
    <div class='analysis-card'>
        <div class='analysis-card-title'>Stress Class Distribution</div>
        <div class='analysis-card-sub' style='margin-top:4px;'>
            Relaxed {stress["prob_relaxed"]*100:.0f}% &nbsp;|&nbsp;
            Stressed {stress["prob_stressed"]*100:.0f}% &nbsp;|&nbsp;
            Amusement {stress["prob_amused"]*100:.0f}%
        </div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)


# ── Section: Trends + Cycle ───────────────────────────────────
st.markdown("<div class='section-label'>Trends and Cycle</div>",
            unsafe_allow_html=True)

t1, t2, t3 = st.columns([2, 2, 2])

with t1:
    st.markdown("<div style='font-size:0.72rem; color:#475569; "
                "margin-bottom:6px;'>HRV RMSSD — 3 Day Trend</div>",
                unsafe_allow_html=True)
    st.plotly_chart(
        make_trend(hrv_history, "rmssd", "#3b82f6"),
        use_container_width=True
    )
    st.markdown("<div style='font-size:0.72rem; color:#475569; "
                "margin-bottom:6px; margin-top:8px;'>"
                "Heart Rate — 3 Day Trend</div>",
                unsafe_allow_html=True)
    st.plotly_chart(
        make_trend(hrv_history, "hr", "#8b5cf6"),
        use_container_width=True
    )

with t2:
    hours  = ["06:00","08:00","10:00","12:00","14:00","16:00","18:00","Now"]
    s_vals = [0.15, 0.22, 0.41, 0.55, 0.48, 0.62, 0.70,
              round(stress["score"], 2)]
    fig_s  = go.Figure()
    fig_s.add_trace(go.Scatter(
        x         = hours,
        y         = s_vals,
        mode      = "lines+markers",
        line      = dict(color="#ef4444", width=1.8, shape="spline"),
        marker    = dict(size=5, color="#ef4444",
                         line=dict(color="#0f1829", width=1.5)),
        fill      = "tozeroy",
        fillcolor = "rgba(239,68,68,0.08)",
    ))
    fig_s.update_layout(
        height = 140,
        xaxis  = dict(showgrid=False, tickfont=dict(size=9)),
        yaxis  = dict(gridcolor="#1e2d45", tickfont=dict(size=9),
                      range=[0, 1.1]),
        **CHART_DEFAULTS
    )
    st.markdown("<div style='font-size:0.72rem; color:#475569; "
                "margin-bottom:6px;'>Stress Score — Today</div>",
                unsafe_allow_html=True)
    st.plotly_chart(fig_s, use_container_width=True)

    spo2_vals = [98, 97, 98, 97, 96, 97, 98, spo2]
    fig_o     = go.Figure()
    fig_o.add_trace(go.Scatter(
        x         = hours,
        y         = spo2_vals,
        mode      = "lines+markers",
        line      = dict(color="#10b981", width=1.8, shape="spline"),
        marker    = dict(size=5, color="#10b981",
                         line=dict(color="#0f1829", width=1.5)),
        fill      = "tozeroy",
        fillcolor = "rgba(16,185,129,0.08)",
    ))
    fig_o.update_layout(
        height = 140,
        xaxis  = dict(showgrid=False, tickfont=dict(size=9)),
        yaxis  = dict(gridcolor="#1e2d45", tickfont=dict(size=9),
                      range=[85, 102]),
        **CHART_DEFAULTS
    )
    st.markdown("<div style='font-size:0.72rem; color:#475569; "
                "margin-bottom:6px; margin-top:8px;'>"
                "SpO2 — Today</div>",
                unsafe_allow_html=True)
    st.plotly_chart(fig_o, use_container_width=True)

with t3:
    st.markdown("<div style='font-size:0.72rem; color:#475569; "
                "margin-bottom:6px;'>Menstrual Cycle</div>",
                unsafe_allow_html=True)
    st.plotly_chart(
        make_cycle_donut(cycle_day, cycle_len, "#8b5cf6"),
        use_container_width=True
    )
    st.markdown(f"""
    <div class='cycle-info-card'>
        <div class='cycle-phase-label'>{cycle["phase"]} Phase</div>
        <div class='cycle-meta'>
            Confidence: {cycle["confidence"]}% &nbsp;|&nbsp;
            cilab-ufersa + Oura Research
        </div>
        <div class='cycle-meta' style='margin-top:4px; color:#94a3b8;'>
            {cycle["description"]}
        </div>
        <div class='cycle-next'>
            Next period: <strong>{cycle["next_period"]}</strong>
            &nbsp; ({cycle["days_left"]} days remaining)
            <br>
            <span style='font-size:0.75rem; color:#475569;'>
                Temp rise: +{cycle["temp_rise"]}C above follicular baseline
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)


# ── Section: Suggestions ─────────────────────────────────────
st.markdown("<div class='section-label'>Personalised Recommendations</div>",
            unsafe_allow_html=True)

high_tips   = [t for t in suggestions if t["priority"] == "high"]
normal_tips = [t for t in suggestions if t["priority"] == "normal"]

if high_tips:
    s1, s2 = st.columns(2)
    cols_h  = [s1, s2]
    for i, tip in enumerate(high_tips):
        cols_h[i % 2].markdown(f"""
        <div class='suggestion-card suggestion-card-priority'>
            <div style='font-size:0.68rem; font-weight:700;
                        color:#ef4444; text-transform:uppercase;
                        letter-spacing:0.06em; margin-bottom:6px;'>
                Priority
            </div>
            {tip["text"]}
        </div>""", unsafe_allow_html=True)

if normal_tips:
    n_cols = st.columns(3)
    for i, tip in enumerate(normal_tips):
        n_cols[i % 3].markdown(f"""
        <div class='suggestion-card'>
            {tip["text"]}
        </div>""", unsafe_allow_html=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)


# ── Section: Raw Data ─────────────────────────────────────────
with st.expander("Raw Sensor Data — Developer View"):
    raw_df = pd.DataFrame([
        {"Metric": "Heart Rate",    "Value": hr,
         "Unit": "BPM",   "Range": "60–100",    "Status": "Normal" if 60 <= hr <= 100 else "Check"},
        {"Metric": "SpO2",          "Value": spo2,
         "Unit": "%",     "Range": "95–100",    "Status": "Healthy" if spo2 >= 95 else "Low"},
        {"Metric": "HRV RMSSD",     "Value": rmssd,
         "Unit": "ms",    "Range": "40–80",     "Status": "Good" if rmssd >= 40 else "Reduced"},
        {"Metric": "HRV SDNN",      "Value": sdnn,
         "Unit": "ms",    "Range": "30–70",     "Status": "Good" if sdnn >= 30 else "Reduced"},
        {"Metric": "LF/HF Ratio",   "Value": lf_hf,
         "Unit": "ratio", "Range": "1.0–2.5",   "Status": "Balanced" if lf_hf <= 2.5 else "Elevated"},
        {"Metric": "Skin Temp",     "Value": temp_c,
         "Unit": "C",     "Range": "36.0–37.5", "Status": "Normal" if 36 <= temp_c <= 37.5 else "Elevated"},
        {"Metric": "Stress Score",  "Value": stress["score"],
         "Unit": "0–1",   "Range": "< 0.35",    "Status": stress["label"]},
        {"Metric": "Fatigue Score", "Value": fatigue["score"],
         "Unit": "0–1",   "Range": "< 0.35",    "Status": fatigue["label"]},
        {"Metric": "Distress Score","Value": distress_score,
         "Unit": "0–1",   "Range": "< 0.35",    "Status": emotion_sel},
        {"Metric": "Cycle Day",     "Value": cycle_day,
         "Unit": "day",   "Range": f"1–{cycle_len}", "Status": cycle["phase"]},
    ])
    st.dataframe(raw_df, use_container_width=True, hide_index=True)


# ── Footer ────────────────────────────────────────────────────
st.markdown("""
<div style='padding:24px 0 8px 0; text-align:center;
            font-size:0.72rem; color:#334155;
            border-top:1px solid #1e2d45; margin-top:8px;'>
    Smart Ring &nbsp;&nbsp;|&nbsp;&nbsp;
    WESAD Random Forest &nbsp;&nbsp;|&nbsp;&nbsp;
    librosa Audio Features &nbsp;&nbsp;|&nbsp;&nbsp;
    PhysioNet HRV Baselines &nbsp;&nbsp;|&nbsp;&nbsp;
    cilab-ufersa Cycle Model
</div>
""", unsafe_allow_html=True)
