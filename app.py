"""
app.py - Traffic Sign Recognition - Premium HUD Dashboard
==========================================================
Clean rewrite: zero DOMPurify-breaking patterns.
- No CSS custom properties (--var) in inline styles
- No dangling/orphaned HTML tags across st.markdown calls
- No corrupted emoji bytes in HTML output
- All result HTML assembled in one single st.markdown call
"""

import os
import math
import base64
import tempfile
import numpy as np
import streamlit as st
from PIL import Image

from predict import load_model_and_labels, preprocess_image

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Traffic Sign Recognition | AI Dashboard",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS - single block, no CSS custom properties, no animation var() references
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background-color: #101013; min-height: 100vh; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; max-width: 1400px !important; }
section[data-testid="stSidebar"] { display: none; }

.surface-card {
    background: #1D1D22;
    border: 1px solid #33333C;
    border-radius: 16px;
    padding: 1.5rem;
}

.status-bar {
    display: flex; align-items: center; gap: 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.76rem; color: #FFC72C; letter-spacing: 1.5px;
    padding: 0.7rem 0; margin-bottom: 1.8rem;
    border-bottom: 1px solid #33333C; flex-wrap: wrap;
}
.status-ring {
    width: 16px; height: 16px; border-radius: 50%;
    border: 2px solid #52B78866;
    display: flex; align-items: center; justify-content: center;
    animation: ringPulse 2s ease-in-out infinite; flex-shrink: 0;
}
.status-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #52B788; animation: dotPulse 2s ease-in-out infinite;
}
.sep { color: #33333C; margin: 0 4px; }

@keyframes dotPulse {
    0%,100% { opacity: 1; transform: scale(1); }
    50%      { opacity: .4; transform: scale(.8); }
}
@keyframes ringPulse {
    0%,100% { border-color: #52B78855; transform: scale(1); }
    50%      { border-color: #52B788BB; transform: scale(1.3); }
}

.road-divider {
    width: 100%; height: 4px; margin: 1.8rem 0; border-radius: 2px;
    background: repeating-linear-gradient(90deg, #FFC72C 0px, #FFC72C 38px, transparent 38px, transparent 60px);
    background-size: 60px 4px;
    animation: roadScroll 1.8s linear infinite;
}
@keyframes roadScroll {
    from { background-position: 0 0; }
    to   { background-position: 60px 0; }
}

[data-testid="stFileUploader"] > div:first-child {
    background: transparent !important; border: none !important; padding: 0 !important;
}
[data-testid="stFileUploader"] label {
    color: #8B8B95 !important; font-size: 0.78rem !important;
    font-family: 'JetBrains Mono', monospace !important;
}

.viewfinder {
    position: relative; width: 100%; aspect-ratio: 4 / 3;
    background: #0D0D10; border: 1px dashed #33333C; border-radius: 12px;
    overflow: hidden; display: flex; align-items: center; justify-content: center;
    transition: border-color .3s;
}
.viewfinder:hover { border-color: #FFC72C44; }

.corner {
    position: absolute; width: 22px; height: 22px;
    border-color: #FFC72C; border-style: solid; border-width: 0;
    z-index: 10; transition: all .25s ease;
}
.viewfinder:hover .corner { width: 28px; height: 28px; }
.c-tl { top:12px;    left:12px;  border-top-width:2px;    border-left-width:2px;   border-radius:4px 0 0 0; }
.c-tr { top:12px;    right:12px; border-top-width:2px;    border-right-width:2px;  border-radius:0 4px 0 0; }
.c-bl { bottom:12px; left:12px;  border-bottom-width:2px; border-left-width:2px;   border-radius:0 0 0 4px; }
.c-br { bottom:12px; right:12px; border-bottom-width:2px; border-right-width:2px;  border-radius:0 0 4px 0; }

.vf-img { width: 100%; height: 100%; object-fit: cover; border-radius: 11px; display: block; }
.vf-placeholder { text-align: center; color: #8B8B95; pointer-events: none; }
.vf-placeholder p { font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; letter-spacing: 1px; margin-top: 0.8rem; }

.scan-line {
    position: absolute; left: 0; top: 0; width: 100%; height: 2px;
    background: linear-gradient(90deg, transparent 0%, #FFC72C 20%, #FFC72CAA 50%, #FFC72C 80%, transparent 100%);
    box-shadow: 0 0 14px 3px #FFC72C55;
    animation: scanMove 1.6s linear infinite; z-index: 20;
}
@keyframes scanMove {
    0%   { top: 0%;   opacity: 0; }
    5%   { opacity: 1; }
    95%  { opacity: 1; }
    100% { top: 100%; opacity: 0; }
}

.pipeline {
    font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;
    letter-spacing: 0.5px; color: #8B8B95;
    display: flex; align-items: center; gap: 8px; margin: 0.8rem 0;
}
.pipeline.active { color: #FFC72C; }
.pipeline.done   { color: #52B788; }
.pdot { width:6px; height:6px; border-radius:50%; background:currentColor; flex-shrink:0; }

[data-testid="stButton"] > button {
    width: 100%; font-family: 'Bebas Neue', sans-serif;
    font-size: 1.25rem; letter-spacing: 2.5px;
    border-radius: 10px; padding: 0.7rem 1.5rem;
    transition: all .2s ease; cursor: pointer; border: none;
}
[data-testid="stButton"] > button[kind="primary"] {
    background: #FFC72C; color: #101013; box-shadow: 0 4px 24px #FFC72C2A;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
    background: #FFD555; box-shadow: 0 6px 32px #FFC72C44; transform: translateY(-2px);
}
[data-testid="stButton"] > button[kind="secondary"] {
    background: transparent; color: #8B8B95; border: 1px solid #33333C !important;
}
[data-testid="stButton"] > button[kind="secondary"]:hover {
    border-color: #FFC72C55 !important; color: #FFC72C; background: #FFC72C0A;
}

.result-panel { animation: fadeUp .5s ease-out; }
@keyframes fadeUp {
    from { opacity:0; transform:translateY(16px); }
    to   { opacity:1; transform:translateY(0); }
}

.result-name {
    font-family: 'Bebas Neue', sans-serif; font-size: 2.8rem;
    letter-spacing: 3px; line-height: 1.1; margin: 0;
}

.micro-label {
    font-family: 'JetBrains Mono', monospace; font-size: 0.68rem;
    letter-spacing: 2px; text-transform: uppercase;
    color: #8B8B95; margin-bottom: 0.7rem;
}

.bar-row { margin: 0.5rem 0; }
.bar-head {
    display: flex; justify-content: space-between;
    font-size: 0.78rem; color: #8B8B95; margin-bottom: 4px;
    font-family: 'JetBrains Mono', monospace;
}
.bar-head.hi { color:#F1F1F1; font-weight:600; }
.bar-bg { background: #33333C; border-radius: 999px; height: 6px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 999px; animation: barIn .7s ease-out; }
@keyframes barIn { from { width: 0 !important; } }

.gauge-wrap { display:flex; flex-direction:column; align-items:center; }

@media (max-width: 768px) {
    .result-name { font-size: 2rem; }
}

div[data-testid="stSpinner"] > div { border-top-color: #FFC72C !important; }
[data-testid="column"] { gap: 0 !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIGN METADATA
# ══════════════════════════════════════════════════════════════════════════════
SIGN_COLOR = {
    "Stop":           "#E63946",
    "No Entry":       "#E63946",
    "Turn Left":      "#3A86FF",
    "Turn Right":     "#3A86FF",
    "Speed Limit 40": "#FFC72C",
}

# Text-only labels (no emoji) - safe for HTML output
SIGN_LABEL = {
    "Stop": "STOP", "No Entry": "NO ENTRY",
    "Turn Left": "TURN LEFT", "Turn Right": "TURN RIGHT", "Speed Limit 40": "SPEED 40",
}


# ══════════════════════════════════════════════════════════════════════════════
# SVG HELPERS - pure SVG shapes, no inline style animation
# ══════════════════════════════════════════════════════════════════════════════
def sign_icon_svg(class_name: str, size: int = 64) -> str:
    h = w = size
    cx = cy = size / 2

    if class_name == "Stop":
        pts = " ".join(
            f"{cx + (w*0.44)*math.cos(math.radians(22.5 + i*45))},"
            f"{cy + (h*0.44)*math.sin(math.radians(22.5 + i*45))}"
            for i in range(8)
        )
        fs = int(size * 0.19)
        return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">'
                f'<polygon points="{pts}" fill="#E63946" stroke="#F1F1F1" stroke-width="2.5"/>'
                f'<text x="{cx}" y="{cy+fs//3}" text-anchor="middle" fill="#F1F1F1" '
                f'font-family="sans-serif" font-size="{fs}" letter-spacing="1">STOP</text>'
                f'</svg>')

    elif class_name == "No Entry":
        r  = int(w * 0.44)
        bh = int(h * 0.19)
        bw = int(w * 0.74)
        return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">'
                f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#E63946" stroke="#F1F1F1" stroke-width="2.5"/>'
                f'<rect x="{cx - bw//2}" y="{cy - bh//2}" width="{bw}" height="{bh}" fill="#F1F1F1" rx="3"/>'
                f'</svg>')

    elif class_name == "Turn Left":
        r = int(w * 0.44)
        ax0 = int(cx * 0.30); ax1 = int(cx * 0.88)
        ay0 = int(cy * 0.82); ay1 = int(cy * 1.18)
        amid = int(cy); head = int(cx * 0.45)
        shaft_y0 = int(cy * 0.88); shaft_y1 = int(cy * 1.12)
        pts = (f"{ax0},{amid} {ax0+head},{ay0} {ax0+head},{shaft_y0} "
               f"{ax1},{shaft_y0} {ax1},{shaft_y1} {ax0+head},{shaft_y1} {ax0+head},{ay1}")
        return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">'
                f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#F1F1F1" stroke="#E63946" stroke-width="4.5"/>'
                f'<polygon points="{pts}" fill="#1D1D22"/>'
                f'</svg>')

    elif class_name == "Turn Right":
        r = int(w * 0.44)
        ax0 = int(cx * 1.70); ax1 = int(cx * 1.12)
        ay0 = int(cy * 0.82); ay1 = int(cy * 1.18)
        amid = int(cy); head = int(cx * 0.45)
        shaft_y0 = int(cy * 0.88); shaft_y1 = int(cy * 1.12)
        pts = (f"{ax0},{amid} {ax0-head},{ay0} {ax0-head},{shaft_y0} "
               f"{ax1},{shaft_y0} {ax1},{shaft_y1} {ax0-head},{shaft_y1} {ax0-head},{ay1}")
        return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">'
                f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#F1F1F1" stroke="#E63946" stroke-width="4.5"/>'
                f'<polygon points="{pts}" fill="#1D1D22"/>'
                f'</svg>')

    elif class_name == "Speed Limit 40":
        r      = int(w * 0.44)
        border = max(4, int(w * 0.10))
        fs     = int(w * 0.30)
        return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">'
                f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#E63946"/>'
                f'<circle cx="{cx}" cy="{cy}" r="{r - border}" fill="#F1F1F1"/>'
                f'<text x="{cx}" y="{cy + fs*0.35}" text-anchor="middle" fill="#101013" '
                f'font-family="sans-serif" font-size="{fs}" font-weight="bold">40</text>'
                f'</svg>')

    return ""


def gauge_svg(confidence: float, color: str) -> str:
    """
    Animated gauge using a keyframe animation embedded in a <style> tag.
    No CSS custom properties -- fully compatible with DOMPurify.
    The animation name is unique per confidence value to avoid conflicts.
    """
    r    = 70
    cx   = cy = 90
    circ = 440
    arc  = 330
    final_offset = arc * (1.0 - confidence)
    pct  = int(confidence * 100)
    # Use confidence value in animation name to make it unique per render
    anim_name = f"gaugeAnim{int(confidence * 10000)}"

    return f"""<div class="gauge-wrap">
<style>
@keyframes {anim_name} {{
  from {{ stroke-dashoffset: 330; }}
  to   {{ stroke-dashoffset: {final_offset:.1f}; }}
}}
.gauge-arc-{int(confidence * 10000)} {{
  stroke-dashoffset: 330;
  animation: {anim_name} 1.2s cubic-bezier(.25,.46,.45,.94) .15s forwards;
}}
</style>
<svg width="180" height="155" viewBox="0 0 180 155" xmlns="http://www.w3.org/2000/svg">
  <circle cx="{cx}" cy="{cy}" r="{r}"
    fill="none" stroke="#2A2A32" stroke-width="13"
    stroke-dasharray="{arc} {circ}" stroke-linecap="round"
    transform="rotate(135 {cx} {cy})"/>
  <circle cx="{cx}" cy="{cy}" r="{r}"
    fill="none" stroke="{color}" stroke-width="13"
    stroke-dasharray="{arc} {circ}" stroke-linecap="round"
    transform="rotate(135 {cx} {cy})"
    class="gauge-arc-{int(confidence * 10000)}"/>
  <text x="{cx}" y="{cy - 5}" text-anchor="middle"
    font-family="monospace" font-size="27" font-weight="700" fill="#F1F1F1">{pct}%</text>
  <text x="{cx}" y="{cy + 14}" text-anchor="middle"
    font-family="monospace" font-size="8" fill="#8B8B95" letter-spacing="2.5">CONFIDENCE</text>
</svg>
</div>"""


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
for key, default in [("uploader_key", 0), ("prediction", None), ("analyzing", False)]:
    if key not in st.session_state:
        st.session_state[key] = default


# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODEL
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading CNN model...")
def get_model_and_labels():
    try:
        m, lbl = load_model_and_labels()
        return m, lbl, None
    except FileNotFoundError as e:
        return None, None, str(e)

model, labels, load_error = get_model_and_labels()


# ══════════════════════════════════════════════════════════════════════════════
# STATUS BAR
# ══════════════════════════════════════════════════════════════════════════════
n_classes    = len(labels) if labels else 0
model_ok     = model is not None
status_color = "#52B788" if model_ok else "#E63946"
status_text  = "READY"   if model_ok else "NOT LOADED"
classes_info = f"{n_classes} CLASSES LOADED" if model_ok else "RUN python train.py"

st.markdown(f"""
<div class="status-bar">
  <div class="status-ring"><div class="status-dot"></div></div>
  <span>MODEL STATUS:&nbsp;<strong style="color:{status_color};">{status_text}</strong></span>
  <span class="sep">&#8226;</span>
  <span>{classes_info}</span>
  <span class="sep">&#8226;</span>
  <span style="color:#8B8B95;">CNN &mdash; TENSORFLOW / KERAS</span>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="margin-bottom:.2rem;">
  <span style="font-family:'Bebas Neue',sans-serif;font-size:3.8rem;letter-spacing:6px;color:#F1F1F1;line-height:1;">
    TRAFFIC SIGN
  </span>
</div>
<div style="margin-bottom:.7rem;">
  <span style="font-family:'Bebas Neue',sans-serif;font-size:3.8rem;letter-spacing:6px;color:#FFC72C;line-height:1;">
    RECOGNITION
  </span>
</div>
<p style="color:#8B8B95;font-size:.92rem;max-width:480px;margin:0;line-height:1.65;">
  Upload an image to detect and classify Indian traffic signs
  using a trained Convolutional Neural Network.
</p>
""", unsafe_allow_html=True)

st.markdown('<div class="road-divider"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL MISSING
# ══════════════════════════════════════════════════════════════════════════════
if not model_ok:
    st.markdown(f"""
<div class="surface-card" style="border-color:#E6394644;text-align:center;padding:3rem;">
  <p style="font-family:'Bebas Neue';font-size:2rem;color:#E63946;letter-spacing:3px;margin-bottom:.5rem;">
    MODEL NOT FOUND
  </p>
  <p style="color:#8B8B95;margin-bottom:1.2rem;">{load_error}</p>
  <p style="font-family:'JetBrains Mono';font-size:.85rem;color:#FFC72C;
            background:#1D1D22;display:inline-block;padding:.5rem 1.4rem;
            border-radius:8px;border:1px solid #33333C;">
    python train.py
  </p>
</div>
""", unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
col_left, col_right = st.columns([1, 1], gap="large")

# ── LEFT COLUMN ───────────────────────────────────────────────────────────────
with col_left:
    st.markdown('<p class="micro-label">CAMERA INPUT</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Select traffic sign image",
        type=["jpg", "jpeg", "png"],
        key=f"uploader_{st.session_state.uploader_key}",
        label_visibility="visible",
    )

    has_file   = uploaded_file is not None
    has_result = st.session_state.prediction is not None
    scanning   = st.session_state.analyzing

    corners = '<div class="c-tl corner"></div><div class="c-tr corner"></div><div class="c-bl corner"></div><div class="c-br corner"></div>'

    if has_file:
        ext  = uploaded_file.name.rsplit(".", 1)[-1].lower()
        mime = "image/png" if ext == "png" else "image/jpeg"
        b64  = base64.b64encode(uploaded_file.getvalue()).decode()
        scan = '<div class="scan-line"></div>' if scanning else ""
        st.markdown(f"""
<div class="viewfinder">
  {corners}{scan}
  <img src="data:{mime};base64,{b64}" class="vf-img" alt="uploaded sign"/>
</div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
<div class="viewfinder">
  {corners}
  <div class="vf-placeholder">
    <svg width="52" height="52" viewBox="0 0 52 52" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect x="4" y="11" width="44" height="32" rx="5" stroke="#33333C" stroke-width="2"/>
      <circle cx="26" cy="27" r="9" stroke="#33333C" stroke-width="2"/>
      <circle cx="26" cy="27" r="3.5" fill="#33333C"/>
      <rect x="17" y="6" width="18" height="6" rx="3" stroke="#33333C" stroke-width="2"/>
    </svg>
    <p>DRAG &amp; DROP OR BROWSE<br>
       <span style="color:#2A2A35;">JPG &nbsp;/&nbsp; JPEG &nbsp;/&nbsp; PNG</span>
    </p>
  </div>
</div>""", unsafe_allow_html=True)

    # Pipeline status
    if not has_file:
        pipe_cls, pipe_txt = "pipeline",        "AWAITING IMAGE"
    elif scanning:
        pipe_cls, pipe_txt = "pipeline active", "RUNNING CNN INFERENCE..."
    elif has_result:
        pipe_cls, pipe_txt = "pipeline done",   "PREDICTION COMPLETE"
    else:
        pipe_cls, pipe_txt = "pipeline active", "IMAGE LOADED - READY TO ANALYZE"

    st.markdown(
        f'<div class="{pipe_cls}"><div class="pdot"></div>{pipe_txt}</div>',
        unsafe_allow_html=True)

    # Buttons
    if has_file and not has_result:
        analyze = st.button("ANALYZE SIGN", use_container_width=True, key="btn_analyze", type="primary")

        if analyze:
            st.session_state.analyzing = True
            with st.spinner("Running CNN inference..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name

                    processed = preprocess_image(tmp_path)
                    all_probs = model.predict(processed, verbose=0)[0]
                    os.unlink(tmp_path)

                    top_idx   = int(np.argmax(all_probs))
                    top_conf  = float(all_probs[top_idx])
                    top_class = labels[str(top_idx)]

                    st.session_state.prediction = {
                        "class_name": top_class,
                        "confidence": top_conf,
                        "all_probs":  all_probs,
                        "labels":     labels,
                    }
                except Exception as e:
                    st.session_state.prediction = {"error": str(e)}

            st.session_state.analyzing = False
            st.rerun()

    elif has_result:
        if st.button("TRY ANOTHER IMAGE", use_container_width=True, key="btn_reset", type="secondary"):
            st.session_state.prediction   = None
            st.session_state.analyzing    = False
            st.session_state.uploader_key += 1
            st.rerun()


# ── RIGHT COLUMN ──────────────────────────────────────────────────────────────
with col_right:
    st.markdown('<p class="micro-label">DETECTION RESULT</p>', unsafe_allow_html=True)

    pred = st.session_state.prediction

    if pred is None:
        st.markdown("""
<div class="surface-card" style="min-height:360px;display:flex;flex-direction:column;
     align-items:center;justify-content:center;text-align:center;gap:1rem;">
  <svg width="60" height="60" viewBox="0 0 60 60" fill="none" xmlns="http://www.w3.org/2000/svg">
    <circle cx="30" cy="30" r="26" stroke="#2A2A32" stroke-width="2"/>
    <path d="M20 30h20M30 20v20" stroke="#2A2A32" stroke-width="2" stroke-linecap="round"/>
  </svg>
  <p style="font-family:'Bebas Neue';font-size:1.5rem;letter-spacing:3px;color:#2A2A35;margin:0;">
    NO PREDICTION YET
  </p>
  <p style="color:#8B8B95;font-size:.82rem;margin:0;">
    Upload an image and click <strong style="color:#FFC72C;">ANALYZE SIGN</strong>
  </p>
</div>
""", unsafe_allow_html=True)

    elif "error" in pred:
        err_text = pred['error'].replace('<', '&lt;').replace('>', '&gt;')
        st.markdown(f"""
<div class="surface-card" style="border-color:#E6394644;">
  <p style="font-family:'Bebas Neue';font-size:1.6rem;color:#E63946;
            letter-spacing:2px;margin-bottom:.5rem;">INFERENCE ERROR</p>
  <p style="color:#8B8B95;font-family:'JetBrains Mono';font-size:.78rem;
            word-break:break-all;">{err_text}</p>
  <p style="color:#8B8B95;font-size:.84rem;margin-top:.8rem;">
    Please upload a valid JPG or PNG image and try again.
  </p>
</div>
""", unsafe_allow_html=True)

    else:
        class_name = pred["class_name"]
        confidence = pred["confidence"]
        all_probs  = pred["all_probs"]
        lbl_map    = pred["labels"]
        color      = SIGN_COLOR.get(class_name, "#FFC72C")
        high_conf  = confidence >= 0.70

        icon_html   = sign_icon_svg(class_name, size=72)
        gauge_html  = gauge_svg(confidence, color)
        verdict_col = "#52B788" if high_conf else "#FFC72C"
        verdict_txt = "HIGH CONFIDENCE" if high_conf else "LOW CONFIDENCE"
        label_txt   = class_name.upper()

        sorted_cls = sorted(lbl_map.items(), key=lambda x: all_probs[int(x[0])], reverse=True)

        bars_html = ""
        for rank, (idx_str, cname) in enumerate(sorted_cls):
            prob      = float(all_probs[int(idx_str)])
            bar_color = SIGN_COLOR.get(cname, "#FFC72C")
            is_top    = (cname == class_name)
            head_cls  = "bar-head hi" if is_top else "bar-head"
            lbl       = SIGN_LABEL.get(cname, cname.upper())
            pct_str   = f"{prob * 100:.1f}"
            delay     = f"{rank * 0.07:.2f}"
            bars_html += f"""
<div class="bar-row">
  <div class="{head_cls}">
    <span>{lbl}</span>
    <span>{pct_str}%</span>
  </div>
  <div class="bar-bg">
    <div class="bar-fill" style="width:{pct_str}%;background:linear-gradient(90deg,{bar_color}55,{bar_color});animation-delay:{delay}s;"></div>
  </div>
</div>"""

        # Single consolidated st.markdown call - no dangling tags, no CSS variables
        st.markdown(f"""
<div class="result-panel">
  <div style="display:flex;gap:1.5rem;align-items:center;flex-wrap:wrap;">
    <div style="flex:0 0 auto;">
      {gauge_html}
    </div>
    <div style="flex:1;min-width:140px;">
      {icon_html}
      <p class="result-name" style="color:{color};margin-top:.5rem;">{label_txt}</p>
      <p style="font-family:'JetBrains Mono';font-size:.68rem;letter-spacing:1.5px;
                color:{verdict_col};margin:0;">{verdict_txt}</p>
    </div>
  </div>
  <hr style="border:none;border-top:1px solid #33333C;margin:1.2rem 0;">
  <p class="micro-label">ALL CLASS SCORES</p>
  {bars_html}
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="text-align:center;margin-top:2.5rem;padding-top:1.5rem;border-top:1px solid #33333C;">
  <p style="font-family:'JetBrains Mono';font-size:.7rem;color:#33333C;letter-spacing:2px;margin:0;">
    TRAFFIC SIGN RECOGNITION &nbsp;|&nbsp; CNN + TENSORFLOW &nbsp;|&nbsp;
    INDIAN IRC STANDARD &nbsp;|&nbsp; COLLEGE MINI PROJECT
  </p>
</div>
""", unsafe_allow_html=True)
