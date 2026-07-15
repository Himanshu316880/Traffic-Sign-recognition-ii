"""
app.py — Traffic Sign Recognition — Premium HUD Dashboard
==========================================================
This is a PURE UI redesign. Zero ML logic was changed.

All prediction logic is imported unchanged from predict.py:
  - load_model_and_labels()   → loads model + class labels
  - preprocess_image()        → resizes + normalises image (same as training)

The only additions here are layout, styling, and display logic.

DESIGN CONCEPT: Vehicle HUD / ADAS Dashboard
  Colors   : #101013 bg · #1D1D22 surface · #FFC72C accent · #E63946 danger
  Fonts    : Bebas Neue (headings) · Inter (body) · JetBrains Mono (numbers)
  Features : Animated road divider · Viewfinder with corner brackets ·
             Scanning line · SVG speedometer gauge · Inline SVG sign icons ·
             Animated confidence bars · Ghost reset button
"""

import os
import math
import base64
import tempfile
import numpy as np
import streamlit as st
from PIL import Image

# ── ML logic imported unchanged from predict.py ───────────────────────────────
from predict import load_model_and_labels, preprocess_image

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Traffic Sign Recognition | AI Dashboard",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════════════════════════
# ALL CSS — one block, organised into named sections
# Every section is commented so it's easy to understand during a viva.
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>

/* ── TYPOGRAPHY ────────────────────────────────────────────────────────────────
   Load three Google Fonts:
   · Bebas Neue  — bold condensed headings (dashboard labels)
   · Inter       — clean body text
   · JetBrains Mono — monospaced font for status text and numbers
   ---------------------------------------------------------------------------- */
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── LAYOUT — page background and Streamlit padding ───────────────────────────
   Set the entire app background to near-black #101013.
   Reduce default Streamlit container padding so our cards fill the space.
   ---------------------------------------------------------------------------- */
.stApp {
    background-color: #101013;
    min-height: 100vh;
}
/* Hide Streamlit default top bar / footer / hamburger menu */
#MainMenu, footer, header { visibility: hidden; }
/* Reduce default padding; allow wider content */
.block-container {
    padding-top: 1.5rem  !important;
    padding-bottom: 2rem !important;
    max-width: 1400px    !important;
}
/* Remove sidebar entirely for this design */
section[data-testid="stSidebar"] { display: none; }


/* ── CARDS — reusable dark surface with subtle border ─────────────────────────
   Every panel (upload, results) sits on a #1D1D22 "surface" card.
   The border is #33333C — visible but not distracting.
   ---------------------------------------------------------------------------- */
.surface-card {
    background  : #1D1D22;
    border      : 1px solid #33333C;
    border-radius: 16px;
    padding     : 1.5rem;
}


/* ── STATUS BAR — animated live indicator at the very top ────────────────────
   Uses JetBrains Mono for that technical readout feel.
   The green dot and ring both pulse using CSS @keyframes.
   ---------------------------------------------------------------------------- */
.status-bar {
    display      : flex;
    align-items  : center;
    gap          : 10px;
    font-family  : 'JetBrains Mono', monospace;
    font-size    : 0.76rem;
    color        : #FFC72C;
    letter-spacing: 1.5px;
    padding      : 0.7rem 0;
    margin-bottom: 1.8rem;
    border-bottom: 1px solid #33333C;
    flex-wrap    : wrap;
}
.status-ring {
    width: 16px; height: 16px;
    border-radius: 50%;
    border: 2px solid #52B78866;
    display: flex; align-items: center; justify-content: center;
    animation: ringPulse 2s ease-in-out infinite;
    flex-shrink: 0;
}
.status-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #52B788;
    animation: dotPulse 2s ease-in-out infinite;
}
.sep { color: #33333C; margin: 0 4px; }

@keyframes dotPulse {
    0%,100% { opacity: 1; transform: scale(1);   }
    50%      { opacity: .4; transform: scale(.8); }
}
@keyframes ringPulse {
    0%,100% { border-color: #52B78855; transform: scale(1);   }
    50%      { border-color: #52B788BB; transform: scale(1.3); }
}


/* ── ROAD DIVIDER — animated highway lane-marking strip ───────────────────────
   repeating-linear-gradient draws yellow dashes.
   background-position animation makes them scroll rightward continuously,
   giving the illusion of driving past road markings.
   ---------------------------------------------------------------------------- */
.road-divider {
    width         : 100%;
    height        : 4px;
    margin        : 1.8rem 0;
    border-radius : 2px;
    background    : repeating-linear-gradient(
        90deg,
        #FFC72C  0px,
        #FFC72C  38px,
        transparent 38px,
        transparent 60px
    );
    background-size: 60px 4px;
    animation     : roadScroll 1.8s linear infinite;
}
@keyframes roadScroll {
    from { background-position: 0 0;    }
    to   { background-position: 60px 0; }
}


/* ── UPLOADER / VIEWFINDER ─────────────────────────────────────────────────────
   Transforms the file upload area into a camera viewfinder:
   · Dark background  → feels like a live camera feed
   · Dashed border    → classic targeting reticle
   · Yellow corners   → vehicle HUD style corner brackets
   · Hover glow       → interactive feedback
   ---------------------------------------------------------------------------- */

/* Hide Streamlit's default uploader styling */
[data-testid="stFileUploader"] > div:first-child {
    background: transparent !important;
    border    : none        !important;
    padding   : 0           !important;
}
[data-testid="stFileUploader"] label {
    color    : #8B8B95 !important;
    font-size: 0.78rem !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* Outer frame */
.viewfinder {
    position      : relative;
    width         : 100%;
    aspect-ratio  : 4 / 3;
    background    : #0D0D10;
    border        : 1px dashed #33333C;
    border-radius : 12px;
    overflow      : hidden;
    display       : flex;
    align-items   : center;
    justify-content: center;
    transition    : border-color .3s;
}
.viewfinder:hover { border-color: #FFC72C44; }

/* Corner bracket — base style shared by all four corners */
.corner {
    position     : absolute;
    width        : 22px; height: 22px;
    border-color : #FFC72C;
    border-style : solid;
    border-width : 0;
    z-index      : 10;
    transition   : all .25s ease;
}
.viewfinder:hover .corner { width: 28px; height: 28px; }

/* Each corner shows only two of its four sides */
.c-tl { top:12px;    left:12px;  border-top-width:2px;    border-left-width:2px;   border-radius:4px 0 0 0; }
.c-tr { top:12px;    right:12px; border-top-width:2px;    border-right-width:2px;  border-radius:0 4px 0 0; }
.c-bl { bottom:12px; left:12px;  border-bottom-width:2px; border-left-width:2px;   border-radius:0 0 0 4px; }
.c-br { bottom:12px; right:12px; border-bottom-width:2px; border-right-width:2px;  border-radius:0 0 4px 0; }

/* Image inside viewfinder — covers the full frame */
.vf-img {
    width        : 100%;
    height       : 100%;
    object-fit   : cover;
    border-radius: 11px;
    display      : block;
}

/* Placeholder shown when no image is loaded */
.vf-placeholder {
    text-align    : center;
    color         : #8B8B95;
    pointer-events: none;
}
.vf-placeholder p {
    font-family   : 'JetBrains Mono', monospace;
    font-size     : 0.75rem;
    letter-spacing: 1px;
    margin-top    : 0.8rem;
}


/* ── SCANNING LINE — CSS-only animation overlaid on the image ─────────────────
   A thin glowing yellow line moves from top to bottom continuously.
   It is only injected into the DOM during the "analyzing" state.
   Uses position:absolute so it floats above the image.
   ---------------------------------------------------------------------------- */
.scan-line {
    position  : absolute;
    left      : 0; top: 0;
    width     : 100%; height: 2px;
    background: linear-gradient(90deg,
        transparent 0%, #FFC72C 20%, #FFC72CAA 50%, #FFC72C 80%, transparent 100%);
    box-shadow: 0 0 14px 3px #FFC72C55;
    animation : scanMove 1.6s linear infinite;
    z-index   : 20;
}
@keyframes scanMove {
    0%   { top: 0%;   opacity: 0; }
    5%   { opacity: 1;            }
    95%  { opacity: 1;            }
    100% { top: 100%; opacity: 0; }
}


/* ── PIPELINE STATUS LINE ──────────────────────────────────────────────────────
   Monospaced text that shows the current stage: Awaiting → Loaded → Done.
   ---------------------------------------------------------------------------- */
.pipeline {
    font-family   : 'JetBrains Mono', monospace;
    font-size     : 0.75rem;
    letter-spacing: 0.5px;
    color         : #8B8B95;
    display       : flex;
    align-items   : center;
    gap           : 8px;
    margin        : 0.8rem 0;
}
.pipeline.active { color: #FFC72C; }
.pipeline.done   { color: #52B788; }
.pdot { width:6px; height:6px; border-radius:50%; background:currentColor; flex-shrink:0; }


/* ── BUTTONS ───────────────────────────────────────────────────────────────────
   Analyze: large, yellow, dark text — the primary call to action.
   Reset  : ghost (transparent background), muted colour — secondary action.

   We use wrapper divs (.btn-primary / .btn-ghost) because Streamlit generates
   its own button HTML; we target that inner button via CSS.
   ---------------------------------------------------------------------------- */
[data-testid="stButton"] > button {
    width        : 100%;
    font-family  : 'Bebas Neue', sans-serif;
    font-size    : 1.25rem;
    letter-spacing: 2.5px;
    border-radius: 10px;
    padding      : 0.7rem 1.5rem;
    transition   : all .2s ease;
    cursor       : pointer;
    border       : none;
}

[data-testid="stButton"] > button[kind="primary"] {
    background  : #FFC72C;
    color       : #101013;
    box-shadow  : 0 4px 24px #FFC72C2A;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
    background  : #FFD555;
    box-shadow  : 0 6px 32px #FFC72C44;
    transform   : translateY(-2px);
}
[data-testid="stButton"] > button[kind="primary"]:active { transform: translateY(0); }

[data-testid="stButton"] > button[kind="secondary"] {
    background  : transparent;
    color       : #8B8B95;
    border      : 1px solid #33333C !important;
}
[data-testid="stButton"] > button[kind="secondary"]:hover {
    border-color: #FFC72C55 !important;
    color       : #FFC72C;
    background  : #FFC72C0A;
}


/* ── GAUGE — circular SVG speedometer ─────────────────────────────────────────
   The .gauge-fill circle starts with stroke-dashoffset=330 (= empty arc).
   CSS animates it to the value stored in --final-offset (a CSS custom property
   set via inline style by Python at render time).
   This produces the "filling speedometer" effect on page load.
   ---------------------------------------------------------------------------- */
.gauge-wrap { display:flex; flex-direction:column; align-items:center; }
.gauge-fill {
    stroke-dashoffset: 330;
    animation: gaugeIn 1.2s cubic-bezier(.25,.46,.45,.94) .15s forwards;
}
@keyframes gaugeIn {
    from { stroke-dashoffset: 330; }
    to   { stroke-dashoffset: var(--final-offset, 0); }
}


/* ── RESULT PANEL — fade + slide up when results appear ───────────────────────
   Creates a smooth entrance so results don't just pop in abruptly.
   ---------------------------------------------------------------------------- */
.result-panel { animation: fadeUp .5s ease-out; }
@keyframes fadeUp {
    from { opacity:0; transform:translateY(16px); }
    to   { opacity:1; transform:translateY(0);     }
}

/* Big class name displayed in the result panel */
.result-name {
    font-family   : 'Bebas Neue', sans-serif;
    font-size     : 2.8rem;
    letter-spacing: 3px;
    line-height   : 1.1;
    margin        : 0;
}

/* Section micro-label above groups of content */
.micro-label {
    font-family   : 'JetBrains Mono', monospace;
    font-size     : 0.68rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color         : #8B8B95;
    margin-bottom : 0.7rem;
}


/* ── CONFIDENCE BARS — horizontal bars per class ──────────────────────────────
   The fill div's width (%) is set inline by Python.
   Animates in with a slight delay using animation-delay staggered per bar.
   ---------------------------------------------------------------------------- */
.bar-row { margin: 0.5rem 0; }
.bar-head {
    display        : flex;
    justify-content: space-between;
    font-size      : 0.78rem;
    color          : #8B8B95;
    margin-bottom  : 4px;
    font-family    : 'JetBrains Mono', monospace;
}
.bar-head.hi { color:#F1F1F1; font-weight:600; }
.bar-bg {
    background   : #33333C;
    border-radius: 999px;
    height       : 6px;
    overflow     : hidden;
}
.bar-fill {
    height       : 100%;
    border-radius: 999px;
    animation    : barIn .7s ease-out;
}
@keyframes barIn {
    from { width: 0 !important; }
}


/* ── RESPONSIVE — stack on narrow / mobile screens ────────────────────────────
   Gauge and result name scale down on small screens so nothing overflows.
   ---------------------------------------------------------------------------- */
@media (max-width: 768px) {
    .result-name { font-size: 2rem; }
    .gauge-wrap svg { width: 140px !important; height: 130px !important; }
}


/* ── MISC STREAMLIT OVERRIDES ─────────────────────────────────────────────────
   Target stable data-testid selectors rather than generated class names.
   ---------------------------------------------------------------------------- */
/* Spinner colour */
div[data-testid="stSpinner"] > div { border-top-color: #FFC72C !important; }
/* Remove extra gap between columns */
[data-testid="column"] { gap: 0 !important; }

</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SIGN METADATA — colour per class (matches design spec)
# ═══════════════════════════════════════════════════════════════════════════════
SIGN_COLOR = {
    "Stop":           "#E63946",
    "No Entry":       "#E63946",
    "Turn Left":      "#3A86FF",
    "Turn Right":     "#3A86FF",
    "Speed Limit 40": "#FFC72C",
}
SIGN_EMOJI = {
    "Stop": "🛑", "No Entry": "⛔",
    "Turn Left": "⬅️", "Turn Right": "➡️", "Speed Limit 40": "🔢",
}


# ═══════════════════════════════════════════════════════════════════════════════
# INLINE SVG SIGN ICONS — no external assets, pure SVG shapes
# Each function draws the actual Indian IRC-standard sign shape.
# ═══════════════════════════════════════════════════════════════════════════════
def sign_icon_svg(class_name: str, size: int = 64) -> str:
    """
    Returns a minimal inline SVG icon for a traffic sign class.
    Uses only basic SVG shapes: polygon, circle, rect, text.
    """
    h = w = size
    cx = cy = size / 2

    if class_name == "Stop":
        # Red octagon (8-sided polygon) with STOP text
        pts = " ".join(
            f"{cx + (w*0.44)*math.cos(math.radians(22.5 + i*45))},"
            f"{cy + (h*0.44)*math.sin(math.radians(22.5 + i*45))}"
            for i in range(8)
        )
        fs = int(size * 0.19)
        return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">'
                f'<polygon points="{pts}" fill="#E63946" stroke="#F1F1F1" stroke-width="2.5"/>'
                f'<text x="{cx}" y="{cy+fs//3}" text-anchor="middle" fill="#F1F1F1" '
                f'font-family="Bebas Neue,sans-serif" font-size="{fs}" letter-spacing="1">STOP</text>'
                f'</svg>')

    elif class_name == "No Entry":
        # Red filled circle + wide white horizontal bar
        r   = int(w * 0.44)
        bh  = int(h * 0.19)
        bw  = int(w * 0.74)
        return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">'
                f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#E63946" stroke="#F1F1F1" stroke-width="2.5"/>'
                f'<rect x="{cx - bw//2}" y="{cy - bh//2}" width="{bw}" height="{bh}" '
                f'fill="#F1F1F1" rx="3"/>'
                f'</svg>')

    elif class_name == "Turn Left":
        # White circle, red border, solid black left-pointing arrow
        r = int(w * 0.44)
        # Arrow polygon points (pointing left)
        ax0 = int(cx * 0.30); ax1 = int(cx * 0.88)
        ay0 = int(cy * 0.82); ay1 = int(cy * 1.18)
        amid = int(cy)
        head = int(cx * 0.45)
        shaft_y0 = int(cy * 0.88); shaft_y1 = int(cy * 1.12)
        pts = (f"{ax0},{amid} {ax0+head},{ay0} {ax0+head},{shaft_y0} "
               f"{ax1},{shaft_y0} {ax1},{shaft_y1} {ax0+head},{shaft_y1} {ax0+head},{ay1}")
        return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">'
                f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#F1F1F1" stroke="#E63946" stroke-width="4.5"/>'
                f'<polygon points="{pts}" fill="#1D1D22"/>'
                f'</svg>')

    elif class_name == "Turn Right":
        # Mirror image of Turn Left
        r = int(w * 0.44)
        ax0 = int(cx * 1.70); ax1 = int(cx * 1.12)
        ay0 = int(cy * 0.82); ay1 = int(cy * 1.18)
        amid = int(cy)
        head = int(cx * 0.45)
        shaft_y0 = int(cy * 0.88); shaft_y1 = int(cy * 1.12)
        pts = (f"{ax0},{amid} {ax0-head},{ay0} {ax0-head},{shaft_y0} "
               f"{ax1},{shaft_y0} {ax1},{shaft_y1} {ax0-head},{shaft_y1} {ax0-head},{ay1}")
        return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">'
                f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#F1F1F1" stroke="#E63946" stroke-width="4.5"/>'
                f'<polygon points="{pts}" fill="#1D1D22"/>'
                f'</svg>')

    elif class_name == "Speed Limit 40":
        # White circle, red border ring, "40" in dark bold text
        r      = int(w * 0.44)
        border = max(4, int(w * 0.10))
        fs     = int(w * 0.30)
        return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">'
                f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#E63946"/>'
                f'<circle cx="{cx}" cy="{cy}" r="{r - border}" fill="#F1F1F1"/>'
                f'<text x="{cx}" y="{cy + fs*0.35}" text-anchor="middle" fill="#101013" '
                f'font-family="Bebas Neue,sans-serif" font-size="{fs}" font-weight="bold">40</text>'
                f'</svg>')

    return ""


# ═══════════════════════════════════════════════════════════════════════════════
# SVG GAUGE — speedometer-style circular confidence display
# ═══════════════════════════════════════════════════════════════════════════════
def gauge_svg(confidence: float, color: str) -> str:
    """
    Generates an animated circular SVG gauge (like a speedometer).

    HOW THE MATH WORKS:
      Circle radius r=70, so full circumference = 2π×70 ≈ 440.
      We only show 270° of the circle (75% of 440 = 330 units).
      stroke-dasharray="330 440" draws the 270° arc.
      transform="rotate(135 90 90)" rotates so the 90° gap sits at the bottom.

      CSS animates stroke-dashoffset from 330 (empty) down to --final-offset.
      --final-offset = 330 × (1 - confidence), set via inline style.
      This makes the arc fill up to the correct confidence percentage.
    """
    r    = 70
    cx   = cy = 90
    circ = 440          # 2π × 70
    arc  = 330          # 270° portion
    final_offset = arc * (1.0 - confidence)
    pct  = int(confidence * 100)

    return f"""
<div class="gauge-wrap">
  <svg width="180" height="155" viewBox="0 0 180 155"
       xmlns="http://www.w3.org/2000/svg">

    <!-- Background arc: grey, full 270 degrees -->
    <circle cx="{cx}" cy="{cy}" r="{r}"
      fill="none" stroke="#2A2A32" stroke-width="13"
      stroke-dasharray="{arc} {circ}" stroke-linecap="round"
      transform="rotate(135 {cx} {cy})"/>

    <!-- Colored fill arc: animates from 0% to confidence% via CSS -->
    <circle cx="{cx}" cy="{cy}" r="{r}"
      fill="none" stroke="{color}" stroke-width="13"
      stroke-dasharray="{arc} {circ}" stroke-linecap="round"
      transform="rotate(135 {cx} {cy})"
      class="gauge-fill"
      style="--final-offset: {final_offset:.2f};"/>

    <!-- Percentage number in the centre -->
    <text x="{cx}" y="{cy - 5}" text-anchor="middle"
      font-family="JetBrains Mono, monospace"
      font-size="27" font-weight="700" fill="#F1F1F1">{pct}%</text>

    <!-- Small label below number -->
    <text x="{cx}" y="{cy + 14}" text-anchor="middle"
      font-family="JetBrains Mono, monospace"
      font-size="8" fill="#8B8B95" letter-spacing="2.5">CONFIDENCE</text>

  </svg>
</div>"""


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE — tracks which stage the UI is in
# ═══════════════════════════════════════════════════════════════════════════════
# uploader_key  : incrementing this resets the file uploader widget
# prediction    : None until Analyze is clicked; then stores result dict
# analyzing     : True while spinner is running (shows scan line)
for key, default in [("uploader_key", 0), ("prediction", None), ("analyzing", False)]:
    if key not in st.session_state:
        st.session_state[key] = default


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD MODEL — cached once at startup
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading CNN model...")
def get_model_and_labels():
    try:
        m, lbl = load_model_and_labels()
        return m, lbl, None
    except FileNotFoundError as e:
        return None, None, str(e)

model, labels, load_error = get_model_and_labels()


# ═══════════════════════════════════════════════════════════════════════════════
# STATUS BAR
# ═══════════════════════════════════════════════════════════════════════════════
n_classes     = len(labels) if labels else 0
model_ok      = model is not None
status_color  = "#52B788" if model_ok else "#E63946"
status_text   = "READY"   if model_ok else "NOT LOADED"
classes_info  = f"{n_classes} CLASSES LOADED" if model_ok else "RUN python train.py"

st.markdown(f"""
<div class="status-bar">
  <div class="status-ring"><div class="status-dot"></div></div>
  <span>MODEL STATUS:&nbsp;<strong style="color:{status_color};">{status_text}</strong></span>
  <span class="sep">•</span>
  <span>{classes_info}</span>
  <span class="sep">•</span>
  <span style="color:#8B8B95;">CNN &mdash; TENSORFLOW / KERAS</span>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TITLE BLOCK
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="margin-bottom:.2rem;">
  <span style="font-family:'Bebas Neue',sans-serif;font-size:3.8rem;
               letter-spacing:6px;color:#F1F1F1;line-height:1;">
    TRAFFIC SIGN
  </span>
</div>
<div style="margin-bottom:.7rem;">
  <span style="font-family:'Bebas Neue',sans-serif;font-size:3.8rem;
               letter-spacing:6px;color:#FFC72C;line-height:1;">
    RECOGNITION
  </span>
</div>
<p style="color:#8B8B95;font-size:.92rem;max-width:480px;margin:0;line-height:1.65;">
  Upload an image to detect and classify Indian traffic signs
  using a trained Convolutional Neural Network.
</p>
""", unsafe_allow_html=True)

# Animated road divider
st.markdown('<div class="road-divider"></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL MISSING — stop here with helpful error
# ═══════════════════════════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD — two equal columns
# ═══════════════════════════════════════════════════════════════════════════════
col_left, col_right = st.columns([1, 1], gap="large")


# ────────────────────────────────────────────────────────────────────────────
# LEFT COLUMN — Upload / Viewfinder
# ────────────────────────────────────────────────────────────────────────────
with col_left:
    st.markdown('<p class="micro-label">CAMERA INPUT</p>', unsafe_allow_html=True)

    # File uploader widget — label hidden; viewfinder below provides the visual
    uploaded_file = st.file_uploader(
        "Select traffic sign image",
        type=["jpg", "jpeg", "png"],
        key=f"uploader_{st.session_state.uploader_key}",
        label_visibility="visible",
    )

    has_file   = uploaded_file is not None
    has_result = st.session_state.prediction is not None
    scanning   = st.session_state.analyzing

    # ── Viewfinder HTML ────────────────────────────────────────────────────
    corners = '<div class="c-tl corner"></div><div class="c-tr corner"></div><div class="c-bl corner"></div><div class="c-br corner"></div>'

    if has_file:
        # Encode image to base64 so we can embed it without writing to disk
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
        # Camera placeholder icon
        st.markdown(f"""
<div class="viewfinder">
  {corners}
  <div class="vf-placeholder">
    <svg width="52" height="52" viewBox="0 0 52 52" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect x="4" y="11" width="44" height="32" rx="5"
            stroke="#33333C" stroke-width="2"/>
      <circle cx="26" cy="27" r="9" stroke="#33333C" stroke-width="2"/>
      <circle cx="26" cy="27" r="3.5" fill="#33333C"/>
      <rect x="17" y="6" width="18" height="6" rx="3"
            stroke="#33333C" stroke-width="2"/>
    </svg>
    <p>DRAG &amp; DROP OR BROWSE<br>
       <span style="color:#2A2A35;">JPG &nbsp;/&nbsp; JPEG &nbsp;/&nbsp; PNG</span>
    </p>
  </div>
</div>""", unsafe_allow_html=True)

    # ── Pipeline status line ───────────────────────────────────────────────
    if not has_file:
        pipe_cls, pipe_txt = "pipeline",        "AWAITING IMAGE"
    elif scanning:
        pipe_cls, pipe_txt = "pipeline active", "RUNNING CNN INFERENCE..."
    elif has_result:
        pipe_cls, pipe_txt = "pipeline done",   "PREDICTION COMPLETE"
    else:
        pipe_cls, pipe_txt = "pipeline active", "IMAGE LOADED — READY TO ANALYZE"

    st.markdown(
        f'<div class="{pipe_cls}"><div class="pdot"></div>{pipe_txt}</div>',
        unsafe_allow_html=True)

    # ── Action buttons ─────────────────────────────────────────────────────
    if has_file and not has_result:
        analyze = st.button("ANALYZE SIGN", use_container_width=True, key="btn_analyze", type="primary")

        if analyze:
            st.session_state.analyzing = True
            with st.spinner("Running CNN inference..."):
                try:
                    # Save uploaded bytes to a temp file (cv2.imread needs a path)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name

                    # ── PREDICTION (using predict.py functions, unchanged) ──
                    # preprocess_image: resize to 64×64, normalise 0–1, add batch dim
                    processed = preprocess_image(tmp_path)

                    # model.predict: returns probability array for all classes
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


# ────────────────────────────────────────────────────────────────────────────
# RIGHT COLUMN — Results Panel
# ────────────────────────────────────────────────────────────────────────────
with col_right:
    st.markdown('<p class="micro-label">DETECTION RESULT</p>', unsafe_allow_html=True)

    pred = st.session_state.prediction

    if pred is None:
        # Idle placeholder — shown before any analysis
        st.markdown("""
<div class="surface-card" style="min-height:360px;display:flex;flex-direction:column;
     align-items:center;justify-content:center;text-align:center;gap:1rem;">
  <svg width="60" height="60" viewBox="0 0 60 60" fill="none" xmlns="http://www.w3.org/2000/svg">
    <circle cx="30" cy="30" r="26" stroke="#2A2A32" stroke-width="2"/>
    <path d="M20 30h20M30 20v20" stroke="#2A2A32" stroke-width="2" stroke-linecap="round"/>
  </svg>
  <p style="font-family:'Bebas Neue';font-size:1.5rem;letter-spacing:3px;
            color:#2A2A35;margin:0;">NO PREDICTION YET</p>
  <p style="color:#8B8B95;font-size:.82rem;margin:0;">
    Upload an image and click <strong style="color:#FFC72C;">ANALYZE SIGN</strong>
  </p>
</div>
""", unsafe_allow_html=True)

    elif "error" in pred:
        st.markdown(f"""
<div class="surface-card" style="border-color:#E6394644;">
  <p style="font-family:'Bebas Neue';font-size:1.6rem;color:#E63946;
            letter-spacing:2px;margin-bottom:.5rem;">INFERENCE ERROR</p>
  <p style="color:#8B8B95;font-family:'JetBrains Mono';font-size:.78rem;
            word-break:break-all;">{pred['error']}</p>
  <p style="color:#8B8B95;font-size:.84rem;margin-top:.8rem;">
    Please upload a valid JPG or PNG image and try again.
  </p>
</div>
""", unsafe_allow_html=True)

    else:
        # ── Successful prediction ──────────────────────────────────────────
        class_name = pred["class_name"]
        confidence = pred["confidence"]
        all_probs  = pred["all_probs"]
        lbl_map    = pred["labels"]
        color      = SIGN_COLOR.get(class_name, "#FFC72C")
        high_conf  = confidence >= 0.70

        icon_html    = sign_icon_svg(class_name, size=72)
        verdict_col  = "#52B788" if high_conf else "#FFC72C"
        verdict_txt  = "HIGH CONFIDENCE" if high_conf else "LOW CONFIDENCE"
        gauge_html   = gauge_svg(confidence, color)

        result_html = f"""
<div class="result-panel">
  <div style="display:flex;gap:1rem;align-items:center;">
    <div style="flex:1;">
      {gauge_html}
    </div>
    <div style="flex:1;">
      <div style="display:flex;flex-direction:column;justify-content:center;
                  height:155px;gap:.6rem;padding-left:.5rem;">
        {icon_html}
        <p class="result-name" style="color:{color};">{class_name.upper()}</p>
        <p style="font-family:'JetBrains Mono';font-size:.68rem;letter-spacing:1.5px;
                  color:{verdict_col};margin:0;">{verdict_txt}</p>
      </div>
    </div>
  </div>
  <hr style="border:none;border-top:1px solid #33333C;margin:1rem 0;">
  <p class="micro-label">ALL CLASS SCORES</p>
"""

        sorted_cls = sorted(lbl_map.items(),
                            key=lambda x: all_probs[int(x[0])],
                            reverse=True)

        for rank, (idx_str, cname) in enumerate(sorted_cls):
            prob      = float(all_probs[int(idx_str)])
            bar_color = SIGN_COLOR.get(cname, "#FFC72C")
            is_top    = (cname == class_name)
            head_cls  = "bar-head hi" if is_top else "bar-head"
            emo       = SIGN_EMOJI.get(cname, "🚦")
            pct_str   = f"{prob * 100:.1f}"

            result_html += f"""
<div class="bar-row">
  <div class="{head_cls}">
    <span>{emo}&nbsp; {cname}</span>
    <span>{pct_str}%</span>
  </div>
  <div class="bar-bg">
    <div class="bar-fill"
         style="width:{pct_str}%;
                background:linear-gradient(90deg,{bar_color}55,{bar_color});
                animation-delay:{rank * 0.07:.2f}s;">
    </div>
  </div>
</div>
"""
        result_html += "</div>"
        
        st.markdown(result_html, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;margin-top:2.5rem;padding-top:1.5rem;
            border-top:1px solid #33333C;">
  <p style="font-family:'JetBrains Mono';font-size:.7rem;color:#33333C;
            letter-spacing:2px;margin:0;">
    TRAFFIC SIGN RECOGNITION &nbsp;|&nbsp; CNN + TENSORFLOW &nbsp;|&nbsp;
    INDIAN IRC STANDARD &nbsp;|&nbsp; COLLEGE MINI PROJECT
  </p>
</div>
""", unsafe_allow_html=True)
