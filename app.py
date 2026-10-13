"""
PocketScope AI — Low-Cost Pocket Microscope Assistant
Team BIONEX · BIOBYTE 2K26
"""

import html as _html
import io, json, math, os
import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps, ImageDraw

# ── Optional ultralytics (not in requirements.txt) ───────────────────────
try:
    from ultralytics import YOLO
    _HAS_ULTRA = True
except ImportError:
    _HAS_ULTRA = False

_MODEL_PATH = os.path.join("models", "bloodcell.pt")
_METRICS_PATH = os.path.join("models", "metrics.json")
HAS_MODEL = _HAS_ULTRA and os.path.exists(_MODEL_PATH)
MAX_DIM = 1280

# =====================================================================
# DESIGN TOKENS
# =====================================================================
TEMPLATES = {
    "Midnight Brass": {
        "bg": "#070D1A", "surface": "#0F1B2E", "border": "#22364F",
        "accent": "#D4AF6A", "highlight": "#3CC9B2",
        "text": "#F2EDE0", "muted": "#B9B4A6",
    },
    "BioByte Green": {
        "bg": "#07110A", "surface": "#0E2013", "border": "#25472B",
        "accent": "#8FD14F", "highlight": "#8FD14F",
        "text": "#E8F3E6", "muted": "#B5C9B3",
    },
    "Navy Teal": {
        "bg": "#0B1220", "surface": "#15243A", "border": "#29415A",
        "accent": "#55D6BE", "highlight": "#55D6BE",
        "text": "#E5EDF7", "muted": "#B8C7D9",
    },
    "Graphite Cyan": {
        "bg": "#0E1116", "surface": "#171C24", "border": "#2A323D",
        "accent": "#4FC3F7", "highlight": "#4FC3F7",
        "text": "#E8ECF1", "muted": "#A5B0BD",
    },
    "Midnight Indigo": {
        "bg": "#0C0F1E", "surface": "#151A33", "border": "#2B3358",
        "accent": "#8AA2FF", "highlight": "#8AA2FF",
        "text": "#E9ECFA", "muted": "#A4ACD1",
    },
    "Clean Light": {
        "bg": "#F8F9FA", "surface": "#FFFFFF", "border": "#DDE1E5",
        "accent": "#0066CC", "highlight": "#0052A3",
        "text": "#212529", "muted": "#6C757D",
    },
}

FONT_SETS = {
    "Set A · Copperplate": {
        "heading": '"Copperplate Gothic Bold","Copperplate",\'Cinzel\',Georgia,serif',
        "sub":     '"Copperplate Gothic Light","Copperplate",\'Cinzel\',Georgia,serif',
        "body":    '"Footlight MT Light",\'EB Garamond\',Georgia,serif',
    },
    "Set B · Antiqua": {
        "heading": '"Cooper Black",\'Caprasimo\',Georgia,serif',
        "sub":     '"Book Antiqua","Palatino Linotype",Palatino,Georgia,serif',
        "body":    '"Baskerville Old Face",\'Libre Baskerville\',Georgia,serif',
    },
}

# =====================================================================
# ORGANISM REFERENCE LIBRARY (textbook-accurate, conservative)
# =====================================================================
ORGANISMS = {
    "Escherichia coli": {
        "gram": "Gram-negative", "shape": "Rod (bacillus), ~1–3 μm",
        "arrangement": "Single rods or pairs; motile (peritrichous flagella)",
        "colony": "Smooth, round, grey-white, 2–3 mm on nutrient agar; may be mucoid. Pink on MacConkey agar (lactose fermenter).",
        "notes": "Facultative anaerobe. Common gut commensal; certain pathotypes cause disease.",
        "diseases": ["Urinary tract infection", "Gastroenteritis (pathogenic strains)", "Neonatal meningitis", "Septicaemia"],
    },
    "Staphylococcus aureus": {
        "gram": "Gram-positive", "shape": "Coccus, ~0.8–1.0 μm",
        "arrangement": "Grape-like clusters",
        "colony": "Round, golden-yellow, 1–2 mm, opaque, sometimes haemolytic on blood agar.",
        "notes": "Catalase-positive, coagulase-positive. Can produce enterotoxins.",
        "diseases": ["Skin and soft-tissue infections", "Endocarditis", "Osteomyelitis", "Food poisoning", "Toxic shock syndrome"],
    },
    "Streptococcus pyogenes": {
        "gram": "Gram-positive", "shape": "Coccus, ~0.6–1.0 μm",
        "arrangement": "Chains or pairs",
        "colony": "Small, translucent, 1–2 mm; beta-haemolytic on blood agar.",
        "notes": "Lancefield Group A. Catalase-negative.",
        "diseases": ["Pharyngitis", "Scarlet fever", "Impetigo", "Necrotising fasciitis", "Rheumatic fever (post-infection sequela)"],
    },
    "Bacillus subtilis": {
        "gram": "Gram-positive", "shape": "Rod (bacillus), ~2–4 μm, endospore-forming",
        "arrangement": "Single rods or chains",
        "colony": "Large, flat, irregular spreading margins; dry, rough texture.",
        "notes": "Aerobic or facultative. Generally non-pathogenic; widely used in research and industry.",
        "diseases": ["Generally not considered a human pathogen"],
    },
    "Pseudomonas aeruginosa": {
        "gram": "Gram-negative", "shape": "Rod (bacillus), ~1–3 μm",
        "arrangement": "Single rods; motile (polar flagellum)",
        "colony": "Flat, spreading, often blue-green pigment (pyocyanin); grape-like odour.",
        "notes": "Obligate aerobe. Opportunistic pathogen; intrinsically resistant to many antibiotics.",
        "diseases": ["Wound and burn infections", "Pneumonia (especially in cystic fibrosis)", "Urinary tract infection", "Septicaemia"],
    },
    "Klebsiella pneumoniae": {
        "gram": "Gram-negative", "shape": "Rod (bacillus), ~1–2 μm, capsulated",
        "arrangement": "Single rods or pairs; non-motile",
        "colony": "Large, mucoid, dome-shaped; often very sticky when touched with a loop.",
        "notes": "Facultative anaerobe. Lactose fermenter. The mucoid capsule is a virulence factor.",
        "diseases": ["Pneumonia", "Urinary tract infection", "Bloodstream infections", "Liver abscess"],
    },
    "Candida albicans": {
        "gram": "N/A (fungus — use Gram stain: Gram-positive appearance)",
        "shape": "Yeast (oval budding cells, ~3–6 μm); can form pseudohyphae and true hyphae",
        "arrangement": "Budding yeast cells, pseudohyphal chains",
        "colony": "Cream-white, smooth, pasty on Sabouraud agar; sometimes rough with age.",
        "notes": "Dimorphic fungus. Commensal of skin, mouth and GI tract.",
        "diseases": ["Oral thrush", "Vulvovaginal candidiasis", "Cutaneous candidiasis", "Systemic candidiasis (in immunocompromised)"],
    },
    "Saccharomyces cerevisiae": {
        "gram": "N/A (fungus)",
        "shape": "Yeast (oval to round, ~5–10 μm)",
        "arrangement": "Budding yeast cells, sometimes short chains",
        "colony": "Cream-white, smooth, convex, ~2–3 mm on YPD agar.",
        "notes": "Baker’s / brewer’s yeast. Generally non-pathogenic. Model organism in genetics.",
        "diseases": ["Generally not considered a human pathogen; very rare opportunistic infections reported"],
    },
}

# =====================================================================
# PAGE CONFIG
# =====================================================================
st.set_page_config(page_title="PocketScope AI | BIONEX", page_icon="🔬",
                   layout="wide", initial_sidebar_state="collapsed")

# ── Session-state defaults ───────────────────────────────────────────
for _k, _v in {"page": "start", "template_name": "Midnight Brass",
               "font_set": "Set A · Copperplate",
               "sample_type": "Culture plate (colonies)",
               "upload_key": 0, "camera_key": 0}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Read current tokens ──────────────────────────────────────────────
tok = TEMPLATES.get(st.session_state.template_name, TEMPLATES["Midnight Brass"])
fnt = FONT_SETS.get(st.session_state.font_set, FONT_SETS["Set A · Copperplate"])

# =====================================================================
# CSS INJECTION (scoped, no broad !important colour, no * font)
# =====================================================================
st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700&family=Caprasimo&family=EB+Garamond:wght@400;500&family=Libre+Baskerville:wght@400;700&display=swap');
:root{{
  --bg:{tok['bg']};--sf:{tok['surface']};--bd:{tok['border']};
  --ac:{tok['accent']};--hl:{tok['highlight']};--tx:{tok['text']};--mt:{tok['muted']};
  --fh:{fnt['heading']};--fs:{fnt['sub']};--fb:{fnt['body']};
}}
.stApp{{background:var(--bg)!important;color:var(--tx);font-family:var(--fb);font-size:18px;line-height:1.6}}
h1{{font-family:var(--fh)!important;color:var(--tx);text-transform:uppercase;letter-spacing:.06em;font-size:calc(2rem + 3px)!important}}
h2{{font-family:var(--fh)!important;color:var(--tx);text-transform:uppercase;letter-spacing:.06em;font-size:calc(1.5rem + 3px)!important}}
h3{{font-family:var(--fh)!important;color:var(--tx);text-transform:uppercase;letter-spacing:.06em;font-size:calc(1.2rem + 3px)!important}}
h4,h5,h6{{font-family:var(--fh)!important;color:var(--tx);text-transform:uppercase;letter-spacing:.06em;font-size:calc(1rem + 2px)!important}}

/* ── Cards ──────────────────────────────────────────────────── */
@keyframes fadeUp{{from{{opacity:0;transform:translateY(10px)}}to{{opacity:1;transform:translateY(0)}}}}
@media(prefers-reduced-motion:reduce){{.ps-c,.ps-ring-pulse,.ps-hero-lens::after{{animation:none!important}}}}
.ps-c{{background:var(--sf);border:1px solid var(--bd);border-radius:14px;padding:18px 20px;margin-bottom:14px;
  animation:fadeUp .4s ease both;transition:box-shadow .18s,border-color .18s,transform .18s}}
.ps-c:hover{{box-shadow:0 4px 20px rgba(0,0,0,.35);border-color:var(--ac);transform:translateY(-2px)}}
.ps-c.accent-top{{border-top:2px solid var(--ac)}}
.ps-ct{{font-family:var(--fs);font-size:calc(1rem + 2px);font-weight:700;color:var(--tx);letter-spacing:.04em;margin-bottom:6px}}
.ps-cm{{font-size:calc(.82rem + 2px);color:var(--mt);margin-bottom:6px}}
.ps-pill{{display:inline-block;padding:2px 10px;border-radius:20px;font-size:calc(.78rem + 2px);font-family:var(--fs);
  border:1px solid var(--bd);color:var(--mt);margin:2px 4px 2px 0}}
.ps-pill.active{{background:var(--ac);color:var(--bg);border-color:var(--ac)}}
.ps-swatch{{display:inline-block;width:18px;height:18px;border-radius:4px;vertical-align:middle;margin-right:8px;border:1px solid var(--bd)}}
.ps-mono{{font-family:'JetBrains Mono',Consolas,monospace;font-size:calc(.85rem + 2px)}}
.ps-divider{{text-align:center;margin:20px 0;position:relative}}
.ps-divider::before{{content:'';position:absolute;left:0;right:0;top:50%;height:1px;background:var(--bd)}}
.ps-divider span{{position:relative;background:var(--bg);padding:0 12px;color:var(--ac);font-family:var(--fs);font-size:calc(.82rem + 2px);letter-spacing:.08em}}
.ps-hero-num{{font-family:var(--fh);font-size:clamp(2.4rem,8.2vw,3.7rem);color:var(--ac);line-height:1;letter-spacing:.02em}}

/* ── Bars ───────────────────────────────────────────────────── */
.ps-bw{{margin-bottom:6px}}.ps-bl{{font-size:calc(.82rem + 2px);color:var(--mt);margin-bottom:2px}}
.ps-bt{{height:7px;background:var(--bd);border-radius:4px;overflow:hidden}}
.ps-bf{{height:7px;background:var(--ac);border-radius:4px;transition:width .5s ease}}

/* ── Streamlit overrides (scoped, not broad) ─────────────── */
div[data-testid="stMetric"]{{background:var(--sf);border:1px solid var(--bd);border-radius:12px;padding:12px 16px}}
div[data-testid="stMetricLabel"] p{{color:var(--mt)!important;font-family:var(--fs);font-size:calc(.78rem + 2px);text-transform:uppercase;letter-spacing:.05em}}
div[data-testid="stMetricValue"]{{color:var(--ac)!important;font-family:var(--fh);font-size:calc(1.5rem + 2px)}}
div[data-testid="stTabs"] button[role="tab"]{{color:var(--mt)!important;font-family:var(--fs);font-weight:600;font-size:calc(.92rem + 2px);
  min-height:44px;border-bottom:2px solid transparent;padding:8px 16px;letter-spacing:.04em;transition:color .15s,border-color .15s}}
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"]{{color:var(--ac)!important;border-bottom-color:var(--ac)!important}}
div.stButton>button,div.stDownloadButton>button{{background:var(--sf)!important;color:var(--tx)!important;
  border:1px solid var(--bd)!important;border-radius:10px!important;min-height:44px!important;
  font-family:var(--fs)!important;font-weight:600;letter-spacing:.03em;transition:border-color .15s,color .15s,transform .1s}}
div.stButton>button:hover,div.stDownloadButton>button:hover{{border-color:var(--ac)!important;color:var(--ac)!important}}
div.stButton>button:active{{transform:scale(.97)}}
div[data-testid="stFileUploader"],div[data-testid="stCameraInput"]{{background:var(--sf);border:1px dashed var(--bd);border-radius:12px;padding:10px}}
div[data-testid="stExpander"]{{background:var(--sf);border:1px solid var(--bd);border-radius:12px;margin-bottom:14px}}
section[data-testid="stSidebar"]{{background:var(--sf);border-right:1px solid var(--bd)}}
input,select,textarea{{font-size:18px!important}}

/* ── Demo banner ───────────────────────────────────────────── */
.ps-demo{{background:repeating-linear-gradient(-45deg,rgba(255,200,0,.06) 0 10px,transparent 10px 20px);
  border:1px dashed #f0a000;border-radius:10px;padding:8px 14px;margin-bottom:12px;font-size:calc(.84rem + 2px);color:#f0a000;font-weight:600}}

/* ── Hero (start screen) ───────────────────────────────────── */
.ps-hero{{text-align:center;padding:24px 16px 16px}}
.ps-hero-lens{{position:relative;width:170px;height:170px;margin:0 auto 20px;display:inline-block}}
.ps-hero-lens::after{{content:'';position:absolute;left:22%;right:22%;height:1.5px;background:var(--ac);
  top:50%;animation:psScan 4.5s ease-in-out infinite;opacity:.45}}
@keyframes psScan{{0%,100%{{top:28%;opacity:0}}15%{{opacity:.45}}85%{{opacity:.45}}100%{{top:72%;opacity:0}}}}
.ps-ring-pulse{{animation:psRing 3s ease-in-out infinite}}
@keyframes psRing{{0%,100%{{stroke-opacity:1}}50%{{stroke-opacity:.5}}}}
.ps-hero-eye{{font-family:var(--fs);font-size:13px;letter-spacing:.18em;text-transform:uppercase;color:var(--ac);margin-bottom:6px}}
.ps-hero-title{{font-family:var(--fh);font-size:clamp(28px,7.2vw,46px);letter-spacing:.08em;text-transform:uppercase;
  color:var(--tx);margin:0 0 6px;line-height:1.15;word-wrap:break-word}}
.ps-hero-tag{{font-family:var(--fs);font-size:clamp(16px,3.2vw,20px);color:var(--mt);margin-bottom:8px}}
.ps-hero-desc{{font-family:var(--fb);font-size:18px;color:var(--mt);line-height:1.6;max-width:480px;margin:0 auto 20px}}
.ps-hero-foot{{font-size:14px;color:var(--mt);margin-top:20px}}
@media(max-width:640px){{.ps-c{{padding:14px}}.ps-hero{{padding:16px 12px 12px}}}}
</style>""", unsafe_allow_html=True)


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================
def card(title, body, meta="", accent_top=False):
    """Render a themed card as one HTML string."""
    cls = "ps-c accent-top" if accent_top else "ps-c"
    m = f'<div class="ps-cm">{_html.escape(meta)}</div>' if meta else ""
    st.markdown(f'<div class="{cls}"><div class="ps-ct">{_html.escape(title)}</div>{m}{body}</div>',
                unsafe_allow_html=True)

def bars(data):
    """Render themed bar rows from {label: count}."""
    if not data:
        return ""
    total = max(sum(data.values()), 1)
    h = ""
    for label, count in data.items():
        pct = 100 * count / total
        h += (f'<div class="ps-bw"><div class="ps-bl">{_html.escape(label)} — {count}</div>'
              f'<div class="ps-bt"><div class="ps-bf" style="width:{pct:.1f}%"></div></div></div>')
    return h

def divider(numeral=""):
    """Ornamental hairline with optional centred text."""
    inner = f"<span>{_html.escape(numeral)}</span>" if numeral else "<span>◆</span>"
    st.markdown(f'<div class="ps-divider">{inner}</div>', unsafe_allow_html=True)

def load_image(fobj):
    """Load image: EXIF fix, resize to max 1280px."""
    pil = ImageOps.exif_transpose(Image.open(fobj)).convert("RGB")
    w, h = pil.size
    if max(w, h) > MAX_DIM:
        s = MAX_DIM / float(max(w, h))
        pil = pil.resize((int(w * s), int(h * s)), Image.Resampling.LANCZOS)
    return np.array(pil)

def quality(img, blur_thr):
    """Laplacian sharpness and mean brightness."""
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var()), float(g.mean())

def classify_color(r, g, b):
    """HSV-based readable colour name, hex, RGB tuple."""
    h_raw, s_raw, v_raw = cv2.cvtColor(np.uint8([[[r, g, b]]]), cv2.COLOR_RGB2HSV)[0][0]
    h, s, v = float(h_raw), float(s_raw), float(v_raw)
    hx = f"#{r:02x}{g:02x}{b:02x}"
    if v < 40:   return "Very dark / black", hx, (r, g, b)
    if s < 25:
        if v > 200: return "White / pale", hx, (r, g, b)
        return "Grey", hx, (r, g, b)
    if v < 80:   return "Very dark / black", hx, (r, g, b)
    # chromatic
    if h < 10 or h >= 165: return "Red / pink", hx, (r, g, b)
    if h < 22:  return "Orange", hx, (r, g, b)
    if h < 35:  return "Yellow / cream", hx, (r, g, b)
    if h < 45:  return "Yellow / cream" if s < 120 else "Brown", hx, (r, g, b)
    if h < 78:  return "Green", hx, (r, g, b)
    if h < 130: return "Blue", hx, (r, g, b)
    return "Purple / violet", hx, (r, g, b)

def region_color(img, contours):
    """Mean colour inside contours."""
    if not contours:
        return "Undetermined", "#808080", (128, 128, 128)
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(mask, contours, -1, 255, -1)
    px = img[mask == 255]
    if len(px) == 0:
        return "Undetermined", "#808080", (128, 128, 128)
    mr = int(np.clip(np.mean(px[:, 0]), 0, 255))
    mg = int(np.clip(np.mean(px[:, 1]), 0, 255))
    mb = int(np.clip(np.mean(px[:, 2]), 0, 255))
    return classify_color(mr, mg, mb)

def slide_shape(c):
    """Shape from minAreaRect. Area<25 = Undetermined."""
    area = cv2.contourArea(c)
    peri = cv2.arcLength(c, True)
    if area < 25 or peri <= 0:
        return "Undetermined"
    circ = (4 * math.pi * area) / (peri * peri)
    rect = cv2.minAreaRect(c)
    w, h = rect[1]
    asp = max(w, h) / max(1, min(w, h))
    if circ >= 0.78 and asp < 1.35:  return "Round / spherical"
    if asp >= 2.2:                    return "Rod-like"
    if circ >= 0.55 and asp < 2.2:   return "Oval"
    if circ < 0.45:                   return "Irregular"
    return "Undetermined"

def find_dish(img, enabled=True):
    """HoughCircles on half-size for speed."""
    if not enabled:
        return False, None
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    half = cv2.resize(gray, (w // 2, h // 2))
    hh, hw = half.shape
    blur = cv2.medianBlur(half, 7)
    md = min(hh, hw)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=md // 2,
                               param1=70, param2=45,
                               minRadius=int(md * 0.28), maxRadius=int(md * 0.52))
    if circles is not None:
        cx, cy, r = [int(v * 2) for v in circles[0][0]]
        if 0 <= cx < w and 0 <= cy < h and r > 30:
            return True, (cx, cy, r)
    return False, None

def size_bins(diameters, unit="px"):
    """5 fixed-width bins for colony/object diameter distribution."""
    if not diameters:
        return {}
    lo, hi = min(diameters), max(diameters)
    if hi - lo < 0.5:
        return {f"{lo:.1f} {unit}": len(diameters)}
    bw = (hi - lo) / 5
    out = {}
    for i in range(5):
        a, b = lo + i * bw, lo + (i + 1) * bw
        cnt = sum(1 for d in diameters if a <= d < b or (i == 4 and d <= b))
        out[f"{a:.1f}–{b:.1f} {unit}"] = cnt
    return out

def to_png(arr):
    """Convert numpy image to PNG bytes."""
    buf = io.BytesIO(); Image.fromarray(arr).save(buf, format="PNG"); return buf.getvalue()


# =====================================================================
# CALLBACKS (use on_click for page/mode switching)
# =====================================================================
def _go_plate():
    st.session_state.page = "app"
    st.session_state.sample_type = "Culture plate (colonies)"
def _go_slide():
    st.session_state.page = "app"
    st.session_state.sample_type = "Microscope slide (cells)"
def _go_start():
    st.session_state.page = "start"


# =====================================================================
# START SCREEN
# =====================================================================
if st.session_state.page == "start":
    # SVG hero lens
    svg = f'''<svg viewBox="0 0 200 200" width="160" height="160" xmlns="http://www.w3.org/2000/svg">
      <defs><radialGradient id="gl"><stop offset="0%" stop-color="var(--ac)" stop-opacity=".12"/>
        <stop offset="100%" stop-color="transparent" stop-opacity="0"/></radialGradient></defs>
      <circle cx="100" cy="100" r="96" fill="url(#gl)"/>
      <circle cx="100" cy="100" r="88" fill="none" stroke="var(--ac)" stroke-width="2.5" class="ps-ring-pulse"/>
      <circle cx="100" cy="100" r="74" fill="none" stroke="var(--ac)" stroke-width="1.2" opacity=".35"/>
      <circle cx="100" cy="100" r="62" fill="none" stroke="var(--ac)" stroke-width=".8" opacity=".2"/>
      <circle cx="100" cy="100" r="58" fill="var(--sf)" opacity=".85"/>
      <circle cx="80" cy="82" r="5" fill="var(--hl)" opacity=".7"/>
      <circle cx="108" cy="76" r="3.5" fill="var(--hl)" opacity=".6"/>
      <circle cx="92" cy="105" r="4" fill="var(--hl)" opacity=".65"/>
      <circle cx="120" cy="98" r="3" fill="var(--hl)" opacity=".55"/>
      <circle cx="85" cy="118" r="4.5" fill="var(--hl)" opacity=".6"/>
      <circle cx="115" cy="112" r="3.5" fill="var(--hl)" opacity=".5"/>
      <circle cx="97" cy="88" r="2.5" fill="var(--hl)" opacity=".45"/>
    </svg>'''
    st.markdown(f'''<div class="ps-hero">
      <div class="ps-hero-lens">{svg}</div>
      <div class="ps-hero-eye">BIONEX · SMART MICROSCOPY</div>
      <div class="ps-hero-title">POCKETSCOPE AI</div>
      <div class="ps-hero-tag">See beyond the visible.</div>
      <div class="ps-hero-desc">A pocket-sized microscopy system for culture plates and microscope slides.</div>
    </div>''', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.button("🧫  Inspect a culture plate", on_click=_go_plate, use_container_width=True)
    with c2:
        st.button("🔬  Inspect a microscope slide", on_click=_go_slide, use_container_width=True)
    st.stop()


# =====================================================================
# WORKSPACE
# =====================================================================
# ── Compact header + back button (header first) ──────────────────────
hdr_l, hdr_r = st.columns([4, 1])
with hdr_l:
    st.markdown('<div style="font-family:var(--fh);font-size:clamp(20px,4.2vw,26px);letter-spacing:.06em;'
                'text-transform:uppercase;color:var(--tx)">POCKETSCOPE AI</div>', unsafe_allow_html=True)
with hdr_r:
    st.button("← Start", on_click=_go_start, use_container_width=True)

# ── Controls (after header) ──────────────────────────────────────────
sample_opts = ["Culture plate (colonies)", "Microscope slide (cells)"]
if HAS_MODEL:
    sample_opts.append("Blood smear (trained model)")

ct1, ct2, ct3 = st.columns([2, 1, 1])
with ct1:
    st.radio("Sample type", sample_opts, key="sample_type", horizontal=True)
with ct2:
    st.selectbox("Template", list(TEMPLATES.keys()), key="template_name")
with ct3:
    st.selectbox("Font pairing", list(FONT_SETS.keys()), key="font_set")

is_plate = st.session_state.sample_type == "Culture plate (colonies)"
is_slide = st.session_state.sample_type == "Microscope slide (cells)"
is_model = st.session_state.sample_type == "Blood smear (trained model)"
item_word = "Colonies" if is_plate else "Objects"

# ── Detection settings ───────────────────────────────────────────────
with st.expander("Detection settings", expanded=False):
    s1, s2 = st.columns(2)
    blur_thr = 60; min_area = 30; max_area = 3500; show_nums = False
    slide_pol = "Darker than background (stained)"; thr_adj = 0
    dish_mm = 90.0; plate_pol = "Colonies lighter than agar"; morph_k = 3
    min_contrast = 8; auto_dish = True; split_touching = True
    model_conf = 0.25; model_iou = 0.45

    if is_slide:
        with s1:
            blur_thr = st.slider("Blur threshold", 10, 300, 60, key="sl_blur")
            slide_pol = st.radio("Object polarity", ["Darker than background (stained)",
                                 "Lighter than background (darkfield)"], key="sl_pol")
            thr_adj = st.slider("Threshold adjust", -60, 60, 0, key="sl_thr")
        with s2:
            min_area = st.slider("Min object area (px)", 5, 500, 30, key="sl_mina")
            max_area = st.slider("Max object area (px)", 200, 10000, 3500, key="sl_maxa")
            show_nums = st.checkbox("Show index numbers", key="sl_nums")
    elif is_plate:
        with s1:
            dish_mm = st.number_input("Dish diameter (mm)", 20., 200., 90., 5., key="pl_mm")
            plate_pol = st.radio("Colony appearance", ["Colonies lighter than agar",
                                 "Colonies darker than agar"], key="pl_pol")
            auto_dish = st.checkbox("Auto-detect dish boundary", value=True, key="pl_auto")
            split_touching = st.checkbox("Split touching colonies", value=True, key="pl_split")
        with s2:
            min_area = st.slider("Min colony area (px)", 5, 800, 25, key="pl_mina")
            max_area = st.slider("Max colony area (px)", 500, 50000, 12000, key="pl_maxa")
            morph_k = st.slider("Morphology kernel", 1, 7, 3, step=2, key="pl_morph")
            min_contrast = st.slider("Min contrast floor", 2, 40, 8, key="pl_floor")
            show_nums = st.checkbox("Show colony numbers", key="pl_nums")
    elif is_model:
        with s1:
            model_conf = st.slider("Confidence", 0.05, 0.95, 0.25, 0.05, key="m_conf")
            model_iou = st.slider("IoU threshold", 0.1, 0.9, 0.45, 0.05, key="m_iou")
        with s2:
            blur_thr = st.slider("Blur threshold", 10, 300, 50, key="m_blur")
            show_nums = st.checkbox("Show labels on boxes", True, key="m_nums")

# ── Image input ───────────────────────────────────────────────────────
ic, rc = st.columns([3, 1])
with ic:
    st.markdown("### Acquire sample image")
with rc:
    if st.button("Reset image", use_container_width=True):
        st.session_state.upload_key += 1; st.session_state.camera_key += 1; st.rerun()

src = st.radio("Input source", ["Upload image", "Use camera"], horizontal=True)
loaded = (st.file_uploader("Upload photograph", type=["jpg","jpeg","png"],
            key=f"up_{st.session_state.upload_key}")
          if src == "Upload image" else
          st.camera_input("Capture", key=f"cam_{st.session_state.camera_key}"))

# ── Empty state ───────────────────────────────────────────────────────
if loaded is None:
    desc = ("Upload or capture a top-down photo of your Petri dish." if is_plate
            else "Upload a stained smear or capture through the PocketScope lens." if is_slide
            else "Upload a blood smear photograph." if is_model else "")
    st.markdown(f'''<div style="text-align:center;padding:48px 24px;color:var(--mt)">
      <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none"
           stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round" style="opacity:.45;margin-bottom:16px">
        <path d="M6 18h12"/><path d="M12 18v-4"/><rect x="9" y="2" width="6" height="10" rx="1"/>
        <circle cx="12" cy="18" r="2"/></svg>
      <div style="font-family:var(--fh);font-size:calc(1.1rem + 2px);letter-spacing:.05em;margin-bottom:6px">
        Ready for inspection</div>
      <div style="font-size:calc(.9rem + 2px);max-width:420px;margin:0 auto">{desc}</div></div>''', unsafe_allow_html=True)
    st.stop()

# ── Load & check image ───────────────────────────────────────────────
try:
    img = load_image(loaded)
except Exception:
    st.error("Could not decode image. Supply a valid JPG or PNG."); st.stop()

img_h, img_w = img.shape[:2]
sharp, bright = quality(img, blur_thr)
q_warns = []
if sharp < blur_thr:
    q_warns.append(f"Image may be blurry (sharpness {sharp:.0f} vs threshold {blur_thr}). Refocus the lens.")
if bright < 60:
    q_warns.append(f"Underexposed (brightness {bright:.0f}/255). Increase LED.")
elif bright > 220:
    q_warns.append(f"Overexposed (brightness {bright:.0f}/255). Reduce LED.")

# =====================================================================
# DETECTION
# =====================================================================
contours = []; areas = []; diams = []; shapes_or_forms = {}
objects = []; edge_counts = {}; color_name = "Undetermined"; color_hex = "#808080"
color_rgb = (128,128,128); agar_color = None
dish_found = False; dish_info = None; scale_mm = None
detect_count = 0; warns = []; model_classes = {}
annotated = img.copy()

with st.status("Examining sample…", expanded=True) as status:

    # ── Blood smear (trained model) ──────────────────────────────────
    if is_model and HAS_MODEL:
        st.write("Loading model…")
        model = YOLO(_MODEL_PATH)
        st.write("Detecting…")
        res = model.predict(img, conf=model_conf, iou=model_iou, verbose=False)[0]
        palette = [(255,100,100),(100,220,100),(100,160,255),(255,200,50),(220,100,220)]
        for b in res.boxes:
            cid = int(b.cls[0]); cn = str(res.names.get(cid, f"Class {cid}"))
            model_classes[cn] = model_classes.get(cn, 0) + 1
            x1,y1,x2,y2 = b.xyxy[0].cpu().numpy().astype(int)
            areas.append(float((x2-x1)*(y2-y1)))
            diams.append(math.sqrt((x2-x1)*(y2-y1)))
            cc = palette[cid % len(palette)]
            cv2.rectangle(annotated, (x1,y1),(x2,y2), cc, 2)
            if show_nums:
                cv2.putText(annotated, f"{cn} {float(b.conf[0]):.2f}", (x1,max(y1-6,14)),
                            cv2.FONT_HERSHEY_SIMPLEX, .45, cc, 1, cv2.LINE_AA)
        detect_count = len(res.boxes); shapes_or_forms = model_classes
        item_word = "Objects"

    # ── Microscope slide ─────────────────────────────────────────────
    elif is_slide:
        st.write("Detecting objects…")
        green = img[:,:,1]
        blurred = cv2.GaussianBlur(green, (5,5), 0)
        inv = cv2.THRESH_BINARY_INV if "Darker" in slide_pol else cv2.THRESH_BINARY
        otsu_val, _ = cv2.threshold(blurred, 0, 255, inv + cv2.THRESH_OTSU)
        eff = float(np.clip(otsu_val + thr_adj, 0, 255))
        _, binary = cv2.threshold(blurred, eff, 255, inv)
        k = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
        raw, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sc = {}
        for c in raw:
            a = cv2.contourArea(c)
            if min_area <= a <= max_area:
                contours.append(c); areas.append(float(a))
                d = 2*math.sqrt(a/math.pi); diams.append(d)
                s = slide_shape(c); sc[s] = sc.get(s,0)+1
                (cx,cy),rad = cv2.minEnclosingCircle(c)
                m = np.zeros(img.shape[:2],np.uint8); cv2.drawContours(m,[c],-1,255,-1)
                px = img[m==255]
                oname,ohex,_ = classify_color(int(np.mean(px[:,0])),int(np.mean(px[:,1])),int(np.mean(px[:,2]))) if len(px) else ("Undetermined","#808080",(0,0,0))
                objects.append({"id":len(contours),"cx":int(cx),"cy":int(cy),"radius":max(2,int(rad)),
                    "area":a,"diameter_px":d,"shape":s,"color":oname,"hex":ohex})
                cv2.circle(annotated,(int(cx),int(cy)),max(2,int(rad)),(0,235,215),2)
                if show_nums and len(contours)<=150:
                    cv2.putText(annotated,str(len(contours)),(int(cx)-6,int(cy)+4),cv2.FONT_HERSHEY_SIMPLEX,.38,(255,255,255),1,cv2.LINE_AA)
        shapes_or_forms = sc; detect_count = len(contours)
        color_name,color_hex,color_rgb = region_color(img, contours)

    # ── Culture plate ────────────────────────────────────────────────
    elif is_plate:
        st.write("Locating dish…")
        dish_found, dish_info = find_dish(img, auto_dish)
        analysis_mask = np.zeros((img_h,img_w), np.uint8)
        analysis_r = 0
        if dish_found:
            dcx,dcy,dr = dish_info
            analysis_r = int(dr * 0.92)
            cv2.circle(analysis_mask,(dcx,dcy),analysis_r,255,-1)
            cv2.circle(annotated,(dcx,dcy),dr,(70,200,255),2)
        else:
            analysis_mask[:] = 255

        st.write("Segmenting colonies…")
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Illumination correction
        sigma = 0.03 * max(img_h, img_w)
        ksz = int(sigma * 6) | 1
        if ksz < 3: ksz = 3
        blur_bg = cv2.GaussianBlur(gray, (ksz, ksz), sigma)
        lighter = "lighter" in plate_pol.lower()
        diff = (gray.astype(np.float32) - blur_bg.astype(np.float32)) if lighter else (blur_bg.astype(np.float32) - gray.astype(np.float32))
        diff = np.clip(diff, 0, 255).astype(np.uint8)

        dish_px = diff[analysis_mask == 255]
        if len(dish_px) > 100:
            ov, _ = cv2.threshold(dish_px, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            et = max(float(ov), float(min_contrast))
        else:
            et = float(min_contrast)
        _, binary = cv2.threshold(diff, et, 255, cv2.THRESH_BINARY)
        binary = cv2.bitwise_and(binary, binary, mask=analysis_mask)
        mk = np.ones((morph_k, morph_k), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, mk)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, mk)

        # Watershed split
        if split_touching and np.any(binary):
            dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
            pk = 0.4 * dist.max() if dist.max() > 0 else 0
            _, sfg = cv2.threshold(dist, pk, 255, cv2.THRESH_BINARY)
            sfg = sfg.astype(np.uint8)
            sbg = cv2.dilate(binary, np.ones((3,3),np.uint8), iterations=2)
            unk = cv2.subtract(sbg, sfg)
            _, markers = cv2.connectedComponents(sfg)
            markers += 1; markers[unk==255] = 0
            markers = cv2.watershed(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), markers)
            ws = np.zeros_like(binary); ws[markers > 1] = 255
            binary = ws

        raw, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fc = {}; ec = {}
        for c in raw:
            a = cv2.contourArea(c)
            if a < min_area or a > max_area:
                continue
            # Rim rejection
            if dish_found:
                rim = False
                for pt in c:
                    dd = math.sqrt((pt[0][0]-dcx)**2+(pt[0][1]-dcy)**2)
                    if dd > analysis_r * 0.96: rim=True; break
                if rim: continue
            contours.append(c); areas.append(float(a))
            d = 2*math.sqrt(a/math.pi); diams.append(d)
            peri = cv2.arcLength(c,True)
            circ = (4*math.pi*a)/(peri*peri) if peri>0 else 0
            if circ>=0.85: form="Circular"
            elif circ>=0.65: form="Oval"
            else:
                rect=cv2.minAreaRect(c); ww,hh=rect[1]
                form="Elongated" if max(ww,hh)/max(1,min(ww,hh))>=2 else "Irregular"
            fc[form]=fc.get(form,0)+1
            hull_a = cv2.contourArea(cv2.convexHull(c))
            sol = (a/hull_a) if hull_a>0 else 1
            edge = "Smooth" if sol>=0.90 else "Uneven"
            ec[edge]=ec.get(edge,0)+1
            (cx,cy),rad=cv2.minEnclosingCircle(c)
            m=np.zeros(img.shape[:2],np.uint8); cv2.drawContours(m,[c],-1,255,-1)
            px=img[m==255]
            oname,ohex,_=classify_color(int(np.mean(px[:,0])),int(np.mean(px[:,1])),int(np.mean(px[:,2]))) if len(px) else ("Undetermined","#808080",(0,0,0))
            objects.append({"id":len(contours),"cx":int(cx),"cy":int(cy),"radius":max(2,int(rad)),
                "area":a,"diameter_px":d,"form":form,"edge":edge,"color":oname,"hex":ohex})
            cv2.circle(annotated,(int(cx),int(cy)),max(2,int(rad)),(0,235,120),2)
            if show_nums and len(contours)<=150:
                cv2.putText(annotated,str(len(contours)),(int(cx)-6,int(cy)+4),cv2.FONT_HERSHEY_SIMPLEX,.38,(255,255,255),1,cv2.LINE_AA)
        shapes_or_forms=fc; edge_counts=ec; detect_count=len(contours)
        color_name,color_hex,color_rgb = region_color(img, contours)
        scale_mm = (dish_mm / (2*dish_info[2])) if dish_found and dish_info else None
        # Agar colour
        if dish_found:
            cmask=np.zeros(img.shape[:2],np.uint8)
            cv2.drawContours(cmask,contours,-1,255,-1)
            amask=cv2.bitwise_and(analysis_mask,cv2.bitwise_not(cmask))
            apx=img[amask==255]
            if len(apx)>10:
                agar_color=classify_color(int(np.mean(apx[:,0])),int(np.mean(apx[:,1])),int(np.mean(apx[:,2])))
        # Warnings
        if dish_found and areas:
            dish_area=math.pi*analysis_r**2
            if sum(areas)>0.5*dish_area:
                warns.append("Colonies cover >50% of the dish — possible confluent growth. Count may be inaccurate.")
        if 0 < detect_count < 30:
            warns.append(f"Fewer than 30 colonies ({detect_count}). Statistical methods requiring 30–300 colonies may not apply.")
        if detect_count > 300:
            warns.append(f"More than 300 colonies ({detect_count}). Typically considered ‘too numerous to count’ (TNTC).")
        if detect_count > 0:
            warns.append("Check the annotated image: glare, bubbles, marker pen and dust can be detected as colonies.")
        if not dish_found and auto_dish:
            warns.append("Petri dish boundary not detected; full frame analysed.")

    status.update(label="Examination complete.", state="complete")

avg_d = float(np.mean(diams)) if diams else 0.0
avg_d_mm = avg_d * scale_mm if scale_mm else None
most_common = max(shapes_or_forms.items(), key=lambda x: x[1])[0] if shapes_or_forms and detect_count > 0 else "None"

# =====================================================================
# DETECTION RESULT BANNER
# =====================================================================
chips = f'<span class="ps-pill">{avg_d:.1f} px avg</span>'
if avg_d_mm: chips += f'<span class="ps-pill">~{avg_d_mm:.1f} mm est.</span>'
chips += f'<span class="ps-pill">{_html.escape(most_common)}</span>'
chips += f'<span class="ps-pill"><span class="ps-swatch" style="background:{color_hex}"></span>{_html.escape(color_name)}</span>'
st.markdown(f'''<div class="ps-c accent-top" style="text-align:center;padding:24px">
  <div class="ps-hero-num">{detect_count}</div>
  <div style="font-family:var(--fs);font-size:calc(1rem + 2px);letter-spacing:.04em;color:var(--mt);margin:4px 0 10px">
    {item_word} detected</div>{chips}</div>''', unsafe_allow_html=True)

for w in warns:
    st.warning(w)

# =====================================================================
# RESULT TABS
# =====================================================================
tab_ov, tab_img, tab_chr, tab_rpt = st.tabs(["Overview", "Images", "Characteristics", "Report"])

# ── OVERVIEW ─────────────────────────────────────────────────────────
with tab_ov:
    divider("I")
    q1,q2,q3,q4 = st.columns(4)
    q1.metric("Dimensions", f"{img_w}×{img_h} px")
    q2.metric("Sharpness", f"{sharp:.0f}")
    q3.metric("Brightness", f"{bright:.0f}/255")
    q4.metric("Quality", "Pass" if not q_warns else "Needs attention")
    for w in q_warns: st.warning(w)
    if not q_warns: st.success("Image quality checks passed.")

    divider("II")
    m1,m2,m3 = st.columns(3)
    m1.metric(f"{item_word} detected", detect_count)
    m2.metric("Avg diameter", f"{avg_d:.1f} px")
    if avg_d_mm is not None:
        m3.metric("Avg diameter (est.)", f"~{avg_d_mm:.1f} mm")
        st.caption("Millimetre estimate is approximate, derived from the user-entered dish diameter.")
    else:
        m3.metric("Prevalent form", most_common)

    divider("III")
    st.markdown(f"**{item_word[:-1]} form / outline breakdown**")
    if shapes_or_forms:
        st.markdown(bars(shapes_or_forms), unsafe_allow_html=True)
    else:
        st.info(f"No {item_word.lower()} detected meeting the size criteria.")

    if is_plate and detect_count > 0:
        st.markdown("**Colony size distribution**")
        unit = "mm" if scale_mm else "px"
        ds = [d * scale_mm for d in diams] if scale_mm else diams
        st.markdown(bars(size_bins(ds, unit)), unsafe_allow_html=True)
        if edge_counts:
            st.markdown("**Edge estimate (solidity)**")
            st.markdown(bars(edge_counts), unsafe_allow_html=True)

    # CFU calculator
    if is_plate and detect_count > 0:
        with st.expander("CFU/mL calculator (optional)"):
            do_cfu = st.checkbox("Calculate CFU/mL", key="cfu_on")
            if do_cfu:
                cf1,cf2 = st.columns(2)
                dil = cf1.number_input("Dilution factor", min_value=1, value=1, step=1, key="cfu_dil")
                vol = cf2.number_input("Volume plated (mL)", min_value=0.01, value=0.1, step=0.01, key="cfu_vol")
                cfu = detect_count * dil / vol
                st.metric("CFU/mL (estimated)", f"{cfu:,.0f}")
                st.caption("Meaningful only for 30–300 colonies verified by hand. PocketScope AI counts are estimates.")

# ── IMAGES ───────────────────────────────────────────────────────────
with tab_img:
    blend = st.slider("Original ↔ Detection blend", 0.0, 1.0, 0.5, 0.05, key="blend")
    blended = cv2.addWeighted(img, 1-blend, annotated, blend, 0)
    il,ir = st.columns(2)
    with il:
        st.markdown("**Original**"); st.image(img, use_container_width=True)
    with ir:
        st.markdown(f"**Blend (α={blend:.2f})**"); st.image(blended, use_container_width=True)
    st.markdown("**Detection preview**"); st.image(annotated, use_container_width=True)
    st.download_button("Download annotated PNG", data=to_png(annotated),
                       file_name="pocketscope_detection.png", mime="image/png", use_container_width=True)

# ── CHARACTERISTICS (Panels A, B, D + organism selector) ────────────────
with tab_chr:
    # Organism selector
    org_opts = ["Not confirmed (unknown)"] + list(ORGANISMS.keys())
    sel_org = st.selectbox("Organism confirmed by laboratory methods (optional)", org_opts, key="org_sel")
    st.caption("Select only after Gram stain, culture/biochemical tests, or confirmation by a microbiologist.")
    confirmed = sel_org != "Not confirmed (unknown)"

    divider("A")
    # Panel A: Shape
    note_a = ("Spiral shapes cannot be judged from 2-D contour analysis. "
              "Shape is estimated from outline geometry and is a tentative visual observation.")
    if is_plate:
        card("A — Colony form", bars(shapes_or_forms) +
             (f"<p style='color:var(--mt);font-size:calc(.85rem + 2px)'><b>Edge estimate:</b> "
              + ", ".join(f"{k}: {v}" for k,v in edge_counts.items()) + "</p>" if edge_counts else "") +
             f"<p style='color:var(--mt);font-size:calc(.82rem + 2px)'>{note_a}</p>",
             "Tentative observation from outline geometry", True)
    else:
        card("A — Object outline", bars(shapes_or_forms) +
             f"<p style='color:var(--mt);font-size:calc(.82rem + 2px)'>{note_a}</p>",
             "Tentative observation from outline geometry", True)

    divider("B")
    # Panel B: Colour
    col_body = (f"<p><span class='ps-swatch' style='background:{color_hex}'></span>"
                f"<b>Observed colour:</b> {_html.escape(color_name)}<br>"
                f"<span class='ps-mono'>RGB {color_rgb}  ·  Hex {color_hex}</span></p>")
    if agar_color:
        an,ah,ar = agar_color
        col_body += (f"<p><span class='ps-swatch' style='background:{ah}'></span>"
                     f"<b>Agar colour:</b> {_html.escape(an)} <span class='ps-mono'>({ah})</span></p>")
    col_body += ("<p style='color:var(--mt);font-size:calc(.82rem + 2px)'>Staining, growth medium, lighting and camera settings "
                 "change apparent colour. Colour alone does not identify any organism.</p>")
    card("B — Observed colour", col_body, "Measured inside detected regions", True)

    divider("D")
    # Panel D: Details
    inferred = (f"<b>Inferred from this image:</b><br>"
                f"• Count: {detect_count} {item_word.lower()}<br>"
                f"• Form: {_html.escape(most_common)} (prevalent)<br>"
                f"• Size: avg {avg_d:.1f} px"
                + (f" (~{avg_d_mm:.1f} mm est.)" if avg_d_mm else "") +
                f"<br>• Colour: {_html.escape(color_name)}<br>"
                f"• Quality: {'Pass' if not q_warns else 'Needs attention'}")
    limits = ("<b>Limitations:</b><br>"
              "• Debris, bubbles, glare and dust may be detected as objects.<br>"
              "• Counts depend on threshold settings and image quality.<br>"
              "• Touching colonies can merge or split incorrectly.<br>"
              "• Classification confidence: not available.")
    if confirmed:
        org = ORGANISMS[sel_org]
        edu = (f"<b>General educational information ({_html.escape(sel_org)}):</b><br>"
               f"• {_html.escape(org['gram'])} {_html.escape(org['shape'])}<br>"
               f"• Arrangement: {_html.escape(org['arrangement'])}<br>"
               f"• Colony: {_html.escape(org['colony'])}")
        card("D — Organism details", f"<p>{inferred}</p><hr style='border-color:var(--bd);margin:12px 0'>"
             f"<p>{edu}</p><hr style='border-color:var(--bd);margin:12px 0'><p style='font-size:calc(.82rem + 2px);color:var(--mt)'>{limits}</p>",
             "Selected by user, not detected from image", True)
    else:
        card("D — Organism details", f"<p>{inferred}</p>"
             f"<hr style='border-color:var(--bd);margin:12px 0'><p style='font-size:calc(.82rem + 2px);color:var(--mt)'>{limits}</p>",
             "Only what was measured from this image", True)

# ── REPORT (built from list, no blank gaps) ───────────────────────────
with tab_rpt:
    st.markdown("#### Inspection report preview")
    lines = ["=" * 60, "POCKETSCOPE AI — INSPECTION REPORT", "=" * 60, "",
             "SAMPLE", "-" * 30,
             f"Type          : {st.session_state.sample_type}",
             f"Dimensions    : {img_w} × {img_h} px",
             f"Sharpness     : {sharp:.1f} (threshold {blur_thr})",
             f"Brightness    : {bright:.1f} / 255",
             f"Quality       : {'Pass' if not q_warns else 'Needs attention'}"]
    if is_plate:
        lines += [f"Dish diameter : {dish_mm} mm",
                  f"Dish detected : {'Yes' if dish_found else 'No'}",
                  f"Polarity      : {plate_pol}",
                  f"Split touching: {'Yes' if split_touching else 'No'}"]
    if is_slide:
        lines += [f"Polarity      : {slide_pol}", f"Threshold adj.: {thr_adj}"]
    lines += ["", "MEASUREMENTS", "-" * 30,
              f"{item_word} detected : {detect_count}",
              f"Avg diameter  : {avg_d:.1f} px"]
    if avg_d_mm is not None:
        lines.append(f"Avg diameter  : ~{avg_d_mm:.1f} mm (estimate)")
    lines.append(f"Prevalent form: {most_common}")
    if not is_model:
        lines.append(f"Colour        : {color_name} ({color_hex})")
    if agar_color:
        lines.append(f"Agar colour   : {agar_color[0]} ({agar_color[1]})")
    lines.append(f"Breakdown     : {dict(shapes_or_forms)}")
    if confirmed:
        lines += ["", f"ORGANISM (selected by user, not detected by the app): {sel_org}"]
    if is_plate and st.session_state.get("cfu_on"):
        lines += ["", f"CFU/mL (est.) : {detect_count * st.session_state.get('cfu_dil',1) / st.session_state.get('cfu_vol',0.1):,.0f}",
                  "  (Meaningful only for 30–300 colonies verified by hand.)"]
    lines += ["", "DISCLAIMER", "-" * 30,
              "Experimental analysis for educational use.",
              "=" * 60]
    report = "\n".join(lines)
    st.text_area("Report", report, height=340, label_visibility="collapsed")
    st.download_button("Download report (.txt)", data=report,
                       file_name="pocketscope_report.txt", mime="text/plain", use_container_width=True)

# =====================================================================
# ABOUT & LIMITATIONS
# =====================================================================
with st.expander("About & limitations"):
    a1, a2 = st.columns(2)
    with a1:
        st.markdown("**Project**")
        st.markdown("Problem Statement: Pocket Microscope for Rural Labs")
        st.markdown("Design Goal: Make basic microscopic image capture and analysis more affordable and accessible.")
        st.markdown("Hardware: Smartphone camera · 3D-printed clip-on holder · Lens · LED backlight + stand")
        st.markdown("**Scope of this prototype**")
        st.markdown("Objects are counted by outline.")
        st.markdown("Cell types are not differentiated in OpenCV modes.")
        st.markdown("Colony appearance alone cannot identify species.")
        st.markdown("Colour is influenced by lighting, stain and camera white-balance.")
        st.markdown("Spiral shapes cannot be judged from 2-D contour analysis.")
    with a2:
        st.markdown("**Implemented**")
        st.markdown("Image upload and camera capture with reset")
        st.markdown("Quality checks (sharpness, brightness)")
        st.markdown("OpenCV contour detection and counting")
        st.markdown("Outline shape/form estimate")
        st.markdown("HSV-based colour measurement with swatch")
        st.markdown("Petri dish detection and rim masking")
        st.markdown("Illumination-corrected colony segmentation")
        st.markdown("Watershed colony splitting")
        st.markdown("Colony size distribution (5 bins)")
        st.markdown("Organism reference library (8 organisms, textbook entries)")
        st.markdown("CFU/mL calculator")
        st.markdown("Annotated PNG and TXT report download")
        st.markdown("6 templates (including Clean Light mode), 2 font pairings")
        st.markdown("Optional trained-model mode (when ultralytics and model file are present)")
        st.markdown("**Planned future work**")
        st.markdown("Trained classification on expert-labelled datasets")
        st.markdown("Better segmentation")
        st.markdown("Validated datasets")
        st.markdown("Offline on-phone model")
        st.markdown("Regional-language reports")
