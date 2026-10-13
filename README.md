# PocketScope AI

**Low-cost pocket microscope assistant** by Team BIONEX  
BIOBYTE 2K26 · Crescent Technocrats Club · B.S. Abdur Rahman Crescent Institute

> Educational and research prototype. Not for clinical diagnosis.

---

## What it does

PocketScope AI pairs with a smartphone camera, clip-on macro lens, 3D-printed holder and LED backlight to capture and analyse microscopic images of **culture plates** (colony counting) and **microscope slides** (cell/object counting).

Detection uses **basic OpenCV image processing** (thresholding, contour analysis, morphology, watershed). It is not a trained medical AI model. It does not identify organisms, diagnose diseases, or produce confidence scores.

---

## Quick start (Windows)

### 1. Install Python 3.10 or newer

Download from [python.org](https://www.python.org/downloads/). During installation tick **"Add Python to PATH"**.

### 2. Create a virtual environment

```powershell
cd C:\Users\user\Desktop\PocketScope_AI
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install packages

```powershell
pip install -r requirements.txt
```

### 4. Run the app

```powershell
python -m streamlit run app.py
```

Streamlit prints two URLs:

- **Local URL** `http://localhost:8501` — open this on the laptop
- **Network URL** `http://192.168.x.x:8501` — open this on a phone connected to the same Wi-Fi

---

## Phone + laptop on the same Wi-Fi

1. Both devices must be on the **same network** (same Wi-Fi, or phone hotspot).
2. When Windows Firewall asks, allow access for **Private networks**.
3. Open the **Network URL** (not the Local URL) on the phone browser.
4. Each device gets its own independent session.

### College Wi-Fi blocks devices from seeing each other?

Turn on your phone's **Mobile Hotspot**, connect the laptop to it, re-run the app, and use the new Network URL.

### Phone camera doesn't work in the browser?

Mobile browsers block live camera access over plain `http://`. This is a browser security policy, not a PocketScope AI bug.

**Workaround:** Take the photo with the phone's **native camera app**, then tap **"Upload image"** in PocketScope AI and select the photo from your gallery.

---

## Templates and fonts

The app includes 5 dark colour templates and 2 font pairings, selectable from the workspace controls.

The default fonts (Copperplate Gothic, Footlight MT Light, Cooper Black, Book Antiqua, Baskerville Old Face) are system fonts available on most Windows PCs with Microsoft Office installed. Google Fonts look-alikes (Cinzel, EB Garamond, Caprasimo, Libre Baskerville) are loaded automatically as fallbacks for devices that don't have them.

### Using your own font files

If you have licensed `.ttf` or `.otf` files:

1. Create a `static/` folder inside the project directory.
2. Copy the font files into `static/`.
3. Add `enableStaticServing = true` under `[server]` in `.streamlit/config.toml`.
4. Add `@font-face` rules in the CSS section of `app.py` pointing to `app/static/YourFont.ttf`.

---

## Optional: trained model mode (Blood smear)

This mode is **hidden by default**. It only appears when both:

- The `ultralytics` package is installed: `pip install ultralytics`
- A YOLO model file exists at `models/bloodcell.pt`

To train your own model, see `train_yolo.py` (if present).

The core app runs perfectly without ultralytics. It is not listed in `requirements.txt`.

---

## Project files

| File | Purpose |
|---|---|
| `app.py` | Main Streamlit application |
| `requirements.txt` | Python packages (core only) |
| `.streamlit/config.toml` | Streamlit server configuration |
| `README.md` | This file |
| `models/` | Optional model directory |

---

## Honesty notice

- Detection is basic OpenCV, not a trained medical model.
- No organism names, species, accuracy numbers, disease probabilities, confidence percentages, or diagnoses are generated.
- Colony/object counts are estimates that depend on image quality, lighting, and threshold settings.
- The organism reference library entries are selected by the user, not detected from the image.
- This is an educational prototype for a college expo, not a medical device.

---

*Team BIONEX · BIOBYTE 2K26 · B.S. Abdur Rahman Crescent Institute of Science and Technology*
