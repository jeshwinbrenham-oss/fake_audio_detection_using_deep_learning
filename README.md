# 🛡️ DeepShield — AI-Based Fake Audio & Video Detection

A complete deep learning pipeline for detecting synthetic/deepfake audio and video using spectral, temporal, spatial, and frequency-domain analysis.

This repository also contains the project content prepared for GitHub publication for the AI research paper analyzer workflow.

---

## 🚀 Quick Start (No Commands Needed)

### Windows
Double-click `START_WINDOWS.bat` — it installs everything and launches the server.

### Mac / Linux
```bash
chmod +x START_MAC_LINUX.sh && ./START_MAC_LINUX.sh
```

Then open **http://127.0.0.1:5000** in your browser.

---

## 📁 Project Structure

```
deepfake_detector/
├── app.py                    ← Flask web server (main entry point)
├── database.py               ← MySQL / SQLite persistence layer
├── requirements.txt          ← Python dependencies
├── START_WINDOWS.bat         ← One-click Windows launcher
├── START_MAC_LINUX.sh        ← One-click Mac/Linux launcher
├── utils/
│   ├── audio_detector.py     ← Audio deepfake detection engine
│   └── video_detector.py     ← Video deepfake detection engine
├── templates/
│   └── index.html            ← Full web UI
└── static/
    └── uploads/              ← Temporary upload directory
```

---

## 🧠 Detection Techniques

### Audio Detection
| Feature | Description |
|---|---|
| MFCC (40 coefficients) | Vocal tract shape, formant patterns |
| Chroma Features | Tonal content & pitch class distribution |
| Spectral Centroid | Brightness — TTS voices have narrower range |
| Harmonic Ratio | GAN audio lacks natural voice harmonics |
| Zero Crossing Rate | Noise & voiced/unvoiced proportion |
| Mel Spectrogram | Dynamic range compressed in synthetic audio |
| Tonnetz | Tonal centroid features for naturalness |

### Video Detection
| Technique | What It Catches |
|---|---|
| Face Consistency | Flickering boundaries in GAN-swapped faces |
| Optical Flow | Unnatural motion vectors around face region |
| FFT Frequency Analysis | High-frequency GAN fingerprints |
| Texture Analysis (Laplacian) | Over-smooth skin — neural renderer artifact |
| Blink Detection | Lack of natural eye movement in early deepfakes |

---

## 🗄️ Database Configuration

**Default: SQLite** (zero setup, works immediately)

**To use MySQL:**
1. Open `database.py`
2. Set `USE_MYSQL = True`
3. Fill in your credentials in `MYSQL_CONFIG`
4. Create the database: `CREATE DATABASE deepfake_db;`

The schema is auto-created on first run.

---

## 📡 API Endpoints

| Method | Route | Description |
|---|---|---|
| GET | `/` | Web UI |
| POST | `/api/analyze` | Upload & analyze file |
| GET | `/api/history` | Last 20 detections |
| GET | `/api/stats` | Aggregate statistics |

---

## 🛠️ Tech Stack
- **Backend**: Python 3.8+ / Flask
- **Audio**: Librosa, SoundFile, NumPy, SciPy
- **Video**: OpenCV (cv2), NumPy
- **ML**: Scikit-learn, custom ensemble scoring
- **DB**: MySQL or SQLite (auto-fallback)
- **Frontend**: Vanilla HTML/CSS/JS (no framework needed)
