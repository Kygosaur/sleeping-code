# Sleeping Code — Sleep Posture Monitor

A real-time sleep posture detection system using an overhead IR camera, YOLOv11 pose estimation, and a PyQt6 GUI. The system classifies a person's sleeping posture every frame, records video hourly, and automatically uploads footage to Google Drive.

---

## What it does

- Detects and classifies sleeping postures in real time from a webcam feed:
  - **Supine** (lying on back)
  - **Lateral Left / Lateral Right** (lying on side)
  - **Prone** (lying face down)
- Displays live skeleton overlay and posture label with confidence score
- Records clean video (no overlays) in hourly segments
- Automatically uploads completed segments to Google Drive
- Trains ML models (XGBoost, LSTM, Hybrid) on collected pose data

---

## Project structure

```
sleeping code/
├── Gui.py                  # PyQt6 main window and camera display
├── pose_engine.py          # YOLO inference, video recording, GDrive upload
├── sleeping_pose.py        # Posture classifier and per-person tracker
├── extract_from_video.py   # Extracts pose keypoints from recorded video
├── train_xgboost.py        # XGBoost model training
├── train_lstm.py           # LSTM model training
├── train_hybrid.py         # XGBoost + LSTM hybrid training
├── run_all.py              # Runs all three training scripts back to back
├── requirements.txt        # Python dependencies
└── Dockerfile              # Container config (for deployment)
```

---

## Requirements

- Python 3.12 (the version used by the Docker image)
- Webcam (IR overhead camera recommended)
- YOLOv11 pose model: `yolo11m-pose.pt` (place in project root)
- Google Drive service account JSON key (optional, for auto-upload)

---

## Installation

```bash
git clone https://github.com/Kygosaur/sleeping-code.git
cd sleeping-code
pip install -r requirements.txt
```

---

## Usage

**Run the GUI (live monitoring):**
```bash
python Gui.py
```

**Extract keypoints from recorded video:**
```bash
python extract_from_video.py
```

**Train all models:**
```bash
python run_all.py
```

**Train a specific model:**
```bash
python run_all.py --xgboost
python run_all.py --lstm
python run_all.py --hybrid
```

---

## Google Drive setup

1. Create a service account in Google Cloud Console
2. Download the JSON key and place it somewhere safe (do **not** commit it)
3. Use `.env.example` as a safe template for the required variable names
4. Export the variables before launching the application. For example, in PowerShell:
```powershell
$env:GDRIVE_ENABLED = "true"
$env:GOOGLE_SERVICE_ACCOUNT_JSON = "C:\secure\service-account.json"
$env:GDRIVE_FOLDER_ID = "your-folder-id"
python Gui.py
```

Drive upload is disabled by default. Credential JSON files and local `.env` files
are excluded by `.gitignore`; `.env.example` contains placeholders only.

---

## How posture detection works

The classifier uses four signals from YOLO keypoints:

| Signal | What it detects |
|---|---|
| Nose confidence | High = face up (supine), Low = face down (prone) |
| Eye balance | Asymmetry indicates lateral position |
| Ear Y-asymmetry | One ear pressed to mattress = lateral |
| Shoulder Y-asymmetry | Lifted shoulder = lateral |

A 150-frame calibration period at session start learns the person's baseline measurements. Posture must hold for 20 consecutive frames before being committed to avoid flickering.

---

## Model training

Training results are saved to:
```
results/XGBoost Training/
results/LSTM Training/
results/XGB+LSTM Training/
```

Each folder contains `.xlsx` reports and saved model files.

---

## Tech stack

- [YOLOv11](https://github.com/ultralytics/ultralytics) — pose estimation
- [PyQt6](https://pypi.org/project/PyQt6/) — GUI
- [OpenCV](https://opencv.org/) — video capture and processing
- [XGBoost](https://xgboost.readthedocs.io/) — gradient boosted classifier
- [TensorFlow/Keras](https://www.tensorflow.org/) — LSTM model
- [Google Drive API](https://developers.google.com/drive) — auto-upload
