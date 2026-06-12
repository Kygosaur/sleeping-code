
import cv2
import time
import threading
import queue
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime

# ── Google Drive imports (optional — upload silently skipped if unavailable) ──
try:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    from google.oauth2.service_account import Credentials as SACredentials
    _GDRIVE_AVAILABLE = True
except ImportError:
    _GDRIVE_AVAILABLE = False

# =========================================================
# CONFIGURATION
# =========================================================
PT_MODEL        = "yolo11m-pose.pt"
OPENVINO_MODEL  = "yolo11m-pose_openvino_model"
IMG_SIZE        = 640
OUTPUT_SIZE     = (640, 360)
GUI_SIZE        = (640, 360)
DISPLAY_EVERY   = 2        # send every Nth frame to GUI
INFER_EVERY     = 2        # run inference every Nth frame, reuse last result otherwise
SEGMENT_SECONDS = 3600     # 1 hour per file
OUTPUT_DIR      = r"C:\Pose Estimation\video"
WARMUP_FRAMES   = 10
CONF_THRESHOLD  = 0.45
GDRIVE_ENABLED          = True
SERVICE_ACCOUNT_JSON    = r"C:\Users\USER\OneDrive\Desktop\projects\Sleeping(COde)\agile-stratum-499202-h1-0a388562f734.json"
GDRIVE_FOLDER_ID        = "1T7G-6zncA8Q2zYrAwq4nm15oBIHDlbtn"
GDRIVE_MAX_RETRIES      = 5     # attempts per file before giving up


# =========================================================
# GOOGLE DRIVE CLIENT (cached singleton)
# =========================================================
_gdrive_service      = None
_gdrive_service_lock = threading.Lock()

def _get_gdrive_service():
    """
    Returns a cached Drive API service object.
    Rebuilds automatically if credentials have expired.
    Thread-safe.
    """
    global _gdrive_service
    with _gdrive_service_lock:
        if _gdrive_service is None:
            creds = SACredentials.from_service_account_file(
                SERVICE_ACCOUNT_JSON,
                scopes=["https://www.googleapis.com/auth/drive.file"]
            )
            _gdrive_service = build("drive", "v3", credentials=creds)
        return _gdrive_service

def _invalidate_gdrive_service():
    global _gdrive_service
    with _gdrive_service_lock:
        _gdrive_service = None


# =========================================================
# GOOGLE DRIVE UPLOADER  (called by the upload worker thread)
# =========================================================
def upload_to_gdrive(filepath: str, log_cb=None):
    """
    Upload one file to Google Drive with exponential-backoff retry.
    Safe to call from any thread.  Uses resumable upload so dropped
    connections mid-file can recover without re-sending everything.

    Args:
        filepath : absolute path to the local file to upload.
        log_cb   : optional callable(msg: str) for GUI status messages.
    """
    def _log(msg):
        print(f"[GDrive] {msg}")
        if log_cb:
            log_cb(msg)

    if not GDRIVE_ENABLED:
        return
    if not _GDRIVE_AVAILABLE:
        _log("google-api-python-client not installed — skipping upload.")
        return
    if not os.path.exists(filepath):
        _log(f"File not found, skipping: {filepath}")
        return

    filename = os.path.basename(filepath)
    _log(f"Queued: {filename}")

    delay = 2.0
    for attempt in range(1, GDRIVE_MAX_RETRIES + 1):
        try:
            service       = _get_gdrive_service()
            file_metadata = {"name": filename, "parents": [GDRIVE_FOLDER_ID]}
            media         = MediaFileUpload(
                filepath,
                mimetype  = "video/mp4",
                resumable = True    # handles large files + dropped connections
            )
            result = service.files().create(
                body       = file_metadata,
                media_body = media,
                fields     = "id, name"
            ).execute()
            _log(f"Done: {result.get('name')} (id: {result.get('id')})")
            return  # success — exit retry loop

        except Exception as exc:
            _invalidate_gdrive_service()   # force re-auth on next attempt
            if attempt < GDRIVE_MAX_RETRIES:
                wait = min(delay, 120.0)
                _log(f"Attempt {attempt} failed ({exc}) — retrying in {wait:.0f}s…")
                time.sleep(wait)
                delay *= 2
            else:
                _log(f"All {GDRIVE_MAX_RETRIES} attempts failed for {filename}: {exc}")


# =========================================================
# UPLOAD WORKER
# =========================================================
class UploadWorker:
    """
    Single background thread that serialises all GDrive uploads.
    Using one thread prevents multiple large uploads competing for
    bandwidth on a slow SIM connection.

    Non-daemon: joins cleanly when drain() is called so in-flight
    uploads finish rather than getting killed mid-way.
    """

    def __init__(self, log_cb=None):
        self._queue   = queue.Queue()
        self._log_cb  = log_cb
        self._thread  = threading.Thread(target=self._run, daemon=False)
        self._thread.start()

    def submit(self, filepath: str):
        """Queue a file for upload."""
        self._queue.put(filepath)

    def drain(self):
        """
        Signal the worker to stop after finishing all pending uploads,
        then block until it has.  Call this from stop() so uploads
        survive an app close.
        """
        self._queue.put(None)   # poison pill
        self._thread.join()     # wait — non-daemon so the OS won't kill it

    def _run(self):
        while True:
            filepath = self._queue.get()
            if filepath is None:
                break
            upload_to_gdrive(filepath, self._log_cb)


# =========================================================
# FRAME READER
# =========================================================
class FrameReader:
    """
    Background thread that continuously reads from the webcam.
    grab() returns the most recent frame without blocking.
    """

    def __init__(self, cap: cv2.VideoCapture):
        self._cap    = cap
        self._frame  = None
        self._lock   = threading.Lock()
        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)

    def grab(self):
        """Return the latest frame, or None if none available yet."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def _run(self):
        while not self._stop.is_set():
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame
            else:
                time.sleep(0.005)


# =========================================================
# POSE DATA COLLECTOR
# =========================================================
class PoseDataCollector:
    """
    Records per-frame pose data keyed by person index (0, 1, 2…).
    Foundation for sleep pattern analysis.
    """

    BUFFER = 1800   # frames (~1 min at 30 fps, ~3 min at 10 fps)

    def __init__(self):
        self._data: dict[int, list] = {}

    def record(self, person_idx: int, keypoints: np.ndarray,
               bbox, timestamp: float):
        if person_idx not in self._data:
            self._data[person_idx] = []
        entry = {
            "timestamp"      : timestamp,
            "keypoints"      : keypoints.copy(),
            "bbox"           : bbox,
            "body_angle"     : self._body_angle(keypoints),
            "movement_score" : self._movement(person_idx, keypoints),
        }
        buf = self._data[person_idx]
        buf.append(entry)
        if len(buf) > self.BUFFER:
            buf.pop(0)

    def get(self, person_idx: int) -> list:
        return self._data.get(person_idx, [])

    def get_epoch_features(self, person_idx: int,
                           window_seconds: float = 30.0) -> dict:
        data = self._data.get(person_idx, [])
        if not data:
            return {}
        now    = data[-1]["timestamp"]
        window = [d for d in data if d["timestamp"] >= now - window_seconds]
        if not window:
            return {}
        angles    = [d["body_angle"]     for d in window
                     if d["body_angle"] is not None]
        movements = [d["movement_score"] for d in window]
        return {
            "mean_angle"      : float(np.mean(angles))              if angles else None,
            "angle_variance"  : float(np.var(angles))               if angles else None,
            "mean_movement"   : float(np.mean(movements)),
            "max_movement"    : float(np.max(movements)),
            "stillness_ratio" : float(np.mean([m < 2.0 for m in movements])),
            "frame_count"     : len(window),
            "window_seconds"  : window_seconds,
        }

    def reset(self):
        self._data.clear()

    def _body_angle(self, kp: np.ndarray):
        ls, rs = kp[5], kp[6]
        lh, rh = kp[11], kp[12]
        if ls[2] < 0.5 or rs[2] < 0.5:
            return None
        if lh[2] < 0.5 and rh[2] < 0.5:
            return None
        smid = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
        if lh[2] > 0.5 and rh[2] > 0.5:
            hmid = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
        elif lh[2] > 0.5:
            hmid = (lh[0], lh[1])
        else:
            hmid = (rh[0], rh[1])
        dx = hmid[0] - smid[0]
        dy = hmid[1] - smid[1]
        return round(float(np.degrees(np.arctan2(abs(dy), abs(dx) + 1e-6))), 2)

    def _movement(self, person_idx: int, kp: np.ndarray) -> float:
        data = self._data.get(person_idx)
        if not data:
            return 0.0
        prev    = data[-1]["keypoints"]
        visible = (kp[:, 2] > 0.5) & (prev[:, 2] > 0.5)
        if visible.sum() == 0:
            return 0.0
        return round(float(
            np.linalg.norm(kp[visible, :2] - prev[visible, :2], axis=1).mean()
        ), 3)


# =========================================================
# POSE ENGINE
# =========================================================
class PoseEngine:

    def __init__(self):
        self.model           = None
        self.cap             = None
        self._reader         = None
        self.running         = False

        self.segment_index   = 1
        self.segment_start   = 0.0
        self.segment_label   = ""
        self.session_start   = 0.0
        self.session_label   = ""
        self.prev_time       = 0.0
        self.webcam_fps      = 20.0

        self._last_seg_path  = ""
        self._frame_count    = 0
        self._collector      = PoseDataCollector()
        self._last_results   = None

        # raw video writer + async queue
        self.out_raw         = None
        self._raw_queue      = None
        self._raw_thread     = None

        # upload worker — created in start(), drained in stop()
        self._upload_worker  = None

        # optional callbacks — set by the GUI before calling start()
        self.on_segment_saved = None   # fn(filepath: str)
        self.on_fps_update    = None   # fn(fps: float)
        self.on_gdrive_log    = None   # fn(msg: str)

        # posture label burned onto the live GUI frame
        self._current_posture_label = ""
        self._current_posture_conf  = 0.0

        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ----------------------------------------------------------
    # MODEL LOADING + WARM-UP
    # ----------------------------------------------------------
    def load_model(self, progress_cb=None):
        def _progress(msg):
            if progress_cb:
                progress_cb(msg)

        if not os.path.exists(OPENVINO_MODEL):
            _progress("OpenVINO model not found — exporting (one-time, please wait)…")
            base = YOLO(PT_MODEL)
            base.export(format="openvino", imgsz=IMG_SIZE)
            _progress("Export complete.")

        _progress("Loading OpenVINO model…")
        self.model = YOLO(OPENVINO_MODEL)

        _progress("Warming up model…")
        dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        for i in range(WARMUP_FRAMES):
            self.model.predict(
                dummy, imgsz=IMG_SIZE, conf=CONF_THRESHOLD,
                device="cpu", verbose=False
            )
            _progress(f"Warm-up {i + 1}/{WARMUP_FRAMES}…")

        _progress("Ready.")

    # ----------------------------------------------------------
    # SESSION CONTROL
    # ----------------------------------------------------------
    def start(self):
        if self.model is None:
            raise RuntimeError("Call load_model() before start().")

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam.")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.webcam_fps    = self.cap.get(cv2.CAP_PROP_FPS) or 20.0
        self.session_start = time.time()
        self.session_label = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.prev_time     = time.time()
        self.segment_index = 1
        self._frame_count  = 0
        self.running       = True

        self._collector.reset()
        self._last_results = None

        self._reader = FrameReader(self.cap)
        self._reader.start()

        # start the raw disk writer
        self.out_raw, _, _ = self._new_segment_raw()
        self._raw_queue    = queue.Queue(maxsize=120)
        self._raw_thread   = threading.Thread(
            target=self._raw_writer_loop, daemon=True
        )
        self._raw_thread.start()

        # start the upload worker (non-daemon — survives stop())
        if GDRIVE_ENABLED:
            self._upload_worker = UploadWorker(log_cb=self.on_gdrive_log)

    def stop(self):
        """
        Stop recording, flush the last segment to disk,
        upload it to GDrive, then release all resources.
        """
        self.running = False

        # 1. Stop raw writer — flushes remaining frames to disk
        if self._raw_queue is not None:
            self._raw_queue.put(None)
        if self._raw_thread is not None:
            self._raw_thread.join(timeout=5.0)
            self._raw_thread = None

        # 2. Release the final raw segment file
        if self.out_raw:
            self.out_raw.release()
            self.out_raw = None

        # 3. Upload the final segment (the one that didn't hit rollover)
        if self._last_seg_path and os.path.exists(self._last_seg_path):
            self._on_segment_done(self._last_seg_path)

        # 4. Drain upload queue — blocks until all pending uploads finish.
        #    This is intentional: on a SIM link we want uploads to complete
        #    before the app closes rather than leaving orphaned partial files
        #    on GDrive.
        if self._upload_worker is not None:
            if self.on_gdrive_log:
                self.on_gdrive_log("Finishing uploads before exit…")
            self._upload_worker.drain()
            self._upload_worker = None

        # 5. Release camera
        if self._reader:
            self._reader.stop()
            self._reader = None
        if self.cap:
            self.cap.release()
            self.cap = None

    # ----------------------------------------------------------
    # PER-FRAME STEP
    # ----------------------------------------------------------
    def step(self):
        """
        Grab the latest webcam frame, run inference, draw skeleton
        onto a display copy for the GUI, write raw frame to disk.

        Returns:
            BGR ndarray for GUI on display frames (skeleton + overlays)
            None on skipped display frames or if no frame available yet
        """
        if not self.running or self._reader is None:
            return None

        frame = self._reader.grab()
        if frame is None:
            return None

        now = time.time()
        self._frame_count += 1

        infer_frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

        # inference every INFER_EVERY frames, reuse cached result otherwise
        if self._frame_count % INFER_EVERY == 0 or self._last_results is None:
            results = self.model.predict(
                infer_frame,
                imgsz   = IMG_SIZE,
                conf    = CONF_THRESHOLD,
                device  = "cpu",
                verbose = False,
            )[0]
            self._last_results = results

            if results.keypoints is not None and results.boxes is not None:
                keypoints = results.keypoints.data.cpu().numpy()
                boxes     = results.boxes.xyxy.cpu().numpy()
                for idx, (person, bbox) in enumerate(zip(keypoints, boxes)):
                    self._collector.record(idx, person, bbox, now)
        else:
            results = self._last_results

        # --- build display frame (skeleton, for GUI only — NOT saved to disk) ---
        annotated_full = results.plot()
        annotated      = cv2.resize(annotated_full, GUI_SIZE)

        # --- timing ---
        current_time   = time.time()
        fps            = 1.0 / max(current_time - self.prev_time, 1e-6)
        self.prev_time = current_time

        if self.on_fps_update:
            self.on_fps_update(fps)

        seg_elapsed = current_time - self.segment_start

        # --- overlays on the display frame (GUI only) ---
        posture_label = self._current_posture_label
        posture_conf  = self._current_posture_conf

        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_small = 0.45
        thickness  = 1
        pad_x, pad_y = 16, 16
        line_h       = 20

        def draw_label(img, text, x, y, txt_col):
            (tw, th), _ = cv2.getTextSize(text, font, font_small, thickness)
            cv2.rectangle(img, (x-6, y-th-4), (x+tw+6, y+4), (0,0,0), -1)
            cv2.rectangle(img, (x-6, y-th-4), (x+tw+6, y+4), (255,255,255), 1)
            cv2.putText(img, text, (x, y), font, font_small,
                        txt_col, thickness, cv2.LINE_AA)

        draw_label(annotated, f"FPS  {fps:.1f}",
                   pad_x, pad_y + 12, (200, 255, 200))
        draw_label(annotated, "RECORDING",
                   pad_x, pad_y + 12 + line_h + 6, (80, 80, 255))

        if posture_label:
            colour_map = {
                "Supine"        : (0,   255, 180),
                "Lateral_Left"  : (0,   180, 255),
                "Lateral_Right" : (0,   220, 255),
                "Prone"         : (255, 140,   0),
                "No_Person"     : (160, 160, 160),
                "Unknown"       : (160, 160, 160),
            }
            p_colour = colour_map.get(posture_label, (200, 200, 200))
            p_text   = posture_label.replace("_", " ")
            if posture_label not in ("No_Person", "Unknown"):
                p_text += f"  {posture_conf:.0%}"
            p_scale = 0.9
            p_thick = 2
            (pw, ph), _ = cv2.getTextSize(p_text, font, p_scale, p_thick)
            px = annotated.shape[1] - pw - 16
            py = 42
            cv2.putText(annotated, p_text, (px, py),
                        font, p_scale, (0,0,0), p_thick+2, cv2.LINE_AA)
            cv2.putText(annotated, p_text, (px, py),
                        font, p_scale, p_colour, p_thick, cv2.LINE_AA)

        # --- push clean raw frame to disk writer (no overlays) ---
        raw_frame = cv2.resize(frame, OUTPUT_SIZE)
        try:
            self._raw_queue.put(raw_frame, timeout=0.5)
        except queue.Full:
            pass

        # --- hourly rollover ---
        if seg_elapsed >= SEGMENT_SECONDS:
            saved = self._last_seg_path
            self._raw_queue.join()
            self.out_raw.release()
            self.segment_index += 1
            self.out_raw, self.segment_start, self.segment_label = \
                self._new_segment_raw()
            self._on_segment_done(saved)

        # return annotated frame to GUI every DISPLAY_EVERY frames
        if self._frame_count % DISPLAY_EVERY == 0:
            return annotated
        return None

    # ----------------------------------------------------------
    # SEGMENT COMPLETION
    # ----------------------------------------------------------
    def _on_segment_done(self, path: str):
        """Notify GUI and queue the file for GDrive upload."""
        if self.on_segment_saved:
            self.on_segment_saved(path)
        if GDRIVE_ENABLED and self._upload_worker is not None:
            self._upload_worker.submit(path)

    # ----------------------------------------------------------
    # PUBLIC HELPERS
    # ----------------------------------------------------------
    def get_epoch_features(self, person_idx: int,
                           window_seconds: float = 30.0) -> dict:
        return self._collector.get_epoch_features(person_idx, window_seconds)

    def set_posture_label(self, posture: str, confidence: float = 0.0):
        self._current_posture_label = posture
        self._current_posture_conf  = confidence

    # ----------------------------------------------------------
    # RAW WRITER LOOP
    # ----------------------------------------------------------
    def _raw_writer_loop(self):
        while True:
            frame = self._raw_queue.get()
            if frame is None:
                self._raw_queue.task_done()
                break
            if self.out_raw is not None:
                self.out_raw.write(frame)
            self._raw_queue.task_done()

    # ----------------------------------------------------------
    # SEGMENT FILE CREATION
    # ----------------------------------------------------------
    def _new_segment_raw(self):
        """
        Create a new raw video segment.
        Filename includes milliseconds to prevent same-second collisions.
        """
        now       = datetime.now()
        label     = now.strftime("%Y-%m-%d %H:%M:%S")
        safe_name = now.strftime("%Y-%m-%d_%H-%M-%S") + f"_{now.microsecond // 1000:03d}"
        path      = os.path.join(OUTPUT_DIR, f"{safe_name}_raw.mp4")

        writer = cv2.VideoWriter(
            path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.webcam_fps,
            OUTPUT_SIZE
        )
        if not writer.isOpened():
            raise RuntimeError(f"Raw VideoWriter failed to open: {path}")

        self._last_seg_path = path
        return writer, time.time(), label