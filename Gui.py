import sys
import time
import cv2
import numpy as np
from PyQt6.QtCore    import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui     import QColor, QFont, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel,
    QPushButton, QVBoxLayout, QHBoxLayout,
    QStatusBar, QFrame, QSizePolicy,
    QGraphicsDropShadowEffect
)

from pose_engine import PoseEngine, OUTPUT_DIR
from sleeping_pose import PersonPostureTracker


class PoseWorker(QThread):

    fps_updated     = pyqtSignal(float)
    segment_saved   = pyqtSignal(str)
    status_msg      = pyqtSignal(str)
    error_occurred  = pyqtSignal(str)
    posture_updated = pyqtSignal(str, float)   # (posture_str, confidence)
    gdrive_log      = pyqtSignal(str)          # GDrive upload status messages

    def __init__(self):
        super().__init__()
        self.engine        = PoseEngine()
        self.tracker       = PersonPostureTracker(person_idx=0, smooth=True)
        self._model_loaded = False
        self._stop_flag    = False
        self.latest_frame  = None   # read by main thread QTimer

    def _load_model(self):
        try:
            self.engine.load_model(
                progress_cb=lambda msg: self.status_msg.emit(msg)
            )
            self._model_loaded = True
        except Exception as exc:
            self.error_occurred.emit(f"Model load failed: {exc}")

    # ----------------------------------------------------------
    def run(self):
        self._stop_flag   = False
        self.latest_frame = None

        if not self._model_loaded:
            self._load_model()
            if not self._model_loaded:
                return

        try:
            self.engine.on_fps_update    = lambda fps:  self.fps_updated.emit(fps)
            self.engine.on_segment_saved = lambda path: self.segment_saved.emit(path)
            # wire GDrive log messages back to the GUI status bar
            self.engine.on_gdrive_log    = lambda msg:  self.gdrive_log.emit(msg)
            self.engine.start()
            self.tracker.reset()
            self.status_msg.emit("Recording…")
        except Exception as exc:
            self.error_occurred.emit(str(exc))
            return

        while not self._stop_flag:
            frame = self.engine.step()

            # --- posture classification ---
            collector_data = self.engine._collector.get(person_idx=0)

            # Check whether YOLO found anyone this frame
            last_results = self.engine._last_results
            person_count = (
                len(last_results.boxes)
                if last_results is not None and last_results.boxes is not None
                else 0
            )

            if person_count == 0:
                # No person in frame
                result = self.tracker.update_no_person(time.time())
                self.posture_updated.emit(result.posture, result.confidence)
                self.engine.set_posture_label(result.posture, result.confidence)
            elif collector_data:
                last   = collector_data[-1]
                result = self.tracker.update(last["keypoints"], last["timestamp"])
                self.posture_updated.emit(result.posture, result.confidence)
                self.engine.set_posture_label(result.posture, result.confidence)

                # burn posture label onto the frame shown in the GUI
                if frame is not None:
                    colour = {
                        "Supine"        : (0,   255, 180),
                        "Lateral_Left"  : (0,   180, 255),
                        "Lateral_Right" : (0,   220, 255),
                        "Prone"         : (255, 140,   0),
                        "No_Person"     : (160, 160, 160),
                    }.get(result.posture, (200, 200, 200))
                    label = f"{result.posture.replace('_',' ')}  {result.confidence:.0%}"
                    cv2.putText(frame, label, (12, 42),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                (0, 0, 0), 4, cv2.LINE_AA)   # black outline
                    cv2.putText(frame, label, (12, 42),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                colour,    2, cv2.LINE_AA)   # coloured text

            if frame is not None:
                self.latest_frame = frame   # main thread picks this up

        self.engine.stop()
        self.tracker.reset()
        self.engine.set_posture_label("", 0.0)
        self.latest_frame = None
        self.status_msg.emit("Stopped.")

    def stop(self):
        self._stop_flag = True


# =========================================================
# MAIN WINDOW
# =========================================================
class MainWindow(QMainWindow):

    # ── colour palette ──────────────────────────────────
    BG_DARK  = "#0e0f11"
    BG_MID   = "#16181c"
    BG_CARD  = "#1e2128"
    ACCENT   = "#00e5a0"        # mint-green  — START / active
    ACCENT2  = "#0099ff"        # blue        — info highlights
    DANGER   = "#ff4757"        # red         — STOP / error
    TEXT_PRI = "#e8eaf0"
    TEXT_SEC = "#7a7f8e"
    BORDER   = "#2a2d35"

    def __init__(self):
        super().__init__()
        self.worker         = None
        self.is_running     = False
        self.session_start  = None
        self.saved_segments = []
        self._blink_state   = False

        self._build_ui()
        self._apply_styles()

        # 1-second clock for session elapsed
        self._clock = QTimer(self)
        self._clock.timeout.connect(self._tick_clock)
        self._clock.start(1000)

        # blinking REC dot
        self._blink_timer = QTimer(self)
        self._blink_timer.timeout.connect(self._blink_rec)

        # polls worker.latest_frame every 30ms and renders into PyQt label
        self._cv2_timer = QTimer(self)
        self._cv2_timer.timeout.connect(self._show_cv2_frame)
        self._cv2_timer.setInterval(30)

    # ======================================================
    # UI BUILD
    # ======================================================
    def _build_ui(self):
        self.setWindowTitle("Pose Estimation Controller")
        self.setMinimumSize(1100, 760)

        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ── LEFT PANEL ────────────────────────────────────
        left = QWidget()
        left.setFixedWidth(300)
        left.setObjectName("leftPanel")
        ll = QVBoxLayout(left)
        ll.setContentsMargins(24, 32, 24, 32)
        ll.setSpacing(0)

        brand = QLabel("POSE\nCAM")
        brand.setObjectName("brand")
        brand.setAlignment(Qt.AlignmentFlag.AlignLeft)
        ll.addWidget(brand)

        ll.addSpacing(8)

        tagline = QLabel("Pose Estimation Controller")
        tagline.setObjectName("tagline")
        ll.addWidget(tagline)

        ll.addSpacing(40)

        # status card
        sc = self._card()
        sc_l = QVBoxLayout(sc)
        sc_l.setContentsMargins(16, 16, 16, 16)
        sc_l.setSpacing(8)
        sc_l.addWidget(self._card_title("STATUS"))
        self.lbl_status = QLabel("Idle")
        self.lbl_status.setObjectName("statusValue")
        self.lbl_status.setWordWrap(True)
        sc_l.addWidget(self.lbl_status)
        ll.addWidget(sc)

        ll.addSpacing(16)

        # stats card
        stats = self._card()
        sl = QVBoxLayout(stats)
        sl.setContentsMargins(16, 16, 16, 16)
        sl.setSpacing(10)
        sl.addWidget(self._card_title("LIVE STATS"))
        self.lbl_elapsed  = self._stat_row(sl, "Elapsed",  "—")
        self.lbl_segments = self._stat_row(sl, "Segments", "0")
        self.lbl_save_dir = self._stat_row(sl, "Save dir", OUTPUT_DIR)
        self.lbl_save_dir.setToolTip(OUTPUT_DIR)
        ll.addWidget(stats)

        ll.addSpacing(16)

        # posture card
        posture_card = self._card()
        pc_l = QVBoxLayout(posture_card)
        pc_l.setContentsMargins(16, 16, 16, 16)
        pc_l.setSpacing(8)
        pc_l.addWidget(self._card_title("POSTURE"))
        self.lbl_posture = QLabel("—")
        self.lbl_posture.setObjectName("postureValue")
        self.lbl_posture.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pc_l.addWidget(self.lbl_posture)
        self.lbl_posture_conf = QLabel("")
        self.lbl_posture_conf.setObjectName("postureConf")
        self.lbl_posture_conf.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pc_l.addWidget(self.lbl_posture_conf)
        ll.addWidget(posture_card)

        ll.addSpacing(16)

        # last saved card
        saved = self._card()
        sav_l = QVBoxLayout(saved)
        sav_l.setContentsMargins(16, 16, 16, 16)
        sav_l.setSpacing(6)
        sav_l.addWidget(self._card_title("LAST SAVED"))
        self.lbl_last_saved = QLabel("None yet")
        self.lbl_last_saved.setObjectName("savedPath")
        self.lbl_last_saved.setWordWrap(True)
        sav_l.addWidget(self.lbl_last_saved)
        ll.addWidget(saved)

        ll.addStretch()

        # START button
        self.btn_start = QPushButton("▶  START")
        self.btn_start.setObjectName("btnStart")
        self.btn_start.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_start.clicked.connect(self._on_start)
        self._shadow(self.btn_start, self.ACCENT)
        ll.addWidget(self.btn_start)

        ll.addSpacing(12)

        # STOP button
        self.btn_stop = QPushButton("■  STOP")
        self.btn_stop.setObjectName("btnStop")
        self.btn_stop.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop)
        ll.addWidget(self.btn_stop)

        root_layout.addWidget(left)

        # ── RIGHT PANEL ────────────────────────────────────
        right = QWidget()
        right.setObjectName("rightPanel")
        rl = QVBoxLayout(right)
        rl.setContentsMargins(24, 24, 24, 24)
        rl.setSpacing(12)

        # header row
        hdr = QHBoxLayout()
        feed_title = QLabel("LIVE FEED")
        feed_title.setObjectName("feedTitle")
        hdr.addWidget(feed_title)
        hdr.addStretch()

        self.lbl_rec_dot = QLabel("●")
        self.lbl_rec_dot.setObjectName("recDotOff")
        hdr.addWidget(self.lbl_rec_dot)

        self.lbl_rec_text = QLabel("NOT RECORDING")
        self.lbl_rec_text.setObjectName("recText")
        hdr.addWidget(self.lbl_rec_text)

        rl.addLayout(hdr)

        # video label — frames rendered directly here (no cv2 window needed)
        self.video_label = QLabel("Camera Off")
        self.video_label.setObjectName("videoFrame")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
        rl.addWidget(self.video_label, stretch=1)

        root_layout.addWidget(right, stretch=1)

        # status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready — press START to begin.")

    # ======================================================
    # WIDGET HELPERS
    # ======================================================
    def _card(self):
        f = QFrame()
        f.setObjectName("card")
        return f

    def _card_title(self, text):
        lbl = QLabel(text)
        lbl.setObjectName("cardTitle")
        return lbl

    def _stat_row(self, layout, key, value):
        row = QHBoxLayout()
        k = QLabel(key)
        k.setObjectName("statKey")
        v = QLabel(value)
        v.setObjectName("statVal")
        v.setAlignment(Qt.AlignmentFlag.AlignRight)
        row.addWidget(k)
        row.addWidget(v)
        layout.addLayout(row)
        return v

    def _shadow(self, widget, colour_hex, blur=20, offset=6):
        fx = QGraphicsDropShadowEffect()
        fx.setBlurRadius(blur)
        fx.setOffset(0, offset)
        fx.setColor(QColor(colour_hex))
        widget.setGraphicsEffect(fx)

    # ======================================================
    # STYLESHEET
    # ======================================================
    def _apply_styles(self):
        self.setStyleSheet(f"""
        QMainWindow, QWidget {{
            background: {self.BG_DARK};
            color: {self.TEXT_PRI};
            font-family: 'Consolas', 'Courier New', monospace;
        }}

        #leftPanel {{
            background: {self.BG_MID};
            border-right: 1px solid {self.BORDER};
        }}

        #brand {{
            font-size: 38px;
            font-weight: 900;
            letter-spacing: 6px;
            color: {self.ACCENT};
            line-height: 1.1;
        }}

        #tagline {{
            font-size: 10px;
            letter-spacing: 3px;
            color: {self.TEXT_SEC};
        }}

        #card {{
            background: {self.BG_CARD};
            border: 1px solid {self.BORDER};
            border-radius: 10px;
        }}

        #cardTitle {{
            font-size: 9px;
            letter-spacing: 3px;
            color: {self.TEXT_SEC};
            margin-bottom: 4px;
        }}

        #statusValue {{
            font-size: 13px;
            color: {self.ACCENT};
            font-weight: bold;
        }}

        #statKey {{
            font-size: 11px;
            color: {self.TEXT_SEC};
        }}

        #statVal {{
            font-size: 11px;
            color: {self.TEXT_PRI};
            font-weight: bold;
        }}

        #savedPath {{
            font-size: 10px;
            color: {self.ACCENT2};
        }}

        #postureValue {{
            font-size: 20px;
            font-weight: 900;
            color: {self.TEXT_SEC};
        }}

        #postureConf {{
            font-size: 10px;
            color: {self.TEXT_SEC};
        }}

        #btnStart {{
            background: {self.ACCENT};
            color: #000;
            border: none;
            border-radius: 8px;
            padding: 14px;
            font-size: 14px;
            font-weight: 900;
            letter-spacing: 3px;
        }}
        #btnStart:hover   {{ background: #00ffb3; }}
        #btnStart:pressed {{ background: #00c984; }}
        #btnStart:disabled {{
            background: {self.BG_CARD};
            color: {self.TEXT_SEC};
        }}

        #btnStop {{
            background: transparent;
            color: {self.DANGER};
            border: 2px solid {self.DANGER};
            border-radius: 8px;
            padding: 12px;
            font-size: 14px;
            font-weight: 900;
            letter-spacing: 3px;
        }}
        #btnStop:hover {{
            background: {self.DANGER};
            color: #fff;
        }}
        #btnStop:pressed  {{ background: #cc2233; }}
        #btnStop:disabled {{
            color: {self.BORDER};
            border-color: {self.BORDER};
        }}

        #rightPanel {{ background: {self.BG_DARK}; }}

        #feedTitle {{
            font-size: 10px;
            letter-spacing: 4px;
            color: {self.TEXT_SEC};
        }}

        #recDotOff   {{ color: {self.BORDER};  font-size: 14px; }}
        #recDotOn    {{ color: {self.DANGER};  font-size: 14px; }}
        #recDotBlink {{ color: transparent;    font-size: 14px; }}

        #recText {{
            font-size: 10px;
            letter-spacing: 2px;
            color: {self.TEXT_SEC};
            margin-left: 4px;
        }}

        #videoFrame {{
            background: {self.BG_CARD};
            border: 1px solid {self.BORDER};
            border-radius: 10px;
            color: {self.TEXT_SEC};
            font-size: 16px;
            letter-spacing: 2px;
        }}

        QStatusBar {{
            background: {self.BG_MID};
            color: {self.TEXT_SEC};
            font-size: 10px;
            border-top: 1px solid {self.BORDER};
        }}
        """)

    # ======================================================
    # BUTTON HANDLERS
    # ======================================================
    def _on_start(self):
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.is_running     = True
        self.session_start  = time.time()
        self.saved_segments = []

        self.lbl_status.setText("Starting…")
        self.lbl_rec_dot.setObjectName("recDotOn")
        self.lbl_rec_text.setText("RECORDING")
        self._apply_styles()
        self._blink_timer.start(600)

        self.worker = PoseWorker()
        self.worker.segment_saved.connect(self._on_segment_saved)
        self.worker.status_msg.connect(self._on_status)
        self.worker.error_occurred.connect(self._on_error)
        self.worker.posture_updated.connect(self._on_posture)
        self.worker.finished.connect(self._on_worker_finished)
        # GDrive log messages go to the status bar
        self.worker.gdrive_log.connect(self._on_gdrive_log)
        self.worker.start()

        # start polling for frame display
        self._cv2_timer.start()

    def _on_stop(self):
        self.btn_stop.setEnabled(False)
        self.lbl_status.setText("Stopping…")
        self.status_bar.showMessage("Stopping — finishing current segment…")
        if self.worker:
            self.worker.stop()

    # ======================================================
    # WORKER SIGNAL HANDLERS
    # ======================================================
    def _show_cv2_frame(self):
        """Called by QTimer every 30ms — renders latest frame into the PyQt label."""
        if self.worker is None or self.worker.latest_frame is None:
            return

        frame = self.worker.latest_frame

        # Convert OpenCV BGR → PyQt RGB
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch  = rgb_image.shape
        qt_image  = QImage(
            rgb_image.data, w, h,
            ch * w,
            QImage.Format.Format_RGB888
        )

        # Scale to fit the label while keeping aspect ratio
        pixmap = QPixmap.fromImage(qt_image)
        scaled = pixmap.scaled(
            self.video_label.width(),
            self.video_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.video_label.setPixmap(scaled)

    def _on_posture(self, posture: str, confidence: float):
        colour_map = {
            "Supine"        : "#00e5a0",   # mint
            "Lateral_Left"  : "#0099ff",   # blue
            "Lateral_Right" : "#00ccff",   # light blue
            "Prone"         : "#ff9f43",   # orange
            "No_Person"     : "#ff4757",   # red
            "Unknown"       : "#7a7f8e",   # grey
        }
        colour = colour_map.get(posture, "#7a7f8e")
        self.lbl_posture.setText(posture.replace("_", " "))
        self.lbl_posture.setStyleSheet(
            f"color: {colour}; font-size: 20px; font-weight: 900;"
        )
        if posture == "No_Person":
            self.lbl_posture_conf.setText("no detection")
        else:
            self.lbl_posture_conf.setText(f"{confidence:.0%} confidence")
        self.lbl_posture_conf.setStyleSheet(f"color: {colour}; font-size: 10px;")

    def _on_segment_saved(self, path: str):
        self.saved_segments.append(path)
        self.lbl_segments.setText(str(len(self.saved_segments)))
        name = path.replace("\\", "/").split("/")[-1]
        self.lbl_last_saved.setText(name)
        self.status_bar.showMessage(f"Saved: {name}")

    def _on_status(self, msg: str):
        self.lbl_status.setText(msg)
        self.status_bar.showMessage(msg)

    def _on_error(self, msg: str):
        self.lbl_status.setText(f"ERROR: {msg}")
        self.status_bar.showMessage(f"Error: {msg}")
        self._on_worker_finished()

    def _on_gdrive_log(self, msg: str):
        """GDrive upload status — shown in status bar only (non-intrusive)."""
        self.status_bar.showMessage(f"[GDrive] {msg}")

    def _on_worker_finished(self):
        self.is_running = False
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._blink_timer.stop()
        self._cv2_timer.stop()
        self.lbl_rec_dot.setObjectName("recDotOff")
        self.lbl_rec_text.setText("NOT RECORDING")
        self._apply_styles()
        self.video_label.clear()
        self.video_label.setText("Camera Off")
        self.lbl_elapsed.setText("—")
        self.lbl_posture.setText("—")
        self.lbl_posture.setStyleSheet("")
        self.lbl_posture_conf.setText("")
        self.session_start = None

    # ======================================================
    # TIMERS
    # ======================================================
    def _tick_clock(self):
        if self.session_start and self.is_running:
            elapsed = time.time() - self.session_start
            self.lbl_elapsed.setText(
                time.strftime("%H:%M:%S", time.gmtime(elapsed))
            )

    def _blink_rec(self):
        self._blink_state = not self._blink_state
        self.lbl_rec_dot.setObjectName(
            "recDotBlink" if self._blink_state else "recDotOn"
        )
        self._apply_styles()

    # ======================================================
    # CLOSE — gracefully stop worker before exit
    # ======================================================
    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(4000)
        self._cv2_timer.stop()
        event.accept()


# =========================================================
# ENTRY POINT
# =========================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())   # PyQt6: exec() not exec_()