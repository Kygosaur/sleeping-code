import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

NOSE           = 0
LEFT_EYE       = 1
RIGHT_EYE      = 2
LEFT_EAR       = 3
RIGHT_EAR      = 4
LEFT_SHOULDER  = 5
RIGHT_SHOULDER = 6
LEFT_HIP       = 11
RIGHT_HIP      = 12

CONF_THR      = 0.50
CONF_THR_WEAK = 0.28

class Thresholds:

    # ── Nose confidence ──────────────────────────────────
    # IR camera overhead: supine person's nose points
    # straight at the lens → very high confidence.
    # Prone person's nose points into mattress → near zero.
    NOSE_CONF_SUPINE     = 0.60   # ≥ this → face-up signal
    NOSE_CONF_PRONE      = 0.25   # ≤ this → face-down signal

    # ── Eye confidence ───────────────────────────────────
    BOTH_EYES_CONF       = 0.50   # both eyes clearly visible
    ONE_EYE_CONF         = 0.38   # one eye visible, one hidden

    # ── Ear Y-asymmetry ──────────────────────────────────
    # Normalised by inter-ear horizontal distance.
    # Overhead camera: lateral person has one ear pressed
    # to mattress (lower in image = higher Y value).
    EAR_ASYM_LATERAL     = 0.28   # ≥ this → lateral signal
    EAR_ASYM_PRONE       = 0.58   # ≥ this → prone/deep-lateral

    # ── Shoulder Y-asymmetry ─────────────────────────────
    # Normalised by shoulder width.
    # Supine:  both shoulders same height → low asymmetry
    # Lateral: one shoulder lifted        → high asymmetry
    # Prone:   shoulders roughly level but rotated
    SHLD_ASYM_LATERAL    = 0.18   # ≥ this → lateral signal
    SHLD_ASYM_PRONE      = 0.08   # prone shoulders roughly level

    # ── Prone special rule ───────────────────────────────
    # At 30cm overhead: prone = nose hidden + ears roughly
    # symmetric (head turned slightly, both ears visible)
    # This distinguishes prone from deep lateral.
    PRONE_EAR_ASYM_MAX   = 0.40   # ears roughly level while prone (raised: tilted heads still qualify)

    # ── Smothered-face prone rule ─────────────────────────
    # Nose AND both eyes hidden simultaneously = face buried
    # in pillow/mattress → very strong prone signal regardless
    # of ear or shoulder geometry.
    SMOTHERED_NOSE_MAX   = 0.30   # nose conf below this = hidden
    SMOTHERED_EYE_MAX    = 0.32   # both eyes below this = hidden

    # ── Evidence & confidence ────────────────────────────
    MIN_EVIDENCE         = 0.50   # raised — we expect 4 good signals
    # NOTE: BODY_BONUS removed — was defined but never used

    # ── Calibration ──────────────────────────────────────
    # Session starts supine — first N high-confidence frames
    # are used to learn this person's shoulder/eye/ear widths.
    CALIB_FRAMES         = 150    # frames to collect (15 sec at 10fps)
    CALIB_NOSE_MIN       = 0.65   # min nose conf to count as valid supine
    SHLD_LATERAL_RATIO   = 0.72   # shoulder width < this% of baseline → lateral
    SHLD_PRONE_RATIO     = 0.85   # shoulder width < this% but > lateral → prone hint
    EYE_LATERAL_RATIO    = 0.70   # eye distance < this% of baseline → lateral hint
    EAR_LATERAL_RATIO    = 0.70   # ear distance < this% of baseline → lateral hint

    # ── Smoothing ────────────────────────────────────────
    SMOOTH_WINDOW        = 30     # frames for majority vote
    MIN_HISTORY          = 8      # frames before smoothed result trusted
    STABLE_FRAMES        = 20     # frames posture must hold before commit
                                  # at ~10fps this = ~2 seconds


# =========================================================
# RESULT
# =========================================================
@dataclass
class PostureResult:
    posture    : str   = "Unknown"
    confidence : float = 0.0
    method     : str   = ""
    detail     : dict  = field(default_factory=dict)

    def __str__(self):
        return f"{self.posture} ({self.confidence:.0%})"

    @property
    def is_lateral(self) -> bool:
        return self.posture in ("Lateral_Left", "Lateral_Right")

    @property
    def is_no_person(self) -> bool:
        return self.posture == "No_Person"

    @property
    def base_posture(self) -> str:
        """Returns 'Lateral' for both lateral variants, else posture."""
        return "Lateral" if self.is_lateral else self.posture


# =========================================================
# HELPERS
# =========================================================
def _conf(kp: np.ndarray, idx: int) -> float:
    return float(kp[idx, 2])

def _visible(kp: np.ndarray, idx: int,
             thr: float = CONF_THR) -> bool:
    return _conf(kp, idx) >= thr

def _pt(kp: np.ndarray, idx: int,
        thr: float = CONF_THR) -> Optional[np.ndarray]:
    return kp[idx, :2].copy() if _visible(kp, idx, thr) else None

def _midpoint(kp: np.ndarray, a: int, b: int) -> Optional[np.ndarray]:
    pa, pb = _pt(kp, a), _pt(kp, b)
    if pa is None or pb is None:
        return None
    return (pa + pb) / 2.0

def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


# =========================================================
# CLASSIFIER
# =========================================================
class PostureClassifier:
    """
    Sleep posture classifier optimised for overhead IR camera
    with upper body always in frame.

    Four equal-weight signals:
        1. Nose confidence
        2. Eye balance
        3. Ear Y-asymmetry
        4. Shoulder Y-asymmetry  (promoted to primary — always visible)

    Lateral side (Left/Right) resolved from ear + shoulder
    pressed-down direction.
    """

    def __init__(self, thresholds: Thresholds = None):
        self.t = thresholds or Thresholds()
        self._history: deque = deque(maxlen=self.t.SMOOTH_WINDOW)

        # ── Calibration state ────────────────────────────────
        # Populated during the first CALIB_FRAMES supine frames.
        # Once calibrated, ratio-based signals replace fixed thresholds
        # for shoulder width, eye distance, and ear distance.
        self._calib_done        : bool  = False
        self._calib_frames      : int   = 0
        self._calib_total_seen  : int   = 0   # FIX: total frames seen for fallback
        self._calib_shld_widths : list  = []
        self._calib_eye_dists   : list  = []
        self._calib_ear_dists   : list  = []
        # Baseline medians (pixels) — set after calibration
        self.baseline_shld_width : float = 0.0
        self.baseline_eye_dist   : float = 0.0
        self.baseline_ear_dist   : float = 0.0
        self.calibrated          : bool  = False   # public flag

    # ----------------------------------------------------------
    # PUBLIC
    # ----------------------------------------------------------
    def classify_frame(self, kp: np.ndarray) -> PostureResult:
        detail = {}

        # collect supine baseline during first 15 seconds
        self._collect_calibration(kp)

        # apply calibration ratio signals if available
        calib = self._calibration_signals(kp, detail)

        scores, evidence = self._score(kp, detail, calib)
        detail["scores"] = {k: round(v, 3) for k, v in scores.items()}
        detail["evidence"] = round(evidence, 3)

        if not scores or evidence < self.t.MIN_EVIDENCE:
            detail["reason"] = f"low evidence ({evidence:.2f})"
            return PostureResult("Unknown", 0.0, "unknown", detail)

        best     = max(scores, key=scores.get)
        raw_conf = scores[best]

        # penalise ambiguity
        sorted_vals = sorted(scores.values(), reverse=True)
        if len(sorted_vals) > 1:
            gap      = sorted_vals[0] - sorted_vals[1]
            raw_conf = raw_conf * (0.65 + 0.35 * min(gap / 0.25, 1.0))

        # resolve lateral side
        if best == "Lateral":
            best = self._lateral_side(kp, detail)

        return PostureResult(best, round(raw_conf, 2), "head+shoulder", detail)

    def classify_smoothed(self, kp: np.ndarray) -> PostureResult:
        result = self.classify_frame(kp)

        if result.posture != "Unknown":
            self._history.append((result.posture, result.confidence))

        if len(self._history) < self.t.MIN_HISTORY:
            return result

        counts   = {}
        conf_sum = {}
        for p, c in self._history:
            counts[p]   = counts.get(p, 0) + 1
            conf_sum[p] = conf_sum.get(p, 0.0) + c

        best        = max(counts, key=counts.get)
        vote_ratio  = counts[best] / len(self._history)
        mean_conf   = conf_sum[best] / counts[best]
        smooth_conf = round((vote_ratio + mean_conf) / 2.0, 2)

        return PostureResult(best, smooth_conf,
                             result.method, result.detail)

    def reset(self):
        self._history.clear()
        self._calib_done         = False
        self._calib_frames       = 0
        self._calib_total_seen   = 0   # FIX: reset fallback counter
        self._calib_shld_widths  = []
        self._calib_eye_dists    = []
        self._calib_ear_dists    = []
        self.baseline_shld_width = 0.0
        self.baseline_eye_dist   = 0.0
        self.baseline_ear_dist   = 0.0
        self.calibrated          = False

    # ----------------------------------------------------------
    # SCORING  — four equal-weight signals
    # ----------------------------------------------------------
    def _score(self, kp: np.ndarray, detail: dict, calib: dict = None):
        """
        Returns (scores dict, total evidence accumulated).
        Each of 4 signals contributes up to 0.25 evidence.
        Full evidence = 1.0 (all 4 signals reliable).

        Prone improvements:
          1. Smothered-face rule: nose AND both eyes hidden → direct
             prone override, bypasses normal signal weighting.
          2. Head-tilted prone: nose hidden + at least one eye hidden
             → strong prone signal even if one ear is visible.
          3. Shoulder low-asymmetry is only credited to Supine when
             the nose/eye evidence does NOT point to Prone first.
        """
        scores   = {"Supine": 0.0, "Lateral": 0.0, "Prone": 0.0}
        evidence = 0.0
        w        = 0.25   # equal weight per signal

        nose_c = _conf(kp, NOSE)
        le_c   = _conf(kp, LEFT_EYE)
        re_c   = _conf(kp, RIGHT_EYE)
        lear_c = _conf(kp, LEFT_EAR)
        rear_c = _conf(kp, RIGHT_EAR)

        detail.update({
            "nose_conf"      : round(nose_c,  2),
            "left_eye_conf"  : round(le_c,    2),
            "right_eye_conf" : round(re_c,    2),
            "left_ear_conf"  : round(lear_c,  2),
            "right_ear_conf" : round(rear_c,  2),
        })

        # ── SMOTHERED-FACE PRONE OVERRIDE ─────────────────────
        # Nose AND both eyes all hidden at the same time = face
        # buried in pillow or mattress.  Return a confident Prone
        # result immediately without going through the normal signals.
        # (Head may be tilted so ears/shoulders won't be symmetric.)
        nose_hidden = nose_c <= self.t.SMOTHERED_NOSE_MAX
        both_eyes_hidden = (le_c < self.t.SMOTHERED_EYE_MAX and
                            re_c < self.t.SMOTHERED_EYE_MAX)
        one_eye_hidden   = (le_c < self.t.SMOTHERED_EYE_MAX or
                            re_c < self.t.SMOTHERED_EYE_MAX)

        if nose_hidden and both_eyes_hidden:
            detail["prone_trigger"] = "smothered_face"
            scores["Prone"]   = 0.92
            scores["Supine"]  = 0.04
            scores["Lateral"] = 0.04
            evidence = 1.0
            return scores, evidence

        # ── HEAD-TILTED PRONE (nose hidden + at least one eye hidden) ──
        # Head is turned to the side but face is still down (into pillow).
        # One ear may be partially visible; shoulder asymmetry may be low.
        if nose_hidden and one_eye_hidden:
            detail["prone_trigger"] = "head_tilted_prone"
            scores["Prone"]   = 0.78
            scores["Lateral"] = 0.16
            scores["Supine"]  = 0.06
            evidence = 0.90
            return scores, evidence

        # ── PRONE WITH HEAD TURNED (nose hidden, both eyes visible) ──
        # Person is prone but has turned head fully to the side so both
        # eyes may be partially visible and shoulder asymmetry looks lateral.
        # Key signal: nose is hidden (face not pointing at camera) but
        # shoulder asymmetry is high (body rotated laterally at shoulder).
        # Without this gate, high shoulder asym pushes prone → lateral.
        ls_pt = _pt(kp, LEFT_SHOULDER)
        rs_pt = _pt(kp, RIGHT_SHOULDER)
        if nose_hidden and ls_pt is not None and rs_pt is not None:
            shld_w    = max(abs(ls_pt[0] - rs_pt[0]), 20.0)
            shld_asym = abs(ls_pt[1] - rs_pt[1]) / shld_w
            if shld_asym >= self.t.SHLD_ASYM_LATERAL:
                detail["prone_trigger"] = "prone_head_turned"
                scores["Prone"]   = 0.72
                scores["Lateral"] = 0.22
                scores["Supine"]  = 0.06
                evidence = 0.85
                return scores, evidence

        # Track whether nose+eye signals lean prone — used to gate
        # the shoulder signal so low shoulder asymmetry is not
        # misread as Supine when other signals say Prone.
        head_facing_down = nose_c <= self.t.NOSE_CONF_PRONE

        # ── SIGNAL 1: Nose confidence ─────────────────────────
        # Overhead IR: supine → nose points at lens (high conf)
        #              prone  → nose into mattress (near zero)
        #              lateral → nose sideways (medium conf)
        if nose_c >= self.t.NOSE_CONF_SUPINE:
            scores["Supine"]  += w * nose_c
            scores["Lateral"] += w * 0.20
            evidence += w
        elif nose_c <= self.t.NOSE_CONF_PRONE:
            strength           = 1.0 - (nose_c /
                                 max(self.t.NOSE_CONF_PRONE, 1e-6))
            scores["Prone"]   += w * strength
            scores["Lateral"] += w * 0.15
            evidence += w
        else:
            # mid-range → lateral
            scores["Lateral"] += w * 0.65
            scores["Supine"]  += w * 0.20
            evidence += w * 0.6

        detail["nose_signal"] = round(nose_c, 2)

        # ── SIGNAL 2: Eye visibility balance ──────────────────
        # Both eyes   → Supine  (face directly at camera)
        # One eye     → Lateral (one eye buried in pillow)
        # No eyes     → Prone   (face down)
        both_eyes = (le_c >= self.t.BOTH_EYES_CONF and
                     re_c >= self.t.BOTH_EYES_CONF)
        one_eye   = ((le_c >= self.t.ONE_EYE_CONF) !=
                     (re_c >= self.t.ONE_EYE_CONF))
        no_eyes   = (le_c < CONF_THR_WEAK and re_c < CONF_THR_WEAK)

        if both_eyes:
            scores["Supine"]  += w * 0.90
            scores["Lateral"] += w * 0.08
            evidence += w
        elif one_eye:
            scores["Lateral"] += w * 0.88
            scores["Prone"]   += w * 0.08
            evidence += w
        elif no_eyes:
            scores["Prone"]   += w * 0.75
            scores["Lateral"] += w * 0.18
            evidence += w * 0.8

        detail["eye_state"] = ("both" if both_eyes else
                               "one"  if one_eye   else
                               "none" if no_eyes   else "unclear")

        # ── SIGNAL 3: Ear Y-asymmetry ─────────────────────────
        le_pt = _pt(kp, LEFT_EAR,  CONF_THR_WEAK)
        re_pt = _pt(kp, RIGHT_EAR, CONF_THR_WEAK)

        if le_pt is not None and re_pt is not None:
            inter_ear_x = max(abs(le_pt[0] - re_pt[0]), 20.0)
            ear_asym    = abs(le_pt[1] - re_pt[1]) / inter_ear_x
            detail["ear_asym"] = round(ear_asym, 3)
            evidence += w

            # special prone rule: ears roughly level + nose hidden
            # but head may be tilted so PRONE_EAR_ASYM_MAX is raised
            prone_ear_pattern = (
                ear_asym <= self.t.PRONE_EAR_ASYM_MAX and
                nose_c   <= self.t.NOSE_CONF_PRONE
            )

            if prone_ear_pattern:
                scores["Prone"]   += w * 0.82
                scores["Lateral"] += w * 0.10
            elif ear_asym >= self.t.EAR_ASYM_PRONE:
                scores["Prone"]   += w * 0.50
                scores["Lateral"] += w * 0.45
            elif ear_asym >= self.t.EAR_ASYM_LATERAL:
                scores["Lateral"] += w * 0.88
                scores["Supine"]  += w * 0.08
            else:
                scores["Supine"]  += w * 0.82
                scores["Lateral"] += w * 0.12

        elif le_pt is not None or re_pt is not None:
            # One ear visible — if head_facing_down this is tilted prone
            if head_facing_down:
                scores["Prone"]   += w * 0.65
                scores["Lateral"] += w * 0.25
            else:
                scores["Lateral"] += w * 0.78
                scores["Prone"]   += w * 0.12
            evidence += w * 0.65
            detail["ear_asym"] = "one_ear_only"
        else:
            scores["Prone"]   += w * 0.60
            scores["Lateral"] += w * 0.25
            evidence += w * 0.45
            detail["ear_asym"] = "no_ears"

        # ── SIGNAL 4: Shoulder Y-asymmetry ────────────────────
        # GATE: if nose/eye signals already point to Prone, low shoulder
        # asymmetry should NOT be credited to Supine — prone shoulders
        # can appear level too.
        ls = _pt(kp, LEFT_SHOULDER)
        rs = _pt(kp, RIGHT_SHOULDER)

        if ls is not None and rs is not None:
            shld_width = max(abs(ls[0] - rs[0]), 20.0)
            shld_asym  = abs(ls[1] - rs[1]) / shld_width
            detail["shld_asym"] = round(shld_asym, 3)
            evidence += w

            if shld_asym >= self.t.SHLD_ASYM_LATERAL:
                scores["Lateral"] += w * 0.90
                scores["Supine"]  += w * 0.05
            else:
                # low shoulder asymmetry: only credit Supine when
                # nose/eye do NOT suggest prone face
                if head_facing_down or no_eyes:
                    # prone shoulders can look level — don't reward Supine
                    scores["Prone"]   += w * 0.55
                    scores["Supine"]  += w * 0.25
                    scores["Lateral"] += w * 0.10
                else:
                    scores["Supine"]  += w * 0.55
                    scores["Prone"]   += w * 0.35
                    scores["Lateral"] += w * 0.05

        elif ls is not None or rs is not None:
            evidence += w * 0.4
            detail["shld_asym"] = "one_shoulder_only"
        else:
            detail["shld_asym"] = "no_shoulders"

        # ── SIGNAL 5: Calibration ratio signals (bonus) ────────
        # Only active after 150-frame supine baseline is collected.
        # Uses personal shoulder/eye/ear widths so thresholds adapt
        # to each person's body size and camera distance.
        if calib:
            shld_r = calib.get("shld_ratio")
            eye_r  = calib.get("eye_ratio")
            ear_r  = calib.get("ear_ratio")

            # shoulder width collapsed → lateral or prone
            if shld_r is not None:
                if shld_r < self.t.SHLD_LATERAL_RATIO:
                    # shoulders very close together → strong lateral
                    scores["Lateral"] += w * 0.80
                    scores["Prone"]   += w * 0.15
                    scores["Supine"]  -= w * 0.10
                    evidence += w * 0.5
                    detail["calib_shld"] = f"lateral ({shld_r:.2f})"
                elif shld_r < self.t.SHLD_PRONE_RATIO:
                    # shoulders moderately collapsed → prone hint
                    scores["Prone"]   += w * 0.45
                    scores["Lateral"] += w * 0.35
                    evidence += w * 0.3
                    detail["calib_shld"] = f"prone_hint ({shld_r:.2f})"
                else:
                    # shoulders wide → supine signal
                    scores["Supine"]  += w * 0.40
                    evidence += w * 0.2
                    detail["calib_shld"] = f"supine ({shld_r:.2f})"

            # eye distance collapsed → lateral (one eye buried)
            if eye_r is not None and eye_r < self.t.EYE_LATERAL_RATIO:
                scores["Lateral"] += w * 0.50
                scores["Supine"]  -= w * 0.05
                evidence += w * 0.2
                detail["calib_eye"] = f"lateral ({eye_r:.2f})"

            # ear distance collapsed → lateral (one ear buried)
            if ear_r is not None and ear_r < self.t.EAR_LATERAL_RATIO:
                scores["Lateral"] += w * 0.50
                scores["Supine"]  -= w * 0.05
                evidence += w * 0.2
                detail["calib_ear"] = f"lateral ({ear_r:.2f})"

        # FIX: clamp all scores to >= 0 — calibration penalties can push
        # scores negative which breaks max() selection and confidence calc
        scores = {k: max(0.0, v) for k, v in scores.items()}

        return scores, evidence

    # ----------------------------------------------------------
    # CALIBRATION
    # ----------------------------------------------------------
    def _collect_calibration(self, kp: np.ndarray):
        """
        Called on early supine frames (high nose confidence).
        Records shoulder width, eye distance, ear distance.
        After CALIB_FRAMES valid samples, computes median baselines.

        Fallback: if no valid supine frames are seen within the first
        CALIB_FRAMES * 3 frames (e.g. session starts lateral/prone),
        calibration is skipped and fixed thresholds remain active.
        """
        if self._calib_done:
            return

        self._calib_total_seen += 1

        nose_c = _conf(kp, NOSE)

        # Fallback: session never started supine — give up after 3x window
        # so ratio signals simply stay disabled (fixed thresholds used instead)
        if self._calib_total_seen >= self.t.CALIB_FRAMES * 3 and self._calib_frames == 0:
            self._calib_done = True   # stop trying; calibrated stays False
            return

        if nose_c < self.t.CALIB_NOSE_MIN:
            return   # not a clean supine frame, skip

        ls = _pt(kp, LEFT_SHOULDER)
        rs = _pt(kp, RIGHT_SHOULDER)
        le = _pt(kp, LEFT_EYE)
        re = _pt(kp, RIGHT_EYE)
        le_ear = _pt(kp, LEFT_EAR,  CONF_THR_WEAK)
        re_ear = _pt(kp, RIGHT_EAR, CONF_THR_WEAK)

        if ls is not None and rs is not None:
            self._calib_shld_widths.append(abs(ls[0] - rs[0]))
        if le is not None and re is not None:
            self._calib_eye_dists.append(abs(le[0] - re[0]))
        if le_ear is not None and re_ear is not None:
            self._calib_ear_dists.append(abs(le_ear[0] - re_ear[0]))

        self._calib_frames += 1

        if self._calib_frames >= self.t.CALIB_FRAMES:
            if self._calib_shld_widths:
                self.baseline_shld_width = float(np.median(self._calib_shld_widths))
            if self._calib_eye_dists:
                self.baseline_eye_dist   = float(np.median(self._calib_eye_dists))
            if self._calib_ear_dists:
                self.baseline_ear_dist   = float(np.median(self._calib_ear_dists))
            self._calib_done = True
            self.calibrated  = True

    def _calibration_signals(self, kp: np.ndarray, detail: dict) -> dict:
        """
        Returns ratio-based calibration signals once baseline is known.
        Returns empty dict if not yet calibrated.

        Signals:
            shld_ratio  — current shoulder width / baseline
                          < SHLD_LATERAL_RATIO → strong lateral
                          < SHLD_PRONE_RATIO   → prone hint
            eye_ratio   — current eye distance / baseline
                          < EYE_LATERAL_RATIO  → lateral hint
            ear_ratio   — current ear distance / baseline
                          < EAR_LATERAL_RATIO  → lateral hint
        """
        if not self.calibrated:
            return {}

        sigs = {}

        ls = _pt(kp, LEFT_SHOULDER)
        rs = _pt(kp, RIGHT_SHOULDER)
        if ls is not None and rs is not None and self.baseline_shld_width > 0:
            curr_w = abs(ls[0] - rs[0])
            sigs["shld_ratio"] = round(curr_w / self.baseline_shld_width, 3)

        le = _pt(kp, LEFT_EYE)
        re = _pt(kp, RIGHT_EYE)
        if le is not None and re is not None and self.baseline_eye_dist > 0:
            curr_e = abs(le[0] - re[0])
            sigs["eye_ratio"]  = round(curr_e / self.baseline_eye_dist, 3)

        le_ear = _pt(kp, LEFT_EAR,  CONF_THR_WEAK)
        re_ear = _pt(kp, RIGHT_EAR, CONF_THR_WEAK)
        if le_ear is not None and re_ear is not None and self.baseline_ear_dist > 0:
            curr_a = abs(le_ear[0] - re_ear[0])
            sigs["ear_ratio"]  = round(curr_a / self.baseline_ear_dist, 3)

        detail["calib_sigs"] = sigs
        return sigs

    # ----------------------------------------------------------
    # LATERAL SIDE RESOLUTION
    # ----------------------------------------------------------
    def _lateral_side(self, kp: np.ndarray, detail: dict) -> str:
        """
        Determine Left or Right lateral from which side is
        pressed down (higher Y = closer to mattress in overhead view).

        Uses ear Y position as primary, shoulder Y as backup.
        Returns "Lateral_Left" or "Lateral_Right".

        Lateral_Left  = lying on LEFT side  → left ear/shoulder lower
        Lateral_Right = lying on RIGHT side → right ear/shoulder lower
        """
        votes = {"left": 0, "right": 0}

        # ear vote — pressed ear has higher Y (lower in image)
        # Camera is overhead: YOLO's left/right is from the person's perspective
        # mirrored in image — so left ear higher Y → person lying on RIGHT side
        le_pt = _pt(kp, LEFT_EAR,  CONF_THR_WEAK)
        re_pt = _pt(kp, RIGHT_EAR, CONF_THR_WEAK)
        if le_pt is not None and re_pt is not None:
            if le_pt[1] > re_pt[1]:
                votes["right"] += 2   # left ear lower in image → right side down
            else:
                votes["left"]  += 2   # right ear lower in image → left side down

        # shoulder vote — pressed shoulder has higher Y
        # same mirror correction as ears
        ls = _pt(kp, LEFT_SHOULDER)
        rs = _pt(kp, RIGHT_SHOULDER)
        if ls is not None and rs is not None:
            if ls[1] > rs[1]:
                votes["right"] += 2   # left shoulder lower in image → right side down
            else:
                votes["left"]  += 2   # right shoulder lower in image → left side down

        # eye vote — hidden eye is pressed to mattress
        # same mirror correction
        le_c = _conf(kp, LEFT_EYE)
        re_c = _conf(kp, RIGHT_EYE)
        if abs(le_c - re_c) > 0.15:
            if le_c < re_c:
                votes["right"] += 1   # left eye hidden → right side down
            else:
                votes["left"]  += 1   # right eye hidden → left side down

        detail["lateral_votes"] = votes
        side = "Left" if votes["left"] >= votes["right"] else "Right"
        detail["lateral_side"]  = side
        return f"Lateral_{side}"


# =========================================================
# PER-PERSON TRACKER
# =========================================================
class PersonPostureTracker:
    """
    Stateful wrapper around PostureClassifier for one person.

    Call update() every frame inside the step loop.
    Includes a stability gate — posture must hold STABLE_FRAMES
    consecutive frames before being committed to the timeline.

    update(keypoints, timestamp) → PostureResult
    current_posture()            → PostureResult
    get_timeline()               → list of (timestamp, posture, confidence)
    posture_durations()          → dict {posture: total_seconds}
    reset()
    """

    def __init__(self, person_idx: int = 0,
                 thresholds: Thresholds = None,
                 smooth: bool = True):
        self.person_idx = person_idx
        self.smooth     = smooth
        self._clf       = PostureClassifier(thresholds)
        self._t         = thresholds or Thresholds()
        self._timeline  : list = []
        self._current   : PostureResult = PostureResult()
        self._candidate = "Unknown"
        self._streak    = 0

    def update(self, keypoints: np.ndarray,
               timestamp: float) -> PostureResult:
        result = (self._clf.classify_smoothed(keypoints)
                  if self.smooth
                  else self._clf.classify_frame(keypoints))

        # stability gate
        if result.posture == self._candidate:
            self._streak += 1
        else:
            self._candidate = result.posture
            self._streak    = 1

        committed = (result.posture
                     if self._streak >= self._t.STABLE_FRAMES
                     else (self._timeline[-1][1]
                           if self._timeline else result.posture))

        stable = PostureResult(committed, result.confidence,
                               result.method, result.detail)
        self._current = stable

        if not self._timeline or self._timeline[-1][1] != committed:
            self._timeline.append(
                (timestamp, committed, result.confidence)
            )

        return stable

    def update_no_person(self, timestamp: float) -> PostureResult:
        """
        Call this when no person is detected in the frame.
        Resets the stability streak and returns a No_Person result
        immediately — no smoothing applied.
        """
        result = PostureResult("No_Person", 1.0, "no_detection", {})

        # Reset streak so the next real detection starts fresh
        self._candidate = "No_Person"
        self._streak    = self._t.STABLE_FRAMES   # commit immediately

        self._current = result

        if not self._timeline or self._timeline[-1][1] != "No_Person":
            self._timeline.append((timestamp, "No_Person", 1.0))

        return result

    def current_posture(self) -> PostureResult:
        return self._current

    def get_timeline(self) -> list:
        """List of (timestamp, posture, confidence) for the session."""
        return list(self._timeline)

    def posture_durations(self, session_end_ts: float = None) -> dict:
        """
        Total seconds spent in each posture class.

        Pass session_end_ts (final frame timestamp) so the last posture
        segment gets its correct duration instead of zero.
        Without it, the last segment is excluded from the total.
        """
        durations = {}
        tl = self._timeline
        for i, (ts, posture, _) in enumerate(tl):
            if i + 1 < len(tl):
                end_ts = tl[i + 1][0]
            elif session_end_ts is not None:
                end_ts = session_end_ts
            else:
                continue   # skip last segment — end time unknown
            durations[posture] = (durations.get(posture, 0.0)
                                  + (end_ts - ts))
        return durations

    @property
    def calibrated(self) -> bool:
        """True once the 150-frame supine baseline has been collected."""
        return self._clf.calibrated

    @property
    def baseline(self) -> dict:
        """Returns the calibrated baseline measurements for this person."""
        return {
            "shoulder_width_px" : round(self._clf.baseline_shld_width, 1),
            "eye_distance_px"   : round(self._clf.baseline_eye_dist,   1),
            "ear_distance_px"   : round(self._clf.baseline_ear_dist,   1),
            "calib_frames"      : self._clf._calib_frames,
        }

    def reset(self):
        self._clf.reset()
        self._timeline.clear()
        self._current   = PostureResult()
        self._candidate = "Unknown"
        self._streak    = 0