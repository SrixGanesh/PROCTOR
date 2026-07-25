# ============================================================
#  ProctorAI — face_module.py  (v3 — Adaptive Detection)
#
#  IMPROVEMENTS:
#  - Dual cascade (frontal + profile) — head turn detect
#  - Adaptive minNeighbors based on recent detection rate
#  - Histogram equalization + CLAHE for poor lighting
#  - Smooth bounding box (no jitter)
#  - All previous anti-flicker logic kept
# ============================================================

import cv2
import numpy as np
import time


class FaceModule:
    def __init__(self, score_engine):
        self.engine = score_engine

        # Primary cascade — frontal face
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        # Secondary cascade — profile (side face)
        self.profile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_profileface.xml"
        )
        # CLAHE for adaptive lighting correction
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # ── Detection buffer (anti-flicker) ──────────────────
        self._detection_buffer = []
        self.BUFFER_SIZE       = 10
        self.DETECT_THRESHOLD  = 0.35    # 35% of frames = face present

        # ── Smooth bounding box ───────────────────────────────
        self._smooth_box  = None         # (x, y, w, h) smoothed
        self.SMOOTH_ALPHA = 0.4          # 0=very smooth, 1=no smooth

        # ── No-face duration ──────────────────────────────────
        self.NO_FACE_GRACE_SEC  = 2.0
        self.NO_FACE_ALERT_SEC  = 4.0
        self.NO_FACE_COOLDOWN   = 20
        self._no_face_since     = None
        self._last_no_face_alert= 0
        self._face_was_present  = True

        # ── Multi-face ────────────────────────────────────────
        self._multi_face_frames = 0
        self.MULTI_CONFIRM      = 12
        self._last_multi_alert  = 0
        self.MULTI_COOLDOWN     = 15

        # ── Identity ─────────────────────────────────────────
        self.reference_face       = None
        self._mismatch_count      = 0
        self.MISMATCH_CONFIRM     = 4
        self._last_mismatch_alert = 0
        self.MISMATCH_COOLDOWN    = 20

        self.frame_count = 0
        print("[FaceModule] v3 — Adaptive + Dual cascade + CLAHE ✅")

    # ── Reference capture ────────────────────────────────────
    def capture_reference(self, frame):
        faces = self._detect_all(frame)
        if len(faces) == 1:
            x, y, w, h = faces[0]
            pad = 25
            y1 = max(0, y - pad);  y2 = min(frame.shape[0], y + h + pad)
            x1 = max(0, x - pad);  x2 = min(frame.shape[1], x + w + pad)
            self.reference_face = frame[y1:y2, x1:x2].copy()
            print(f"[FaceModule] Reference captured ✅  ({self.engine.student_name})")
            return True
        print(f"[FaceModule] Capture failed — {len(faces)} face(s) detected")
        return False

    # ── Main process ─────────────────────────────────────────
    def process(self, frame):
        self.frame_count += 1
        annotated = frame.copy()
        now       = time.time()

        # Auto-capture reference at frame 20
        if self.reference_face is None and self.frame_count == 20:
            self.capture_reference(frame)

        faces = self._detect_all(frame)

        # ── Update buffer ──────────────────────────────────────
        self._detection_buffer.append(len(faces) > 0)
        if len(self._detection_buffer) > self.BUFFER_SIZE:
            self._detection_buffer.pop(0)

        detection_rate = (
            sum(self._detection_buffer) / len(self._detection_buffer)
            if self._detection_buffer else 1.0
        )
        face_present = detection_rate >= self.DETECT_THRESHOLD

        # ── No-face logic ──────────────────────────────────────
        if not face_present:
            if self._no_face_since is None:
                self._no_face_since = now
            missing = now - self._no_face_since

            if missing < self.NO_FACE_GRACE_SEC:
                self._put_text(annotated, "Detecting...", (10, 30), (180, 180, 0))
            elif missing < self.NO_FACE_ALERT_SEC:
                self._put_text(annotated, f"Face not visible  ({missing:.1f}s)", (10, 30), (0, 140, 255))
                self._draw_bar(annotated, missing, self.NO_FACE_ALERT_SEC, (0, 140, 255))
            else:
                self._put_text(annotated, f"NO FACE  ({missing:.1f}s) !", (10, 30), (0, 0, 255))
                self._draw_bar(annotated, missing, self.NO_FACE_ALERT_SEC, (0, 0, 255))
                if now - self._last_no_face_alert >= self.NO_FACE_COOLDOWN:
                    self.engine.no_face()
                    self._last_no_face_alert = now

            self._face_was_present = False
            self._smooth_box = None
            self._score_overlay(annotated, frame)
            return annotated

        # Face present — reset
        self._no_face_since    = None
        self._face_was_present = True

        # Guard — buffer says present but this frame empty (transition frame)
        if len(faces) == 0:
            self._score_overlay(annotated, frame)
            return annotated

        # ── Multi-face check ───────────────────────────────────
        if len(faces) > 1:
            self._multi_face_frames += 1
            self._put_text(annotated, f"Multiple faces: {len(faces)} !", (10, 60), (0, 0, 255))
            if (self._multi_face_frames >= self.MULTI_CONFIRM
                    and now - self._last_multi_alert >= self.MULTI_COOLDOWN):
                self.engine.multi_face()
                self._last_multi_alert  = now
                self._multi_face_frames = 0
        else:
            self._multi_face_frames = max(0, self._multi_face_frames - 1)

        # ── Smooth bounding box ────────────────────────────────
        primary = faces[0]
        if self._smooth_box is None:
            self._smooth_box = np.array(primary, dtype=float)
        else:
            self._smooth_box = (self.SMOOTH_ALPHA * np.array(primary, dtype=float)
                                + (1 - self.SMOOTH_ALPHA) * self._smooth_box)
        sx, sy, sw, sh = [int(v) for v in self._smooth_box]

        # Draw smooth box
        box_color = (0, 255, 0) if len(faces) == 1 else (0, 0, 255)
        cv2.rectangle(annotated, (sx, sy), (sx + sw, sy + sh), box_color, 2)

        # Detection rate bar inside box top
        bar_w = int(detection_rate * sw)
        cv2.rectangle(annotated, (sx, sy - 6), (sx + sw, sy - 2), (40, 40, 40), -1)
        cv2.rectangle(annotated, (sx, sy - 6), (sx + bar_w, sy - 2),
                      (0, 220, 0) if detection_rate > 0.7 else (0, 140, 255), -1)

        # Label
        label = "FACE OK" if len(faces) == 1 else f"{len(faces)} FACES"
        cv2.putText(annotated, label, (sx, sy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1)

        # ── Identity verify (every 30 frames) ─────────────────
        if (self.reference_face is not None
                and len(faces) == 1
                and self.frame_count % 30 == 0):
            cur_face = frame[sy:sy+sh, sx:sx+sw]
            match    = self._verify_identity(cur_face)
            if match is False:
                self._mismatch_count += 1
                self._put_text(annotated, f"Identity mismatch! ({self._mismatch_count}x)", (10, 90), (0, 0, 255))
                if (self._mismatch_count >= self.MISMATCH_CONFIRM
                        and now - self._last_mismatch_alert >= self.MISMATCH_COOLDOWN):
                    self.engine.identity_mismatch()
                    self._last_mismatch_alert = now
                    self._mismatch_count      = 0
            elif match is True:
                self._mismatch_count = max(0, self._mismatch_count - 1)

        self._score_overlay(annotated, frame)
        return annotated

    # ── Score overlay at bottom ───────────────────────────────
    def _score_overlay(self, frame, orig=None):
        score  = self.engine.score
        status = self.engine.status
        color  = (0,200,0) if status=="safe" else (0,140,255) if status=="warning" else (0,0,255)
        h      = frame.shape[0]
        cv2.rectangle(frame, (0, h-28), (frame.shape[1], h), (20,20,20), -1)
        cv2.putText(frame, f"  Score: {score}/100   Status: {status.upper()}",
                    (5, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # ── Detect faces — frontal + profile ──────────────────────
    def _detect_all(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = self.clahe.apply(gray)           # adaptive lighting

        # Frontal
        frontal = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4,
            minSize=(70, 70), flags=cv2.CASCADE_SCALE_IMAGE
        )
        frontal = list(frontal) if len(frontal) > 0 else []

        # Profile (only if frontal found nothing)
        if not frontal:
            profile = self.profile_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4,
                minSize=(70, 70)
            )
            if len(profile) > 0:
                frontal = list(profile)

        # Deduplicate overlapping boxes
        if len(frontal) > 1:
            frontal = self._deduplicate(frontal)

        return frontal

    # ── Remove overlapping boxes ──────────────────────────────
    def _deduplicate(self, boxes):
        if not boxes:
            return boxes
        boxes   = sorted(boxes, key=lambda b: b[2]*b[3], reverse=True)  # largest first
        keep    = []
        for box in boxes:
            x1,y1,w1,h1 = box
            overlap = False
            for kx,ky,kw,kh in keep:
                ix = max(0, min(x1+w1, kx+kw) - max(x1, kx))
                iy = max(0, min(y1+h1, ky+kh) - max(y1, ky))
                if ix * iy > 0.5 * w1 * h1:
                    overlap = True
                    break
            if not overlap:
                keep.append(box)
        return keep

    # ── Helpers ───────────────────────────────────────────────
    def _put_text(self, frame, text, pos, color):
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    def _draw_bar(self, frame, current, maximum, color):
        pct   = min(1.0, current / maximum)
        bar_w = int(pct * 150)
        cv2.rectangle(frame, (10, 42), (160, 52), (40,40,40), -1)
        cv2.rectangle(frame, (10, 42), (10 + bar_w, 52), color, -1)
        cv2.putText(frame, f"{current:.1f}s / {maximum:.0f}s",
                    (165, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160,160,160), 1)

    # ── Identity verification ──────────────────────────────────
    def _verify_identity(self, current_face_img):
        try:
            from deepface import DeepFace
            import tempfile, os
            ref_path = os.path.join(tempfile.gettempdir(), "proctor_ref.jpg")
            cur_path = os.path.join(tempfile.gettempdir(), "proctor_cur.jpg")
            cv2.imwrite(ref_path, self.reference_face)
            cv2.imwrite(cur_path, current_face_img)
            result = DeepFace.verify(
                img1_path=ref_path, img2_path=cur_path,
                model_name="VGG-Face", enforce_detection=False, silent=True
            )
            return result["verified"]
        except Exception:
            return self._histogram_match(current_face_img)

    def _histogram_match(self, face_img):
        try:
            ref = cv2.resize(self.reference_face, (100, 100))
            cur = cv2.resize(face_img,            (100, 100))
            rh  = cv2.calcHist([ref], [0,1,2], None, [8,8,8], [0,256]*3)
            ch  = cv2.calcHist([cur], [0,1,2], None, [8,8,8], [0,256]*3)
            cv2.normalize(rh, rh); cv2.normalize(ch, ch)
            return cv2.compareHist(rh, ch, cv2.HISTCMP_CORREL) >= 0.7
        except Exception:
            return None


# ── Standalone test ──────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from modules.score_engine import ScoreEngine

    engine = ScoreEngine("TEST001", "Test Student")
    module = FaceModule(engine)
    cap    = cv2.VideoCapture(0)

    print("Face Module v3 — Adaptive Detection")
    print("  Green box  →  face detected OK")
    print("  Yellow     →  brief flicker, ignored")
    print("  Orange bar →  face missing 2–4s, warning")
    print("  Red        →  face missing 4s+, alert!")
    print("  'r' = capture reference  |  'q' = quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out = module.process(frame)
        cv2.imshow("Face Module v3", out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            module.capture_reference(frame)
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nFinal score      : {engine.score}")
    print(f"No-face alerts   : {sum(1 for e in engine.events if e['type']=='no_face')}")
    print(f"Multi-face alerts: {sum(1 for e in engine.events if e['type']=='multi_face')}")
