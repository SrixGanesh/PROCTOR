# ============================================================
#  ProctorAI — eye_tracker.py  (v6 — Eyes Only, Simple)
#
#  LOGIC (simple):
#  - MediaPipe iris landmarks track where eyes are looking
#  - Eyes looking at screen center = OK
#  - Eyes drift to side for 3+ seconds = FLAG
#  - No head pose, no calibration, no complexity
# ============================================================

import cv2
import mediapipe as mp
import numpy as np
import time
from collections import Counter


class EyeTracker:
    def __init__(self, score_engine):
        self.engine = score_engine

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh    = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        # ── Iris zone thresholds ──────────────────────────────
        # Iris ratio: 0.0 = extreme left, 0.5 = center, 1.0 = extreme right
        # Center zone is wide — only extreme side glances flagged
        self.CENTER_MIN = 0.28   # horizontal: loosened slightly
        self.CENTER_MAX = 0.72   # horizontal: loosened slightly

        # Vertical iris thresholds (iris Y position within eye)
        # 0.0 = iris at top of eye, 1.0 = iris at bottom
        self.V_CENTER_MIN = 0.20  # UP not used — kept low
        self.V_CENTER_MAX = 0.60  # DOWN: more sensitive (was 0.75)

        # ── Timing ────────────────────────────────────────────
        self.FLAG_AFTER_SEC   = 4.0   # flag after 4s continuous (was 3s)
        self.ALERT_COOLDOWN   = 40    # seconds between alerts (was 30s)
        self.FREQ_THRESHOLD   = 3     # flag after 3 look-away events
        self.FREQ_COOLDOWN    = 60

        # ── State ─────────────────────────────────────────────
        self._away_since       = None
        self._last_alert_time  = 0
        self._currently_away   = False
        self._away_event_count = 0
        self._last_freq_alert  = 0

        # Smooth iris ratio over last 8 frames — removes flicker
        self._ratio_history = []
        self.SMOOTH_N       = 10     # smoother — less flicker (was 8)

        print("[EyeTracker] v6.5 — Count=3, DOWN sensitive=0.60 ✅")
        print(f"  Tracks: LEFT RIGHT DOWN  |  UP ignored")
        print(f"  Flag after: {self.FLAG_AFTER_SEC}s  |  Cooldown: {self.ALERT_COOLDOWN}s")

    # ── Main process ─────────────────────────────────────────
    def process(self, frame):
        h, w      = frame.shape[:2]
        rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result    = self.face_mesh.process(rgb)
        annotated = frame.copy()
        now       = time.time()

        if not result.multi_face_landmarks:
            cv2.putText(annotated, "No face", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100,100,100), 1)
            return annotated

        lm = result.multi_face_landmarks[0].landmark

        # ── Get iris ratios ───────────────────────────────────
        h_ratio, v_ratio = self._get_iris_ratio(lm, w, h)
        if h_ratio is None:
            return annotated

        # Smooth over last N frames (store both H and V)
        self._ratio_history.append((h_ratio, v_ratio))
        if len(self._ratio_history) > self.SMOOTH_N:
            self._ratio_history.pop(0)
        smooth_h = float(np.mean([r[0] for r in self._ratio_history]))
        smooth_v = float(np.mean([r[1] for r in self._ratio_history]))

        # ── Classify direction ────────────────────────────────
        if smooth_h < self.CENTER_MIN:
            direction = "RIGHT"
        elif smooth_h > self.CENTER_MAX:
            direction = "LEFT"
        elif smooth_v > self.V_CENTER_MAX:
            direction = "DOWN"   # only down — UP ignored (camera position)
        else:
            direction = "CENTER"

        looking_away = direction != "CENTER"

        # ── Duration logic ────────────────────────────────────
        if looking_away:
            if not self._currently_away:
                # New look-away event just started
                self._currently_away    = True
                self._away_since        = now
                self._away_event_count += 1  # count this new event

                # Check frequency threshold on NEW event
                if self._away_event_count >= self.FREQ_THRESHOLD:
                    self.engine.add_event(
                        "gaze_away",
                        f"Looked away {self._away_event_count} times",
                        10
                    )
                    self._away_event_count = 0   # reset immediately after alert

            # Sustained look-away — 4 seconds
            away_sec = now - self._away_since
            if (away_sec >= self.FLAG_AFTER_SEC
                    and now - self._last_alert_time >= self.ALERT_COOLDOWN):
                self.engine.gaze_away()
                self._last_alert_time = now
                self._away_since      = now   # reset timer for next 4s

        else:
            # Eyes back to screen
            self._currently_away = False
            self._away_since     = None

        # ── Draw ──────────────────────────────────────────────
        self._draw(annotated, direction, smooth_h, smooth_v, now, lm, w, h)
        return annotated

    # ── Get iris ratios — horizontal + vertical ──────────────
    def _get_iris_ratio(self, lm, w, h):
        try:
            # ── Horizontal (left/right) ───────────────────────
            l_iris_x  = np.mean([lm[i].x for i in [474,475,476,477]]) * w
            r_iris_x  = np.mean([lm[i].x for i in [469,470,471,472]]) * w
            l_h = self._ratio(l_iris_x, lm[33].x*w,  lm[133].x*w)
            r_h = self._ratio(r_iris_x, lm[362].x*w, lm[263].x*w)
            h_ratio = (l_h + r_h) / 2

            # ── Vertical (up/down) ────────────────────────────
            # Eye top landmark: 159 (left), 386 (right)
            # Eye bottom landmark: 145 (left), 374 (right)
            l_iris_y  = np.mean([lm[i].y for i in [474,475,476,477]]) * h
            r_iris_y  = np.mean([lm[i].y for i in [469,470,471,472]]) * h

            # Left eye vertical span
            l_top_y    = lm[159].y * h
            l_bot_y    = lm[145].y * h
            # Right eye vertical span
            r_top_y    = lm[386].y * h
            r_bot_y    = lm[374].y * h

            l_v = self._ratio(l_iris_y, l_top_y, l_bot_y)
            r_v = self._ratio(r_iris_y, r_top_y, r_bot_y)
            v_ratio = (l_v + r_v) / 2

            return h_ratio, v_ratio
        except Exception:
            return None, None

    def _ratio(self, iris, start, end):
        span = abs(end - start)
        return (iris - start) / span if span > 2 else 0.5

    # ── Draw UI ───────────────────────────────────────────────
    def _draw(self, frame, direction, h_ratio, v_ratio, now, lm, w, h):
        away_sec = (now - self._away_since) if self._away_since else 0
        fh, fw   = frame.shape[:2]

        # Status label + color
        if direction == "CENTER":
            color = (0, 220, 0)
            label = "Eyes: SCREEN  ok"
        elif away_sec < self.FLAG_AFTER_SEC:
            pct   = away_sec / self.FLAG_AFTER_SEC
            color = (0, int(220 - pct*80), int(220 - pct*220))
            label = f"Eyes: {direction}  ({away_sec:.1f}s / {self.FLAG_AFTER_SEC:.0f}s)"
        else:
            color = (0, 0, 255)
            label = f"Eyes: {direction}  FLAGGED! ({away_sec:.1f}s)"

        cv2.putText(frame, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Progress bar — fills toward flag threshold
        if self._away_since:
            pct   = min(1.0, away_sec / self.FLAG_AFTER_SEC)
            bar_w = int(pct * 200)
            clr   = (0,200,200) if pct < 0.5 else (0,140,255) if pct < 1.0 else (0,0,255)
            cv2.rectangle(frame, (10, 42), (210, 54), (40,40,40), -1)
            cv2.rectangle(frame, (10, 42), (10+bar_w, 54), clr, -1)

        # Event counter
        cnt_c = (0,80,255) if self._away_event_count >= self.FREQ_THRESHOLD-2 else (110,110,110)
        cv2.putText(frame, f"Look-away count: {self._away_event_count}/{self.FREQ_THRESHOLD}",
                    (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.45, cnt_c, 1)

        # Iris ratio gauge — horizontal bar showing eye position
        gauge_x, gauge_y, gauge_w, gauge_h = 10, 80, 200, 14
        cv2.rectangle(frame, (gauge_x, gauge_y),
                      (gauge_x+gauge_w, gauge_y+gauge_h), (40,40,40), -1)
        # Safe zone (center band)
        safe_x1 = gauge_x + int(self.CENTER_MIN * gauge_w)
        safe_x2 = gauge_x + int(self.CENTER_MAX * gauge_w)
        cv2.rectangle(frame, (safe_x1, gauge_y), (safe_x2, gauge_y+gauge_h), (0,60,0), -1)
        # Iris position marker
        iris_x = gauge_x + int(h_ratio * gauge_w)
        iris_c = (0,220,0) if direction=="CENTER" else (0,0,255)
        cv2.rectangle(frame, (iris_x-3, gauge_y-2),
                      (iris_x+3, gauge_y+gauge_h+2), iris_c, -1)
        cv2.putText(frame, "L", (gauge_x-12, gauge_y+11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80,80,80), 1)
        cv2.putText(frame, "R", (gauge_x+gauge_w+4, gauge_y+11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80,80,80), 1)

        # Vertical gauge (up/down)
        vg_x, vg_y, vg_w, vg_h = 220, 80, 14, 60
        cv2.rectangle(frame, (vg_x, vg_y), (vg_x+vg_w, vg_y+vg_h), (40,40,40), -1)
        # Safe zone
        vs1 = vg_y + int(self.V_CENTER_MIN * vg_h)
        vs2 = vg_y + int(self.V_CENTER_MAX * vg_h)
        cv2.rectangle(frame, (vg_x, vs1), (vg_x+vg_w, vs2), (0,60,0), -1)
        # Iris marker
        vy    = vg_y + int(v_ratio * vg_h)
        vy    = max(vg_y+3, min(vg_y+vg_h-3, vy))
        v_dot = (0,220,0) if direction=="CENTER" else (0,0,255)
        cv2.rectangle(frame, (vg_x-2, vy-3), (vg_x+vg_w+2, vy+3), v_dot, -1)
        cv2.putText(frame, "U", (vg_x+2, vg_y-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80,80,80), 1)
        cv2.putText(frame, "D", (vg_x+2, vg_y+vg_h+12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80,80,80), 1)

        # Draw iris dots on face
        try:
            for idx in [474, 469]:
                ix = int(lm[idx].x * w)
                iy = int(lm[idx].y * h)
                cv2.circle(frame, (ix, iy), 4, iris_c, -1)
                cv2.circle(frame, (ix, iy), 6, iris_c, 1)
        except Exception:
            pass

        # Score bar at bottom
        score  = self.engine.score
        status = self.engine.status
        sc     = (0,200,0) if status=="safe" else (0,140,255) if status=="warning" else (0,0,255)
        cv2.rectangle(frame, (0, fh-24), (fw, fh), (20,20,20), -1)
        cv2.putText(frame, f"  Score: {score}/100  |  {status.upper()}",
                    (5, fh-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, sc, 2)

    def close(self):
        self.face_mesh.close()


# ── Standalone test ──────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from modules.score_engine import ScoreEngine

    engine  = ScoreEngine("TEST001", "Test Student")
    tracker = EyeTracker(engine)
    cap     = cv2.VideoCapture(0)

    print("\nEye Tracker v6 — Simple Eyes Only")
    print("  Screen paaru      → Eyes: SCREEN ok  (green)")
    print("  Side paaru < 3s   → timer shows, no flag")
    print("  Side paaru 3s+    → FLAGGED +10 score")
    print("  Gauge bar shows iris position in real time")
    print("  Press 'q' to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out = tracker.process(frame)
        cv2.imshow("Eye Tracker v6", out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    tracker.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nFinal score  : {engine.score}")
    print(f"Alerts fired : {sum(1 for e in engine.events if e['type']=='gaze_away')}")
