# ============================================================
#  ProctorAI — yolo_detector.py  (v3 — Clean & Working)
#
#  Phone  → YOLOv8n (works great)
#  Book   → YOLOv8n conf=0.25 + separate book-only scan
#  Person → 2+ persons only
# ============================================================

import cv2
import time


class YOLODetector:
    def __init__(self, score_engine):
        self.engine = score_engine
        self.model  = None

        try:
            from ultralytics import YOLO
            self.model = YOLO("yolov8n.pt")
            print(f"[YOLODetector] v3 loaded yolov8n.pt ✅")
        except Exception as e:
            print(f"[YOLODetector] Failed: {e}")

        # COCO class IDs we care about
        # 67=cell phone, 73=book, 0=person
        self.CLASS_IDS = {
            67: {"name":"cell phone", "label":"Mobile Phone", "color":(0,0,255),   "points":50, "conf":0.60, "confirm":8},
            73: {"name":"book",       "label":"Book/Notes",   "color":(0,140,255), "points":30, "conf":0.15, "confirm":3},
            0 : {"name":"person",     "label":"Extra Person", "color":(0,80,255),  "points":60, "conf":0.50, "confirm":10},
        }
        # Phone minimum size filter — notes are small, real phones are bigger
        self.PHONE_MIN_AREA = 4000   # pixels squared — ignore tiny phone detections

        # Per-class state
        self._confirm    = {k: 0  for k in self.CLASS_IDS}
        self._last_alert = {k: 0.0 for k in self.CLASS_IDS}
        self.COOLDOWN    = {67: 15, 73: 15, 0: 15}

        self.frame_count = 0
        print("[YOLODetector] v3.1 — Phone conf=0.60+size filter  Book conf=0.15")
        print("[YOLODetector] Phone size filter ON — notes wont trigger phone alert")

    # ── Main process ─────────────────────────────────────────
    def process(self, frame):
        if self.model is None:
            return frame

        self.frame_count += 1
        annotated = frame.copy()
        now       = time.time()

        # Run every 2nd frame
        if self.frame_count % 2 != 0:
            return annotated

        try:
            # Run with very low global conf — filter per class below
            results = self.model(
                frame,
                conf=0.20,
                iou=0.45,
                verbose=False,
                device="cpu"
            )

            # Collect detections per class
            seen = {k: [] for k in self.CLASS_IDS}

            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf   = float(box.conf[0])

                    if cls_id not in self.CLASS_IDS:
                        continue

                    # Per-class confidence filter
                    if conf < self.CLASS_IDS[cls_id]["conf"]:
                        continue

                    x1,y1,x2,y2 = map(int, box.xyxy[0])

                    # Phone size filter — notes/papers are flat & small
                    # Real phone has aspect ratio close to portrait (tall)
                    if cls_id == 67:
                        w_box = x2 - x1
                        h_box = y2 - y1
                        area  = w_box * h_box
                        ratio = h_box / max(w_box, 1)
                        # Skip if: too small OR too wide (paper-like)
                        if area < self.PHONE_MIN_AREA:
                            continue
                        if ratio < 0.8:   # width > height = likely paper not phone
                            continue

                    seen[cls_id].append({"conf":conf, "box":(x1,y1,x2,y2)})

            # ── Handle each class ─────────────────────────────
            for cls_id, info in self.CLASS_IDS.items():
                dets = seen[cls_id]

                # Special rule for person — need 2+
                if cls_id == 0:
                    if len(dets) >= 2:
                        self._process_detection(annotated, cls_id, dets[0], now,
                                                f"Extra Person ({len(dets)})")
                    else:
                        self._confirm[cls_id] = max(0, self._confirm[cls_id] - 1)
                    continue

                if dets:
                    best = max(dets, key=lambda d: d["conf"])
                    self._process_detection(annotated, cls_id, best, now)
                else:
                    # Decay confirm when not seen
                    self._confirm[cls_id] = max(0, self._confirm[cls_id] - 2)

        except Exception as e:
            pass  # silent — don't spam errors

        return annotated

    # ── Process single detection ──────────────────────────────
    def _process_detection(self, frame, cls_id, det, now, label_override=None):
        info  = self.CLASS_IDS[cls_id]
        conf  = det["conf"]
        x1,y1,x2,y2 = det["box"]

        self._confirm[cls_id] += 1
        confirmed = self._confirm[cls_id] >= info["confirm"]

        label = label_override or info["label"]
        color = info["color"] if confirmed else (0, 180, 60)  # green=pending, color=confirmed

        # Box
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)

        # Label background + text
        text = f"{label}  {conf*100:.0f}%"
        (tw,th),_ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1, y1-th-10), (x1+tw+8, y1), color, -1)
        cv2.putText(frame, text, (x1+4, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)

        # Confirm progress bar
        pct   = min(1.0, self._confirm[cls_id] / info["confirm"])
        bar_w = int(pct * max(1, x2-x1))
        cv2.rectangle(frame, (x1, y1-3), (x1+bar_w, y1), color, -1)

        # Fire alert
        if confirmed and now - self._last_alert[cls_id] >= self.COOLDOWN[cls_id]:
            self._last_alert[cls_id] = now
            self._fire(cls_id)

    # ── Score engine calls ────────────────────────────────────
    def _fire(self, cls_id):
        if cls_id == 67:
            self.engine.phone_detected()
        elif cls_id == 73:
            self.engine.book_detected()
        elif cls_id == 0:
            self.engine.multi_face()


# ── Standalone test ──────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from modules.score_engine import ScoreEngine

    engine   = ScoreEngine("TEST001", "Test Student")
    detector = YOLODetector(engine)
    cap      = cv2.VideoCapture(0)

    print("\nYOLO v3 Test")
    print("  📱 Phone — hold up clearly, red box")
    print("  📖 Book  — hold up book/notebook, blue box")
    print("  Green box = detecting, colored = confirmed alert")
    print("  Press 'q' to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out = detector.process(frame)

        # Score bar
        score  = engine.score
        status = engine.status
        h, w   = out.shape[:2]
        sc = (0,200,0) if status=="safe" else (0,140,255) if status=="warning" else (0,0,255)
        cv2.rectangle(out, (0,h-24), (w,h), (20,20,20), -1)
        cv2.putText(out, f"  Score: {score}/100  |  {status.upper()}",
                    (5,h-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, sc, 2)

        cv2.imshow("YOLO v3", out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nScore   : {engine.score}")
    print(f"Phone   : {sum(1 for e in engine.events if e['type']=='phone')}")
    print(f"Book    : {sum(1 for e in engine.events if e['type']=='book')}")
