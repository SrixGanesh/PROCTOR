# ============================================================
#  ProctorAI — score_engine.py
#  Collects all module alerts → computes suspicion score
#  Emits real-time events via Flask-SocketIO
# ============================================================

import sqlite3
import time
from datetime import datetime
from config import *


class ScoreEngine:
    def __init__(self, student_id: str, student_name: str, socketio=None):
        self.student_id   = student_id
        self.student_name = student_name
        self.socketio     = socketio
        self.score        = 0
        self.events       = []   # full timeline of incidents
        self._init_db()

    # ── Status label ─────────────────────────────────────────
    @property
    def status(self):
        if self.score < SAFE_THRESHOLD:
            return "safe"
        elif self.score < WARNING_THRESHOLD:
            return "warning"
        else:
            return "cheating"

    # ── Add an incident ──────────────────────────────────────
    def add_event(self, event_type: str, description: str, points: int):
        """
        Call this from any module when suspicious activity is detected.
        event_type: 'tab_switch' | 'gaze_away' | 'no_face' | 'phone' |
                    'book' | 'multi_face' | 'audio' | 'identity_mismatch'
        """
        self.score = min(100, self.score + points)   # cap at 100

        event = {
            "time"       : datetime.now().strftime("%H:%M:%S"),
            "timestamp"  : time.time(),
            "type"       : event_type,
            "description": description,
            "points"     : points,
            "score_after": self.score,
            "status"     : self.status,
        }
        self.events.append(event)
        self._log_to_db(event)

        print(f"[ALERT] {self.student_name} | {description} | +{points} | Score: {self.score} | {self.status.upper()}")

        # Send to dashboard via WebSocket
        if self.socketio:
            self.socketio.emit("alert", {
                "student_id"  : self.student_id,
                "student_name": self.student_name,
                "event"       : event,
            })
            self.socketio.emit("score_update", {
                "student_id": self.student_id,
                "score"     : self.score,
                "status"    : self.status,
            })

        return event

    # ── Convenience wrappers (called by each module) ─────────
    def tab_switched(self):
        return self.add_event("tab_switch", "Tab switch detected", SCORE_TAB_SWITCH)

    def gaze_away(self):
        return self.add_event("gaze_away", "Looking away from screen", SCORE_GAZE_AWAY)

    def no_face(self):
        return self.add_event("no_face", "Face not visible in frame", SCORE_NO_FACE)

    def phone_detected(self):
        return self.add_event("phone", "Mobile phone detected in frame", SCORE_PHONE_DETECTED)

    def book_detected(self):
        return self.add_event("book", "Book/paper detected in frame", SCORE_BOOK_DETECTED)

    def multi_face(self):
        return self.add_event("multi_face", "Multiple faces detected", SCORE_MULTI_FACE)

    def audio_detected(self, detail=""):
        desc = f"Voice/whisper activity detected. {detail}".strip()
        return self.add_event("audio", desc, SCORE_AUDIO_DETECTED)

    def identity_mismatch(self):
        return self.add_event("identity_mismatch", "Identity mismatch — possible swap", SCORE_IDENTITY_MISMATCH)

    # ── Summary report ───────────────────────────────────────
    def get_report(self):
        return {
            "student_id"  : self.student_id,
            "student_name": self.student_name,
            "final_score" : self.score,
            "status"      : self.status,
            "total_events": len(self.events),
            "events"      : self.events,
            "tab_switches"     : sum(1 for e in self.events if e["type"] == "tab_switch"),
            "gaze_alerts"      : sum(1 for e in self.events if e["type"] == "gaze_away"),
            "phone_detections" : sum(1 for e in self.events if e["type"] == "phone"),
            "audio_flags"      : sum(1 for e in self.events if e["type"] == "audio"),
        }

    # ── Database ─────────────────────────────────────────────
    def _init_db(self):
        # Auto-create database folder if not exists
        import os
        db_dir = os.path.dirname(os.path.abspath(__file__))
        db_dir = os.path.join(db_dir, "..", "database")
        os.makedirs(db_dir, exist_ok=True)
        self._db_path = os.path.join(db_dir, "exam_logs.db")
        conn = sqlite3.connect(self._db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id  TEXT,
                student_name TEXT,
                event_type  TEXT,
                description TEXT,
                points      INTEGER,
                score_after INTEGER,
                status      TEXT,
                timestamp   REAL,
                time_str    TEXT
            )
        """)
        conn.commit()
        conn.close()

    def _log_to_db(self, event: dict):
        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute("""
                INSERT INTO events
                (student_id, student_name, event_type, description, points, score_after, status, timestamp, time_str)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, (
                self.student_id, self.student_name,
                event["type"], event["description"],
                event["points"], event["score_after"],
                event["status"], event["timestamp"], event["time"]
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[DB ERROR] {e}")


# ── Quick test ───────────────────────────────────────────────
if __name__ == "__main__":
    engine = ScoreEngine("CS2301", "Karthik R")
    engine.tab_switched()
    engine.gaze_away()
    engine.phone_detected()
    print("\n--- Report ---")
    import json
    print(json.dumps(engine.get_report(), indent=2))
