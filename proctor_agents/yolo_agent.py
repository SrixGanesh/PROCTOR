# ============================================================
#  ProctorAI — yolo_agent.py
#  YOLO Agent — detects objects, reasons about patterns
# ============================================================

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from proctor_agents.base_agent import BaseAgent


class YOLOAgent(BaseAgent):
    def __init__(self, score_engine, orchestrator=None):
        super().__init__("YOLOAgent", "Detect prohibited objects in frame", orchestrator)
        from modules.yolo_detector import YOLODetector
        self.module       = YOLODetector(score_engine)
        self.score_engine = score_engine
        self.status       = "active"
        print(f"[{self.name}] Initialized ✅")

    def observe(self, frame):
        return self.module.process(frame)

    def think(self, observation: dict) -> dict:
        phone_events = self.recall_type("phone")
        book_events  = self.recall_type("book")

        risk   = "low"
        reason = ""

        if len(phone_events) >= 2:
            risk   = "critical"
            reason = f"Phone detected {len(phone_events)} times — strong cheating signal"
        elif len(book_events) >= 2:
            risk   = "high"
            reason = f"Notes/book detected {len(book_events)} times"
        elif len(phone_events) == 1 and len(book_events) >= 1:
            risk   = "critical"
            reason = "Both phone and notes detected — likely cheating"

        return {"risk": risk, "reason": reason,
                "phone_count": len(phone_events),
                "book_count" : len(book_events)}

    def act(self, decision: dict):
        if decision["risk"] in ("high", "critical"):
            self.report("object_pattern_alert", decision,
                        severity="critical" if decision["risk"]=="critical" else "warning")

    def on_event(self, obj_type: str):
        self.remember({"type": obj_type})
        severity = "critical" if obj_type == "phone" else "warning"
        self.report(obj_type, {"object": obj_type}, severity)
        decision = self.think({})
        self.act(decision)
