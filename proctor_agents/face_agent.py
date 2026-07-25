# ============================================================
#  ProctorAI — face_agent.py
#  Face Agent — wraps FaceModule, reports to orchestrator
# ============================================================

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from proctor_agents.base_agent import BaseAgent


class FaceAgent(BaseAgent):
    def __init__(self, score_engine, orchestrator=None):
        super().__init__("FaceAgent", "Monitor face presence and identity", orchestrator)
        from modules.face_module import FaceModule
        self.module       = FaceModule(score_engine)
        self.score_engine = score_engine
        self.status       = "active"
        print(f"[{self.name}] Initialized ✅")

    def observe(self, frame):
        """Process frame, return annotated frame."""
        return self.module.process(frame)

    def think(self, observation: dict) -> dict:
        """Analyze recent memory for patterns."""
        no_face_events = self.recall_type("no_face")
        multi_events   = self.recall_type("multi_face")

        risk = "low"
        reason = ""

        if len(no_face_events) >= 3:
            risk   = "high"
            reason = f"Face missing {len(no_face_events)} times"
        elif len(multi_events) >= 2:
            risk   = "critical"
            reason = f"Multiple persons detected {len(multi_events)} times"

        return {"risk": risk, "reason": reason}

    def act(self, decision: dict):
        if decision["risk"] in ("high", "critical"):
            self.report("face_pattern_alert", decision, severity="warning")

    def on_event(self, event_type: str, detail: str = ""):
        """Called by FaceModule when something is detected."""
        self.remember({"type": event_type, "detail": detail})
        severity = "critical" if event_type in ("multi_face", "identity_mismatch") else "warning"
        self.report(event_type, {"detail": detail}, severity)
        decision = self.think({})
        self.act(decision)
