# ============================================================
#  ProctorAI — eye_agent.py (DISABLED VERSION)
#  Eye tracking disabled due to mediapipe/protobuf conflicts
# ============================================================

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from proctor_agents.base_agent import BaseAgent


class EyeAgent(BaseAgent):
    def __init__(self, score_engine, orchestrator=None):
        super().__init__("EyeAgent", "Eye tracking (disabled - protobuf conflict)", orchestrator)
        self.score_engine = score_engine
        self.status = "disabled"
        print(f"[{self.name}] Disabled - protobuf/mediapipe conflict ⚠️")

    def observe(self, frame):
        # Just return frame unchanged
        return frame

    def think(self, observation: dict) -> dict:
        return {"risk": "low", "reason": "Eye tracking disabled"}

    def act(self, decision: dict):
        pass

    def on_event(self, direction: str):
        pass

    def close(self):
        pass
