# ============================================================
#  ProctorAI — base_agent.py
#  Base class for all agents
#  Every agent: observes → thinks → acts → reports
# ============================================================

import time
from datetime import datetime


class BaseAgent:
    def __init__(self, name: str, role: str, orchestrator=None):
        self.name         = name
        self.role         = role
        self.orchestrator = orchestrator   # parent agent
        self.memory       = []             # agent's own event memory
        self.status       = "idle"         # idle / active / alert
        self.created_at   = datetime.now()

    # ── Core agent loop ───────────────────────────────────────
    def observe(self, data: dict):
        """Receive raw data from sensor/module."""
        raise NotImplementedError

    def think(self, observation: dict) -> dict:
        """Process observation → decision."""
        raise NotImplementedError

    def act(self, decision: dict):
        """Execute based on decision."""
        raise NotImplementedError

    # ── Memory ────────────────────────────────────────────────
    def remember(self, event: dict):
        event["agent"]     = self.name
        event["timestamp"] = time.time()
        event["time"]      = datetime.now().strftime("%H:%M:%S")
        self.memory.append(event)
        # Keep last 100 events
        if len(self.memory) > 100:
            self.memory.pop(0)

    def recall(self, last_n: int = 10) -> list:
        return self.memory[-last_n:]

    def recall_type(self, event_type: str) -> list:
        return [e for e in self.memory if e.get("type") == event_type]

    # ── Report to orchestrator ────────────────────────────────
    def report(self, event_type: str, data: dict, severity: str = "info"):
        """Send event up to orchestrator."""
        msg = {
            "from"    : self.name,
            "type"    : event_type,
            "data"    : data,
            "severity": severity,   # info / warning / critical
            "time"    : datetime.now().strftime("%H:%M:%S"),
        }
        self.remember(msg)
        if self.orchestrator:
            self.orchestrator.receive(msg)
        return msg

    def __repr__(self):
        return f"<Agent:{self.name} status={self.status}>"
