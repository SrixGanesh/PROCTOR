# ============================================================
#  ProctorAI — orchestrator.py
#  ORCHESTRATOR AGENT — the brain of the entire system
#
#  Responsibilities:
#  1. Receive events from all sub-agents
#  2. Build context window of session activity
#  3. Call Claude API to reason about suspicious patterns
#  4. Make final verdict decisions
#  5. Escalate to dashboard via WebSocket
#
#  Claude API is called every 60 seconds OR when a critical
#  event is received — to give a smart behavioral assessment
# ============================================================

import sys, os, time, json, threading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from proctor_agents.base_agent import BaseAgent
import requests
from datetime import datetime


class OrchestratorAgent(BaseAgent):
    def __init__(self, student_name: str, student_id: str,
                 score_engine=None, socketio=None):
        super().__init__("Orchestrator", "Coordinate all agents and make final decisions")
        self.student_name  = student_name
        self.student_id    = student_id
        self.score_engine  = score_engine
        self.socketio      = socketio

        # All events from sub-agents
        self.event_log     = []

        # Claude assessment history
        self.assessments   = []

        # Timing
        self._last_assessment  = 0
        self.ASSESS_INTERVAL   = 60    # assess every 60 seconds
        self.CRITICAL_ASSESS   = True  # also assess on critical events

        # Auto-assess thread
        self._running = False
        self.status   = "active"

        print(f"[Orchestrator] Initialized for {student_name} ({student_id}) ✅")
        print(f"[Orchestrator] Claude API assessment every {self.ASSESS_INTERVAL}s")

    # ── Start background assessment loop ──────────────────────
    def start(self):
        self._running = True
        t = threading.Thread(target=self._assess_loop, daemon=True)
        t.start()

    def stop(self):
        self._running = False

    # ── Receive events from sub-agents ────────────────────────
    def receive(self, message: dict):
        """All sub-agents report here."""
        self.event_log.append(message)
        print(f"[Orchestrator] ← {message['from']}: {message['type']} [{message['severity']}]")

        # Emit to dashboard
        if self.socketio:
            self.socketio.emit("agent_event", {
                "from"    : message["from"],
                "type"    : message["type"],
                "severity": message["severity"],
                "time"    : message["time"],
                "data"    : message.get("data", {}),
            })

        # Immediately assess on critical events
        if (message["severity"] == "critical"
                and self.CRITICAL_ASSESS
                and time.time() - self._last_assessment > 15):
            threading.Thread(
                target=self._run_claude_assessment,
                args=("critical_event",),
                daemon=True
            ).start()

    # ── Background assessment loop ────────────────────────────
    def _assess_loop(self):
        time.sleep(30)  # wait 30s before first assessment
        while self._running:
            if (len(self.event_log) > 0
                    and time.time() - self._last_assessment >= self.ASSESS_INTERVAL):
                self._run_claude_assessment("periodic")
            time.sleep(10)

    # ── Claude API Assessment ─────────────────────────────────
    def _run_claude_assessment(self, trigger: str = "periodic"):
        self._last_assessment = time.time()

        # Build context from recent events
        recent = self.event_log[-30:]  # last 30 events
        score  = self.score_engine.score if self.score_engine else 0

        # Format events for Claude
        event_summary = []
        for e in recent:
            event_summary.append(
                f"  [{e['time']}] {e['from']} → {e['type']} ({e['severity']})"
                + (f": {e['data'].get('reason','') or e['data'].get('direction','') or e['data'].get('object','')}" if e.get('data') else "")
            )

        context = f"""You are an AI exam proctor analyzing a student's behavior during an online exam.

STUDENT: {self.student_name} (ID: {self.student_id})
CURRENT RISK SCORE: {score}/100
TRIGGER: {trigger}
SESSION TIME: {datetime.now().strftime('%H:%M:%S')}

RECENT EVENTS ({len(recent)} events):
{chr(10).join(event_summary) if event_summary else '  No events yet'}

TASK:
Analyze this behavioral data and provide:
1. VERDICT: SAFE / WARNING / SUSPICIOUS / CHEATING
2. CONFIDENCE: low / medium / high
3. KEY_FINDING: One sentence — what's the most suspicious pattern?
4. REASONING: 2-3 sentences explaining your assessment
5. RECOMMENDATION: What should the proctor do?

Respond ONLY in this exact JSON format:
{{
  "verdict": "SAFE",
  "confidence": "high",
  "key_finding": "No suspicious activity detected",
  "reasoning": "The student has maintained consistent eye contact with the screen and no prohibited objects have been detected.",
  "recommendation": "Continue monitoring normally."
}}"""

        print(f"[Orchestrator] Calling Claude API (trigger={trigger})...")

        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={"Content-Type": "application/json"},
                json={
                    "model"     : "claude-sonnet-4-20250514",
                    "max_tokens": 400,
                    "messages"  : [{"role": "user", "content": context}]
                },
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                raw  = data["content"][0]["text"].strip()

                # Parse JSON response
                try:
                    # Strip markdown code fences if present
                    if "```" in raw:
                        raw = raw.split("```")[1]
                        if raw.startswith("json"):
                            raw = raw[4:]

                    assessment = json.loads(raw)
                    assessment["time"]    = datetime.now().strftime("%H:%M:%S")
                    assessment["trigger"] = trigger
                    assessment["score"]   = score

                    self.assessments.append(assessment)
                    self._handle_assessment(assessment)
                    print(f"[Orchestrator] Claude verdict: {assessment['verdict']} ({assessment['confidence']}) — {assessment['key_finding']}")

                except json.JSONDecodeError:
                    print(f"[Orchestrator] Could not parse Claude response: {raw[:100]}")

            else:
                print(f"[Orchestrator] Claude API error: {response.status_code}")

        except Exception as e:
            print(f"[Orchestrator] Claude API call failed: {e}")

    # ── Handle assessment result ──────────────────────────────
    def _handle_assessment(self, assessment: dict):
        verdict = assessment.get("verdict", "SAFE")

        # Push to dashboard
        if self.socketio:
            self.socketio.emit("ai_assessment", {
                "verdict"    : verdict,
                "confidence" : assessment.get("confidence"),
                "key_finding": assessment.get("key_finding"),
                "reasoning"  : assessment.get("reasoning"),
                "recommendation": assessment.get("recommendation"),
                "time"       : assessment["time"],
                "score"      : assessment["score"],
            })

        # Auto-escalate score if Claude says cheating
        if verdict == "CHEATING" and self.score_engine:
            if self.score_engine.score < 70:
                self.score_engine.add_event(
                    "ai_assessment",
                    f"AI Assessment: {assessment['key_finding']}",
                    20
                )
                print(f"[Orchestrator] Auto-escalated score due to CHEATING verdict")

        # Log to score engine
        if self.score_engine:
            self.score_engine.add_event(
                "ai_assessment",
                f"AI: {verdict} — {assessment['key_finding'][:60]}",
                0   # no score change for info assessments
            )

    # ── Generate final report via Claude ─────────────────────
    def generate_final_report(self) -> dict:
        print("[Orchestrator] Generating final report via Claude...")

        score   = self.score_engine.score if self.score_engine else 0
        report  = self.score_engine.get_report() if self.score_engine else {}

        # Summarize all assessments
        assessment_summary = ""
        for a in self.assessments[-5:]:
            assessment_summary += f"\n  [{a['time']}] {a['verdict']} ({a['confidence']}): {a['key_finding']}"

        prompt = f"""You are writing a final exam integrity report for an invigilator.

STUDENT: {self.student_name} ({self.student_id})
FINAL RISK SCORE: {score}/100
TAB SWITCHES: {report.get('tab_switches', 0)}
GAZE ALERTS: {report.get('gaze_alerts', 0)}
PHONE DETECTIONS: {report.get('phone_detections', 0)}
TOTAL EVENTS: {report.get('total_events', 0)}

AI ASSESSMENTS DURING EXAM:
{assessment_summary or '  No assessments recorded'}

Write a professional 3-paragraph exam integrity report:
1. Executive summary of the session
2. Key incidents and behavioral patterns observed
3. Final recommendation (Pass / Review / Disqualify)

Be concise, professional, and specific."""

        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={"Content-Type": "application/json"},
                json={
                    "model"     : "claude-sonnet-4-20250514",
                    "max_tokens": 600,
                    "messages"  : [{"role": "user", "content": prompt}]
                },
                timeout=30
            )

            if response.status_code == 200:
                text = response.json()["content"][0]["text"]
                return {
                    "report"      : text,
                    "assessments" : self.assessments,
                    "event_count" : len(self.event_log),
                    "final_score" : score,
                }
        except Exception as e:
            print(f"[Orchestrator] Final report error: {e}")

        return {"report": "Report generation failed.", "final_score": score}

    # ── Observe (not used directly) ───────────────────────────
    def observe(self, data):
        pass

    def think(self, observation):
        return {}

    def act(self, decision):
        pass
