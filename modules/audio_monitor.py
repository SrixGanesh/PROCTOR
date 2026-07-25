# ============================================================
#  ProctorAI — audio_monitor.py
#  Background audio monitoring — runs in a separate thread
#  Detects voice activity, uses Whisper for speech-to-text
# ============================================================

import threading
import time
import queue
from config import AUDIO_THRESHOLD, WHISPER_MODEL


class AudioMonitor:
    def __init__(self, score_engine):
        self.engine       = score_engine
        self._running     = False
        self._thread      = None
        self._alert_queue = queue.Queue()
        self._last_alert  = 0
        self.COOLDOWN     = 15    # seconds between audio alerts
        self.whisper_model = None
        self._load_whisper()
        print("[AudioMonitor] Initialized ✅")

    # ── Load Whisper model ────────────────────────────────────
    def _load_whisper(self):
        try:
            import whisper
            self.whisper_model = whisper.load_model(WHISPER_MODEL)
            print(f"[AudioMonitor] Whisper '{WHISPER_MODEL}' model loaded ✅")
        except Exception as e:
            print(f"[AudioMonitor] Whisper not available: {e} — using VAD only")

    # ── Start background thread ───────────────────────────────
    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        print("[AudioMonitor] Background monitoring started 🎙️")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        print("[AudioMonitor] Stopped")

    # ── Background monitoring loop ────────────────────────────
    def _monitor_loop(self):
        """Runs in background thread. Checks mic every 2 seconds."""
        try:
            import speech_recognition as sr
            recognizer = sr.Recognizer()
            recognizer.energy_threshold = AUDIO_THRESHOLD

            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=1)
                print(f"[AudioMonitor] Mic ready. Energy threshold: {recognizer.energy_threshold:.0f}")

                while self._running:
                    try:
                        audio = recognizer.listen(source, timeout=2, phrase_time_limit=4)
                        # Voice Activity Detected
                        self._handle_voice(audio, recognizer)
                    except sr.WaitTimeoutError:
                        pass   # silence, continue
                    except Exception as e:
                        if self._running:
                            print(f"[AudioMonitor] Listen error: {e}")
                        time.sleep(1)

        except Exception as e:
            print(f"[AudioMonitor] Mic init failed: {e}")
            print("[AudioMonitor] Running in simulation mode")
            self._simulate()

    # ── Handle detected voice ─────────────────────────────────
    def _handle_voice(self, audio, recognizer):
        now = time.time()
        if now - self._last_alert < self.COOLDOWN:
            return

        detail = ""

        # Try Whisper transcription
        if self.whisper_model is not None:
            try:
                import tempfile, os
                import numpy as np

                raw = np.frombuffer(audio.get_wav_data(), dtype=np.int16).astype(np.float32) / 32768.0
                result = self.whisper_model.transcribe(raw, fp16=False, language="en")
                text = result.get("text", "").strip()
                if text:
                    detail = f'Transcription: "{text[:80]}"'
                    print(f"[AudioMonitor] Whisper: {text}")
            except Exception as e:
                print(f"[AudioMonitor] Whisper transcription error: {e}")

        # Fallback — Google SR
        if not detail:
            try:
                import speech_recognition as sr
                text = recognizer.recognize_google(audio)
                detail = f'Detected speech: "{text[:80]}"'
            except Exception:
                detail = "Voice activity detected"

        self.engine.audio_detected(detail)
        self._last_alert = now

    # ── Simulation fallback (when no mic) ─────────────────────
    def _simulate(self):
        """Simulate audio events for testing without a microphone."""
        import random
        while self._running:
            time.sleep(random.uniform(30, 90))
            if self._running:
                self.engine.audio_detected("Simulated voice activity")

    # ── Check alert queue (call from main thread) ─────────────
    def check_alerts(self):
        alerts = []
        while not self._alert_queue.empty():
            alerts.append(self._alert_queue.get_nowait())
        return alerts


# ── Quick standalone test ────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from modules.score_engine import ScoreEngine

    engine  = ScoreEngine("TEST001", "Test Student")
    monitor = AudioMonitor(engine)
    monitor.start()

    print("Audio monitoring for 30 seconds — speak into mic to test...")
    time.sleep(30)
    monitor.stop()
    print(f"\nFinal score: {engine.score}")
    print(f"Audio alerts: {sum(1 for e in engine.events if e['type']=='audio')}")
