# ============================================================
#  ProctorAI — app.py  (v3 — Multi-Agent)
#  Flask + SocketIO backend
#  All modules run as agents under an Orchestrator
# ============================================================

import sys, os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import cv2, base64, threading, time
from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit

from modules.score_engine import ScoreEngine
from proctor_agents.face_agent    import FaceAgent
from proctor_agents.eye_agent     import EyeAgent
from proctor_agents.yolo_agent    import YOLOAgent
from proctor_agents.orchestrator  import OrchestratorAgent

# ── App ───────────────────────────────────────────────────────
app      = Flask(__name__, static_folder="static", template_folder="static")
app.config["SECRET_KEY"] = "proctor_ai_2025"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ── Global state ──────────────────────────────────────────────
engine       = None
face_agent   = None
eye_agent    = None
yolo_agent   = None
orchestrator = None
camera       = None
running      = False
cam_thread   = None

# ── Routes ────────────────────────────────────────────────────
@app.route("/")
def index():
    with open(os.path.join(BASE_DIR, "static", "index.html"), encoding="utf-8") as f:
        return f.read()

@app.route("/api/score")
def get_score():
    if engine:
        return jsonify({"score":engine.score,"status":engine.status,"events":len(engine.events)})
    return jsonify({"score":0,"status":"safe","events":0})

@app.route("/api/report")
def get_report():
    if engine:
        return jsonify(engine.get_report())
    return jsonify({})

@app.route("/api/ai_report")
def get_ai_report():
    if orchestrator:
        return jsonify(orchestrator.generate_final_report())
    return jsonify({"report":"No session active."})

@app.route("/api/assessments")
def get_assessments():
    if orchestrator:
        return jsonify(orchestrator.assessments)
    return jsonify([])

@app.route("/api/agent_events")
def get_agent_events():
    if orchestrator:
        return jsonify(orchestrator.event_log[-50:])
    return jsonify([])

# ── SocketIO ──────────────────────────────────────────────────
@socketio.on("connect")
def on_connect():
    print(f"[WS] Client: {request.sid}")
    if engine:
        emit("score_update", {"score":engine.score,"status":engine.status,"events":len(engine.events)})

@socketio.on("start_exam")
def on_start(data):
    global engine, face_agent, eye_agent, yolo_agent, orchestrator, running, cam_thread

    name = data.get("name", "Student")
    sid  = data.get("id",   "S001")

    # Init score engine
    engine = ScoreEngine(sid, name, socketio)

    # Init orchestrator FIRST (agents need it)
    orchestrator = OrchestratorAgent(name, sid, engine, socketio)

    # Init agents — each reports to orchestrator
    face_agent = FaceAgent(engine, orchestrator)
    eye_agent  = EyeAgent(engine,  orchestrator)
    yolo_agent = YOLOAgent(engine, orchestrator)

    # Start orchestrator assessment loop
    orchestrator.start()

    # Start camera
    running    = True
    cam_thread = threading.Thread(target=camera_loop, daemon=True)
    cam_thread.start()

    emit("exam_started", {"name": name, "id": sid})
    print(f"\n[App] Exam started — {name} ({sid})")
    print(f"[App] Agents: FaceAgent + EyeAgent + YOLOAgent + Orchestrator")

@socketio.on("stop_exam")
def on_stop():
    global running
    running = False
    if orchestrator: orchestrator.stop()
    if eye_agent:    eye_agent.close()
    emit("exam_stopped", {})
    print("[App] Exam stopped")

@socketio.on("tab_hidden")
def on_tab():
    if engine:
        engine.tab_switched()
        if orchestrator:
            orchestrator.receive({
                "from"    : "BrowserAgent",
                "type"    : "tab_switch",
                "severity": "warning",
                "time"    : "",
                "data"    : {"detail": "Student switched tab"},
            })

@socketio.on("request_assessment")
def on_request_assessment():
    """Dashboard can request an immediate Claude assessment."""
    if orchestrator:
        threading.Thread(
            target=orchestrator._run_claude_assessment,
            args=("manual_request",),
            daemon=True
        ).start()
        emit("assessment_requested", {"status": "Claude is analysing..."})

# ── Camera loop ───────────────────────────────────────────────
def camera_loop():
    global camera, running
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 30)

    frame_count = 0
    print("[Camera] Started")

    while running:
        ret, frame = camera.read()
        if not ret:
            time.sleep(0.05)
            continue

        # Each agent processes the frame
        frame = face_agent.observe(frame)
        frame = eye_agent.observe(frame)
        if frame_count % 3 == 0:
            frame = yolo_agent.observe(frame)

        # Stream to browser
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
        b64    = base64.b64encode(buf).decode("utf-8")
        socketio.emit("frame", {"image": b64})

        # Score update every 30 frames
        if frame_count % 30 == 0 and engine:
            socketio.emit("score_update", {
                "score" : engine.score,
                "status": engine.status,
                "events": len(engine.events),
            })

        frame_count += 1
        time.sleep(1/30)

    camera.release()
    print("[Camera] Stopped")

# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(os.path.join(BASE_DIR, "static"),   exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "database"), exist_ok=True)

    print("\n" + "="*55)
    print("  ProctorAI — Multi-Agent System")
    print("  Agents: Face · Eye · YOLO · Orchestrator")
    print("  Brain:  Claude Sonnet (claude-sonnet-4-20250514)")
    print("  URL:    http://127.0.0.1:5000")
    print("="*55 + "\n")

    socketio.run(app, host="127.0.0.1", port=5000,
                 debug=False, use_reloader=False, allow_unsafe_werkzeug=True)
