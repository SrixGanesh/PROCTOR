# ============================================================
#  ProctorAI — config.py
#  All settings in one place. Edit here, affects everything.
# ============================================================

# ---------- Suspicion Scoring ----------
SCORE_TAB_SWITCH       = 20   # Tab switch detected
SCORE_GAZE_AWAY        = 10   # Looking away from screen
SCORE_NO_FACE          = 25   # Face not visible
SCORE_PHONE_DETECTED   = 50   # Mobile phone in frame
SCORE_BOOK_DETECTED    = 30   # Book/paper in frame
SCORE_MULTI_FACE       = 60   # More than 1 face
SCORE_AUDIO_DETECTED   = 30   # Voice/whisper detected
SCORE_IDENTITY_MISMATCH= 70   # Different person detected

# ---------- Thresholds ----------
SAFE_THRESHOLD    = 30    # 0–30   → Safe ✅
WARNING_THRESHOLD = 70    # 30–70  → Warning ⚠️
                          # 70+    → Cheating 🚨

# ---------- Camera ----------
CAMERA_INDEX      = 0     # 0 = default webcam
FRAME_WIDTH       = 640
FRAME_HEIGHT      = 480
FPS               = 30

# ---------- Face Module ----------
FACE_CONFIDENCE   = 0.7   # DeepFace match threshold
GAZE_FRAMES_LIMIT = 15    # Frames looking away before alert (0.5 sec at 30fps)

# ---------- YOLO ----------
YOLO_MODEL        = "yolov8n.pt"   # nano = fastest, good for CPU
YOLO_CONFIDENCE   = 0.5
YOLO_TARGETS      = ["cell phone", "book", "person"]

# ---------- Audio ----------
AUDIO_THRESHOLD   = 500   # Mic energy threshold
WHISPER_MODEL     = "tiny"  # tiny/base/small — tiny is fastest on CPU

# ---------- Flask Server ----------
HOST              = "127.0.0.1"
PORT              = 5000
DEBUG             = True

# ---------- Database ----------
DB_PATH           = "database/exam_logs.db"

# ---------- Exam Info ----------
EXAM_NAME         = "CS3401 - Data Structures"
EXAM_DURATION_MIN = 180   # 3 hours
