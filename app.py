"""
AI-Based Fake Audio & Video Detection System
Flask Backend — Main Application
"""

import os
import uuid
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from database import init_db, save_detection, get_recent_detections
from utils.audio_detector import analyze_audio
from utils.video_detector import analyze_video

# ─── App setup ───────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB

AUDIO_EXTENSIONS = {"wav", "mp3", "ogg", "flac", "m4a", "aac", "webm"}
VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm", "flv", "wmv"}


def allowed_file(filename: str) -> bool:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in (AUDIO_EXTENSIONS | VIDEO_EXTENSIONS)


def get_file_type(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext in AUDIO_EXTENSIONS:
        return "audio"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    return "unknown"


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    # Save with unique name
    original_name = secure_filename(file.filename)
    ext = original_name.rsplit(".", 1)[-1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(save_path)

    file_type = get_file_type(original_name)

    # Run detection
    if file_type == "audio":
        result = analyze_audio(save_path)
    elif file_type == "video":
        result = analyze_video(save_path)
    else:
        return jsonify({"error": "Could not determine file type"}), 400

    if result.get("error"):
        return jsonify({"error": result["error"]}), 422

    # Persist to DB
    record_id = save_detection(
        filename=original_name,
        file_type=file_type,
        verdict=result["verdict"],
        confidence=result["confidence"],
        details={k: v for k, v in result.items() if k not in ("verdict", "confidence")},
    )
    result["record_id"] = record_id
    result["filename"] = original_name

    # Clean up upload
    try:
        os.remove(save_path)
    except Exception:
        pass

    return jsonify(result)


@app.route("/api/history")
def history():
    records = get_recent_detections(limit=20)
    return jsonify(records)


@app.route("/api/stats")
def stats():
    records = get_recent_detections(limit=1000)
    total = len(records)
    fake = sum(1 for r in records if r["verdict"] == "FAKE")
    real = sum(1 for r in records if r["verdict"] == "REAL")
    uncertain = total - fake - real
    audio = sum(1 for r in records if r["file_type"] == "audio")
    video = sum(1 for r in records if r["file_type"] == "video")
    avg_conf = round(sum(r["confidence"] for r in records) / total, 1) if total else 0
    return jsonify({
        "total": total,
        "fake": fake,
        "real": real,
        "uncertain": uncertain,
        "audio": audio,
        "video": video,
        "avg_confidence": avg_conf,
    })


# ─── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  AI Deepfake Detection System")
    print("  http://127.0.0.1:5000")
    print("=" * 60 + "\n")
    init_db()
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
