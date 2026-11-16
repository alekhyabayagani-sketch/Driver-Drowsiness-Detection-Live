import os
import cv2
import mediapipe as mp
import math
import time
import sqlite3
import numpy as np
from functools import wraps
from threading import Lock
from flask import Flask, render_template, jsonify, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE = os.path.join(BASE_DIR, "drivesafe.db")

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "change-this-secret-key-before-production")
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = os.environ.get("COOKIE_SECURE", "false").lower() == "true"

# Existing monitoring state and detection settings
monitoring = False
blink_count = 0
closed_since = None
current_status = "Monitoring stopped"
eyes_were_closed = False
face_lock = Lock()

face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
EYE_THRESHOLD = 0.23
DROWSINESS_TIME = 2.0


def get_db():
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    with get_db() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            if request.path in ("/process_frame", "/status", "/start", "/stop"):
                return jsonify({"error": "Please log in first."}), 401
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped


def distance(p1, p2):
    return math.dist(p1, p2)


def eye_ratio(points, indexes):
    p = [points[i] for i in indexes]
    vertical1 = distance(p[1], p[5])
    vertical2 = distance(p[2], p[4])
    horizontal = distance(p[0], p[3])
    if horizontal == 0:
        return 0
    return (vertical1 + vertical2) / (2 * horizontal)


@app.route("/")
@login_required
def home():
    return render_template("index.html", user_name=session.get("full_name", "Driver"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if "user_id" in session:
        return redirect(url_for("home"))
    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not full_name or not email or not password or not confirm_password:
            flash("Please fill in all fields.", "error")
        elif len(full_name) > 80:
            flash("Name must be 80 characters or fewer.", "error")
        elif len(password) < 8:
            flash("Password must contain at least 8 characters.", "error")
        elif password != confirm_password:
            flash("Passwords do not match.", "error")
        else:
            try:
                with get_db() as connection:
                    cursor = connection.execute(
                        "INSERT INTO users (full_name, email, password_hash) VALUES (?, ?, ?)",
                        (full_name, email, generate_password_hash(password))
                    )
                    user_id = cursor.lastrowid
                session.clear()
                session["user_id"] = user_id
                session["full_name"] = full_name
                session["email"] = email
                flash("Your account has been created. Welcome to DriveSafe!", "success")
                return redirect(url_for("home"))
            except sqlite3.IntegrityError:
                flash("An account with this email already exists. Please log in.", "error")
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("home"))
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        with get_db() as connection:
            user = connection.execute(
                "SELECT * FROM users WHERE email = ?", (email,)
            ).fetchone()
        if user and check_password_hash(user["password_hash"], password):
            session.clear()
            session["user_id"] = user["id"]
            session["full_name"] = user["full_name"]
            session["email"] = user["email"]
            return redirect(url_for("home"))
        flash("Invalid email or password. Please try again.", "error")
    return render_template("login.html")


@app.route("/logout", methods=["POST"])
@login_required
def logout():
    global monitoring, closed_since, current_status, eyes_were_closed
    monitoring = False
    closed_since = None
    eyes_were_closed = False
    current_status = "Monitoring stopped"
    session.clear()
    return redirect(url_for("login"))


@app.route("/start")
@login_required
def start():
    global monitoring, blink_count, closed_since, current_status, eyes_were_closed
    blink_count = 0
    closed_since = None
    eyes_were_closed = False
    current_status = "Face not detected"
    monitoring = True
    return jsonify({"status": "Monitoring started"})


@app.route("/stop")
@login_required
def stop():
    global monitoring, closed_since, current_status, eyes_were_closed
    monitoring = False
    closed_since = None
    eyes_were_closed = False
    current_status = "Monitoring stopped"
    return jsonify({"status": "Monitoring stopped"})


@app.route("/process_frame", methods=["POST"])
@login_required
def process_frame():
    global monitoring, blink_count, closed_since, current_status, eyes_were_closed
    if not monitoring:
        return jsonify({"monitoring": False, "blink_count": blink_count,
                        "detection_status": "Monitoring stopped"})
    file = request.files.get("frame")
    if file is None:
        return jsonify({"error": "No camera frame received"}), 400
    try:
        image_bytes = file.read()
        image_array = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Invalid camera frame"}), 400
        height, width = frame.shape[:2]
        if width > 640:
            new_height = int(height * 640 / width)
            frame = cv2.resize(frame, (640, new_height))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        with face_lock:
            results = face_mesh.process(rgb_frame)
        if not results.multi_face_landmarks:
            current_status = "Face not detected"
            closed_since = None
            eyes_were_closed = False
            return jsonify({"monitoring": True, "blink_count": blink_count,
                            "detection_status": current_status, "eye_ratio": None})
        face = results.multi_face_landmarks[0]
        height, width = frame.shape[:2]
        points = [(landmark.x * width, landmark.y * height) for landmark in face.landmark]
        left_ear = eye_ratio(points, LEFT_EYE)
        right_ear = eye_ratio(points, RIGHT_EYE)
        average_ear = (left_ear + right_ear) / 2
        if average_ear < EYE_THRESHOLD:
            if not eyes_were_closed:
                closed_since = time.time()
                eyes_were_closed = True
            if closed_since is not None:
                closed_duration = time.time() - closed_since
                current_status = "DROWSINESS ALERT!" if closed_duration >= DROWSINESS_TIME else "Eyes Closed"
        else:
            if eyes_were_closed and closed_since is not None:
                closed_duration = time.time() - closed_since
                if 0.08 <= closed_duration < DROWSINESS_TIME:
                    blink_count += 1
            closed_since = None
            eyes_were_closed = False
            current_status = "Eyes Open"
        return jsonify({"monitoring": True, "blink_count": blink_count,
                        "detection_status": current_status, "eye_ratio": round(average_ear, 3)})
    except Exception as error:
        print("Frame processing error:", error)
        return jsonify({"error": "Frame processing failed", "details": str(error)}), 500


@app.route("/status")
@login_required
def status():
    return jsonify({"monitoring": monitoring, "blink_count": blink_count,
                    "detection_status": current_status})


init_db()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)),
            debug=False, use_reloader=False)
