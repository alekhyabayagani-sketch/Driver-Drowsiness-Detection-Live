
from flask import Flask, render_template, jsonify, request
import cv2
import mediapipe as mp
import math
import time
import numpy as np
from threading import Lock

app = Flask(__name__)

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
def home():
    return render_template("index.html")


@app.route("/start")
def start():
    global monitoring, blink_count, closed_since
    global current_status, eyes_were_closed

    blink_count = 0
    closed_since = None
    eyes_were_closed = False
    current_status = "Face not detected"
    monitoring = True

    return jsonify({"status": "Monitoring started"})


@app.route("/stop")
def stop():
    global monitoring, closed_since
    global current_status, eyes_were_closed

    monitoring = False
    closed_since = None
    eyes_were_closed = False
    current_status = "Monitoring stopped"

    return jsonify({"status": "Monitoring stopped"})


@app.route("/process_frame", methods=["POST"])
def process_frame():
    global monitoring, blink_count, closed_since
    global current_status, eyes_were_closed

    if not monitoring:
        return jsonify({
            "monitoring": False,
            "blink_count": blink_count,
            "detection_status": "Monitoring stopped"
        })

    file = request.files.get("frame")

    if file is None:
        return jsonify({"error": "No camera frame received"}), 400

    try:
        image_bytes = file.read()
        image_array = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Invalid camera frame"}), 400

        # Resize large frames for stable processing
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

            return jsonify({
                "monitoring": True,
                "blink_count": blink_count,
                "detection_status": current_status,
                "eye_ratio": None
            })

        face = results.multi_face_landmarks[0]
        height, width = frame.shape[:2]

        points = [
            (landmark.x * width, landmark.y * height)
            for landmark in face.landmark
        ]

        left_ear = eye_ratio(points, LEFT_EYE)
        right_ear = eye_ratio(points, RIGHT_EYE)
        average_ear = (left_ear + right_ear) / 2

        if average_ear < EYE_THRESHOLD:
            if not eyes_were_closed:
                closed_since = time.time()
                eyes_were_closed = True

            if closed_since is not None:
                closed_duration = time.time() - closed_since

                if closed_duration >= DROWSINESS_TIME:
                    current_status = "DROWSINESS ALERT!"
                else:
                    current_status = "Eyes Closed"

        else:
            if eyes_were_closed and closed_since is not None:
                closed_duration = time.time() - closed_since

                if 0.08 <= closed_duration < DROWSINESS_TIME:
                    blink_count += 1

            closed_since = None
            eyes_were_closed = False
            current_status = "Eyes Open"

        return jsonify({
            "monitoring": True,
            "blink_count": blink_count,
            "detection_status": current_status,
            "eye_ratio": round(average_ear, 3)
        })

    except Exception as error:
        print("Frame processing error:", error)
        return jsonify({
            "error": "Frame processing failed",
            "details": str(error)
        }), 500


@app.route("/status")
def status():
    return jsonify({
        "monitoring": monitoring,
        "blink_count": blink_count,
        "detection_status": current_status
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        use_reloader=False
    )
