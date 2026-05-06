from flask import Flask, render_template, Response, jsonify, request, send_from_directory
import cv2
import sqlite3
from ultralytics import YOLO
import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import time
import os
import webbrowser
import face_recognition
import numpy as np

# ✅ DATABASE IMPORT
from database import init_db, get_db_connection, log_mail

# --- Configuration ---
SENDER_EMAIL = "negipallu11@gmail.com"
SENDER_PASSWORD = "hfiwkhdamztksatn"
RECIPIENT_EMAIL = "pallavin167@gmail.com"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

# Alert cooldown
last_alert_time = 0
ALERT_COOLDOWN_SECONDS = 15

app = Flask(__name__)
init_db()

# --- Load YOLO Model ---
try:
    model = YOLO('yolov8n.pt')
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit()

# --- Camera ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

SNAPSHOT_DIR = "snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# --- ✅ FACE RECOGNITION SETUP ---
known_face_encodings = []
known_face_names = []

def load_known_faces():
    base_path = "known_faces" 
    if not os.path.exists(base_path):
        print(f"❌ Error: Directory '{base_path}' not found.")
        return

    for person_name in os.listdir(base_path):
        person_folder = os.path.join(base_path, person_name)
        if not os.path.isdir(person_folder): continue

        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)
            try:
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                
                if len(encodings) > 0:
                    known_face_encodings.append(encodings[0])
                    known_face_names.append(person_name)
                    print(f"✅ Successfully loaded face for: {person_name}")
                else:
                    print(f"⚠️ WARNING: No face found in {image_name}.")
            
            except Exception as e:
                print(f"❌ Error loading {image_name}: {e}")
    
    print(f"--- Final Count: {len(known_face_names)} faces active in system ---")

load_known_faces()

# --- DATABASE LOG FUNCTION ---
def log_detection_db(event_type, confidence, snapshot_path,
                     camera_id="Camera 1", latitude=30.3165, longitude=78.0322):
    conn = get_db_connection()
    cursor = conn.cursor()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO detections
        (timestamp, camera_id, alert_type, confidence, snapshot_path, latitude, longitude)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (timestamp, camera_id, event_type, confidence, snapshot_path, latitude, longitude))
    conn.commit()
    conn.close()

# --- EMAIL ALERT ---
def send_alert_email(event_type, snapshot_path):
    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECIPIENT_EMAIL
        msg['Subject'] = f"CRITICAL AI SURVEILLANCE ALERT: {event_type}"
        body = f"AI Surveillance Alert\n\nEvent Type : {event_type}\nTime : {datetime.datetime.now().strftime('%H:%M:%S')}\nCamera : Camera 1\nStatus : Automated Alert"
        msg.attach(MIMEText(body, 'plain'))
        if snapshot_path and os.path.exists(snapshot_path):
            with open(snapshot_path, 'rb') as f:
                img = MIMEImage(f.read())
                img.add_header('Content-Disposition', 'attachment', filename=os.path.basename(snapshot_path))
                msg.attach(img)
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, msg.as_string())
        server.quit()
        log_mail(subject=f"CRITICAL AI SURVEILLANCE ALERT: {event_type}", recipient=RECIPIENT_EMAIL, status="SENT")
        print(f"Email alert sent for: {event_type}")
    except Exception as e:
        print("Email error:", e)
        log_mail(subject=f"FAILED: ALERT: {event_type}", recipient=RECIPIENT_EMAIL, status="FAILED")

# --- AI + STREAMING ---
def generate_frames():
    global last_alert_time
    process_this_frame = True
    
    face_locations = []
    face_names = []

    while True:
        success, frame = cap.read()
        if not success: break

        results = model(frame, stream=True, conf=0.5, verbose=False)
        annotated_frame = frame.copy()
        detections = []

        # RESET presence flag every frame
        is_known_person_present = False

        if process_this_frame:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                name = "Intruder"
                if len(known_face_encodings) > 0:
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
                    if True in matches:
                        first_match_index = matches.index(True)
                        name = known_face_names[first_match_index]
                face_names.append(name)
        
        process_this_frame = not process_this_frame

        # Check if ANY known face is in the current face_names list
        is_known_person_present = any(name != "Intruder" for name in face_names)

        # Draw YOLO detections (Weapons)
        for r in results:
            for box in r.boxes:
                label = model.names[int(box.cls)]
                conf = float(box.conf)
                detections.append((label, conf))
                if label in ["gun", "knife"]:
                    b = box.xyxy[0].cpu().numpy()
                    cv2.rectangle(annotated_frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 3)

        # Draw Face Recognition Boxes
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4; right *= 4; bottom *= 4; left *= 4
            color = (0, 255, 0) if name != "Intruder" else (0, 0, 255)
            cv2.rectangle(annotated_frame, (left, top), (right, bottom), color, 2)
            cv2.putText(annotated_frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # --- REFINED ALERT LOGIC ---
        threat_detected = False
        event_type = None
        event_confidence = 0.0

        for obj, conf in detections:
            # 1. Weapon Detection always triggers an alert
            if obj in ["gun", "knife"]:
                event_type = "Weapon Detected"
                event_confidence = conf
                threat_detected = True
                break 
            
            # 2. Person Detection only triggers an alert if NOT recognized as Pallavi
            elif obj == "person":
                if not is_known_person_present:
                    event_type = "Intruder Detected"
                    event_confidence = conf
                    threat_detected = True

        if threat_detected and (time.time() - last_alert_time > ALERT_COOLDOWN_SECONDS):
            last_alert_time = time.time()
            filename = f"{event_type.replace(' ', '_')}_{int(time.time())}.jpg"
            snapshot_path = os.path.join(SNAPSHOT_DIR, filename)
            cv2.imwrite(snapshot_path, annotated_frame)
            log_detection_db(event_type, event_confidence, snapshot_path)
            send_alert_email(event_type, snapshot_path)

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# --- ROUTES ---
@app.route('/get_snapshot/<path:filename>')
def get_snapshot(filename):
    return send_from_directory('.', filename)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_mail_logs')
def get_mail_logs():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, subject, recipient, status FROM mail_logs ORDER BY timestamp DESC LIMIT 20")
    rows = cursor.fetchall()
    conn.close()
    return jsonify([{"timestamp": r[0], "subject": r[1], "recipient": r[2], "status": r[3]} for r in rows])

@app.route('/get_logs')
def get_logs():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, camera_id, alert_type, snapshot_path, latitude, longitude FROM detections ORDER BY timestamp DESC LIMIT 20")
    rows = cursor.fetchall()
    conn.close()
    logs = [{"timestamp": r[0], "camera_id": r[1], "alert_type": r[2], "snapshot_path": r[3], "latitude": r[4] or 30.3165, "longitude": r[5] or 78.0322} for r in rows]
    return jsonify({"logs": logs, "alert_count": len(logs)})

@app.route('/manual_alert', methods=['POST'])
def manual_alert():
    send_alert_email("Manual Alert", "")
    return jsonify({"message": "Manual alert sent"})

if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:5000") 
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)