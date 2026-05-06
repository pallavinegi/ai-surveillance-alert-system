import cv2
from ultralytics import YOLO
import face_recognition
import os
import numpy as np
import time
import datetime

# --- CONFIG ---
CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 0.5
SCALE_FACTOR = 0.25 
COOLDOWN = 10
SNAPSHOT_DIR = "snapshots"

# Ensure snapshot directory exists
if not os.path.exists(SNAPSHOT_DIR):
    os.makedirs(SNAPSHOT_DIR)

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
            except Exception as e:
                print(f"Skipping {image_name}: {e}")
    print(f"✅ Loaded {len(known_face_names)} reference faces.")

def run_local_detector():
    load_known_faces()
    # Loading the model with 'verbose=False' reduces terminal lag
    model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    last_capture_time = 0
    process_this_frame = True
    
    # Initialize variables to persist between frames to prevent flickering
    face_locations = []
    face_names = []

    print("✅ System Live (press 'q' to exit)")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. YOLO Detection (Run every frame for smooth tracking)
        results = model(frame, stream=True, conf=CONFIDENCE_THRESHOLD, verbose=False)
        annotated_frame = frame.copy() 

        # 2. FACE Recognition Logic (Runs every other frame for performance)
        if process_this_frame:
            # Resize frame to 1/4 size for 4x faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
            # Convert BGR to RGB
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Detect all faces in the current frame
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                name = "Intruder" 

                if len(known_face_encodings) > 0:
                    # 1. Check for a basic match
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
                    
                    # 2. Calculate which known face is the absolute closest
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                face_names.append(name)

                # 3. Snapshot Logic (Only for Intruders)
                current_time = time.time()
                if name == "Intruder" and (current_time - last_capture_time > COOLDOWN):
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{SNAPSHOT_DIR}/intruder_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print("🚨 ALERT: Intruder captured:", filename)
                    last_capture_time = current_time

        # Toggle processing to skip every other frame
        process_this_frame = not process_this_frame

        # 4. DRAWING YOLO BOXES (For weapons/objects)
        for r in results:
            for box in r.boxes:
                label = model.names[int(box.cls)]
                if label in ["gun", "knife"]: 
                    b = box.xyxy[0].cpu().numpy() 
                    cv2.rectangle(annotated_frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 3)
                    cv2.putText(annotated_frame, f"THREAT: {label}", (int(b[0]), int(b[1] - 15)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 5. DRAWING FACE BOXES
        # This now uses the persistent face_locations/names so the box stays visible
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale coordinates back up by 4 (since we used 0.25 Scale Factor)
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Green for YOU, Red for Intruders
            color = (0, 255, 0) if name != "Intruder" else (0, 0, 255)
            
            # Draw Face Bounding Box
            cv2.rectangle(annotated_frame, (left, top), (right, bottom), color, 2)
            
            # Draw Label Background
            cv2.rectangle(annotated_frame, (left, top - 35), (right, top), color, cv2.FILLED)
            
            # Draw Name Text
            cv2.putText(annotated_frame, name, (left + 6, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display the resulting image
        cv2.imshow("AI Surveillance System - Local Detector", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_local_detector()