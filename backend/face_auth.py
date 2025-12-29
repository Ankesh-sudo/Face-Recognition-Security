import cv2
import face_recognition
import pickle
import os
import numpy as np

# 🔹 Project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 🔹 Encodings path
ENCODINGS_PATH = os.path.join(BASE_DIR, "data", "encodings", "face_encodings.pkl")

# Load encodings
if not os.path.exists(ENCODINGS_PATH):
    print("❌ No encodings found. Run face_encode.py first.")
    exit()

with open(ENCODINGS_PATH, "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]

print("✅ Encodings loaded. Starting camera...")

# Start camera
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("❌ Camera not accessible")
    exit()

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Resize for speed
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # BGR → RGB
    rgb_small_frame = small_frame[:, :, ::-1]

    # ✅ CRITICAL FIX: make array C-contiguous for dlib
    rgb_small_frame = np.ascontiguousarray(rgb_small_frame, dtype=np.uint8)

    # Face detection
    face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")

    # Face encoding (SAFE call)
    face_encodings = face_recognition.face_encodings(
        rgb_small_frame,
        known_face_locations=face_locations,
        num_jitters=0
    )

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(
            known_encodings, face_encoding, tolerance=0.5
        )

        name = "UNAUTHORIZED"

        if True in matches:
            matched_index = matches.index(True)
            name = known_names[matched_index]

        # Scale back coordinates
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        color = (0, 255, 0) if name != "UNAUTHORIZED" else (0, 0, 255)

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(
            frame,
            name,
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2
        )

    cv2.imshow("Face Authentication", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
