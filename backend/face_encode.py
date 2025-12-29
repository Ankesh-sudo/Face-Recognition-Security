import face_recognition
import os
import pickle

# Get project root (one level above backend)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

KNOWN_FACES_DIR = os.path.join(BASE_DIR, "data", "known_faces")
ENCODINGS_DIR = os.path.join(BASE_DIR, "data", "encodings")
ENCODINGS_FILE = os.path.join(ENCODINGS_DIR, "face_encodings.pkl")


def encode_faces():
    known_encodings = []
    known_names = []

    if not os.path.exists(KNOWN_FACES_DIR):
        print(f"❌ known_faces directory not found at {KNOWN_FACES_DIR}")
        return

    for user in os.listdir(KNOWN_FACES_DIR):
        user_path = os.path.join(KNOWN_FACES_DIR, user)

        if not os.path.isdir(user_path):
            continue

        print(f"🔍 Processing user: {user}")

        for image_name in os.listdir(user_path):
            image_path = os.path.join(user_path, image_name)

            try:
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)

                if encodings:
                    known_encodings.append(encodings[0])
                    known_names.append(user)
                else:
                    print(f"⚠️ No face found in {image_name}")

            except Exception as e:
                print(f"❌ Error processing {image_name}: {e}")

    if not known_encodings:
        print("❌ No encodings generated")
        return

    os.makedirs(ENCODINGS_DIR, exist_ok=True)

    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(
            {
                "encodings": known_encodings,
                "names": known_names
            },
            f
        )

    print("✅ Face encodings saved successfully")


if __name__ == "__main__":
    encode_faces()
