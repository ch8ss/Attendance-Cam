"""
enroll.py — Enroll a new student.

Usage:
    python enroll.py

Prompts for the kid's name + class, opens the webcam, captures several photos,
computes an averaged face encoding, generates a "Good morning, NAME" audio file
using pyttsx3, and saves everything to the database.

Controls during capture:
  SPACE — capture current frame as a sample
  Q     — quit early (need at least 3 samples to save)
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import face_recognition
import pyttsx3

import db


AUDIO_DIR = Path(__file__).parent / "audio"
AUDIO_DIR.mkdir(exist_ok=True)

SAMPLES_TARGET = 5  # how many photos we try to capture per kid


def generate_greeting_audio(name: str, out_path: Path):
    """Generate a 'Good morning, NAME' WAV file using offline TTS."""
    engine = pyttsx3.init()
    engine.setProperty("rate", 160)  # words per minute
    engine.save_to_file(f"Good morning, {name}", str(out_path))
    engine.runAndWait()


def capture_face_samples(target_count: int = SAMPLES_TARGET):
    """
    Open the webcam and let the user capture face samples.
    Returns a list of 128-D face encodings.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: could not open webcam.")
        sys.exit(1)

    encodings = []
    print(f"\nWebcam window will open. Press SPACE to capture a sample "
          f"(target: {target_count}). Press Q to finish early.\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read from webcam.")
            break

        # face_recognition expects RGB; OpenCV gives BGR.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb, model="hog")

        # Draw boxes for visual feedback.
        display = frame.copy()
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(display, (left, top), (right, bottom), (0, 255, 0), 2)

        status = f"Samples: {len(encodings)}/{target_count}  |  Faces: {len(face_locations)}"
        cv2.putText(display, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, "SPACE = capture, Q = done", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        cv2.imshow("Enrollment — face the camera", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            if len(face_locations) != 1:
                print(f"  Skipped: need exactly 1 face in frame, "
                      f"saw {len(face_locations)}.")
                continue
            sample_encs = face_recognition.face_encodings(rgb, face_locations)
            if sample_encs:
                encodings.append(sample_encs[0])
                print(f"  Captured sample {len(encodings)}/{target_count}")
            if len(encodings) >= target_count:
                break

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return encodings


def main():
    db.init_db()

    print("=== New Student Enrollment ===")
    name = input("Student name: ").strip()
    if not name:
        print("Name cannot be empty.")
        return
    class_name = input("Class (e.g. 'Nursery', 'KG-1', 'KG-2'): ").strip()
    if not class_name:
        print("Class cannot be empty.")
        return

    encodings = capture_face_samples()
    if len(encodings) < 3:
        print(f"\nOnly captured {len(encodings)} samples; need at least 3. "
              f"Aborting.")
        return

    # Average all samples into a single, more robust encoding.
    avg_encoding = np.mean(encodings, axis=0)

    # Make a filesystem-safe filename for the audio file.
    safe_name = "".join(c if c.isalnum() else "_" for c in name).strip("_")
    audio_path = AUDIO_DIR / f"{safe_name}_{int(time.time())}.wav"

    print(f"\nGenerating greeting audio at {audio_path.name}...")
    generate_greeting_audio(name, audio_path)

    student_id = db.add_student(name, class_name, avg_encoding, audio_path)
    print(f"\n✓ Enrolled '{name}' (class: {class_name}) with id={student_id}.")
    print(f"  {len(encodings)} face samples averaged.")
    print(f"  Audio file: {audio_path}")


if __name__ == "__main__":
    main()
