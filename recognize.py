"""
recognize.py — Main attendance loop.

Run this when school is starting. It opens the webcam and watches for a face
in the center of the frame (the "square"). When it confidently recognizes an
enrolled student, it plays their greeting and marks them present.

Controls:
  Q — quit
"""

import sys
import time
import threading
import subprocess
import platform
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

import db


# --- Tunable thresholds ---
# Cosine similarity: higher = more confident match (range 0–1).
# Raise to be stricter (fewer false positives); lower to catch more kids.
# 0.45 is a safe default for a small enrolled group.
MATCH_THRESHOLD = 0.45

# A face must occupy at least this fraction of the frame width to count.
MIN_FACE_WIDTH_FRACTION = 0.20

# The face must be roughly centered horizontally within this fraction of frame.
CENTER_TOLERANCE_FRACTION = 0.30

# Require this many consecutive frames of the same match before greeting.
CONFIRM_FRAMES = 5


def play_audio(path: Path):
    """Play a WAV file in the background. Mac uses 'afplay'; Linux uses 'aplay'."""
    def _run():
        try:
            if platform.system() == "Darwin":
                subprocess.run(["afplay", str(path)], check=False)
            else:
                subprocess.run(["aplay", str(path)], check=False)
        except Exception as e:
            print(f"  [audio error: {e}]")
    threading.Thread(target=_run, daemon=True).start()


def is_in_square(bbox, frame_shape):
    """
    bbox is [x1, y1, x2, y2]. Returns True if the face is large enough
    and roughly centered.
    """
    x1, y1, x2, y2 = bbox
    fh, fw = frame_shape[:2]

    face_w = x2 - x1
    if face_w / fw < MIN_FACE_WIDTH_FRACTION:
        return False

    face_cx = (x1 + x2) / 2
    if abs(face_cx - fw / 2) / fw > CENTER_TOLERANCE_FRACTION:
        return False

    return True


def pick_primary_face(faces, frame_shape):
    """
    Of all detected faces, return the index of the largest one that is in the
    square, or None if none qualify.
    """
    candidates = []
    for i, face in enumerate(faces):
        if face.embedding is None:
            continue
        if is_in_square(face.bbox, frame_shape):
            x1, y1, x2, y2 = face.bbox
            area = (x2 - x1) * (y2 - y1)
            candidates.append((area, i))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def main():
    db.init_db()
    students = db.get_all_students()
    if not students:
        print("No students enrolled yet. Run enroll.py first.")
        sys.exit(0)

    known_encodings = np.array([s["face_encoding"] for s in students])
    print(f"Loaded {len(students)} enrolled students.")

    print("Loading face recognition model (may download ~300 MB on first run)...")
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(320, 240))
    print("Model ready.\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: could not open webcam.")
        sys.exit(1)

    last_match_id = None
    consecutive_count = 0
    last_greeted = {}  # student_id -> timestamp

    print("Attendance loop running. Press Q in the window to quit.\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue

        # Downscale for speed; detect on small frame, draw on full.
        small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # InsightFace takes BGR directly — no RGB conversion needed.
        faces = app.get(small)
        primary_idx = pick_primary_face(faces, small.shape)

        display = frame.copy()

        # Draw the "stand here" square.
        fh, fw = display.shape[:2]
        sq_size = int(min(fh, fw) * 0.5)
        sq_x = (fw - sq_size) // 2
        sq_y = (fh - sq_size) // 2
        cv2.rectangle(display, (sq_x, sq_y), (sq_x + sq_size, sq_y + sq_size),
                      (255, 255, 0), 2)
        cv2.putText(display, "Stand here", (sq_x, sq_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        status_text = "Waiting..."
        status_color = (200, 200, 200)

        if primary_idx is not None:
            face = faces[primary_idx]
            enc = face.embedding  # 512-dim, L2-normalized

            # Cosine similarity via dot product (embeddings are L2-normalized).
            similarities = known_encodings @ enc
            best_idx = int(np.argmax(similarities))
            best_sim = float(similarities[best_idx])

            # Scale bbox back to full-frame coordinates.
            x1, y1, x2, y2 = (int(v * 2) for v in face.bbox)

            if best_sim > MATCH_THRESHOLD:
                matched = students[best_idx]
                matched_id = matched["id"]
                color = (0, 255, 0)
                status_text = f"{matched['name']}  ({best_sim:.2f})"
                status_color = color

                if matched_id == last_match_id:
                    consecutive_count += 1
                else:
                    last_match_id = matched_id
                    consecutive_count = 1

                if consecutive_count == CONFIRM_FRAMES:
                    last_t = last_greeted.get(matched_id, 0)
                    if time.time() - last_t > 30:
                        last_greeted[matched_id] = time.time()
                        if db.is_marked_today(matched_id):
                            print(f"  {matched['name']}: already marked today.")
                        else:
                            created = db.mark_present(matched_id)
                            if created:
                                print(f"  ✓ Marked present: {matched['name']} "
                                      f"({matched['class_name']})")
                                play_audio(Path(matched["audio_path"]))

            else:
                color = (0, 0, 255)
                status_text = f"Unknown  ({best_sim:.2f})"
                status_color = color
                last_match_id = None
                consecutive_count = 0

            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

        else:
            last_match_id = None
            consecutive_count = 0

        cv2.putText(display, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        cv2.imshow("Attendance — press Q to quit", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
