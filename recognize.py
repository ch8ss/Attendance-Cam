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
import face_recognition

import db


# --- Tunable thresholds ---
# Lower = stricter match. 0.6 is the library default. 0.5 is safer for kids
# (fewer false positives, occasional miss is OK — teacher can fix on dashboard).
MATCH_THRESHOLD = 0.5

# A face must occupy at least this fraction of the frame width to count.
# This is what enforces "stand in the square" — far-away background faces
# are too small and get ignored.
MIN_FACE_WIDTH_FRACTION = 0.20

# The face must be roughly centered horizontally — within this fraction
# of the frame center.
CENTER_TOLERANCE_FRACTION = 0.30

# Require this many consecutive frames of the same match before greeting.
# Prevents single-frame false positives from triggering greetings.
CONFIRM_FRAMES = 5


def play_audio(path: Path):
    """Play a WAV file in the background. Mac uses 'afplay'; Linux uses 'aplay'."""
    def _run():
        try:
            if platform.system() == "Darwin":
                subprocess.run(["afplay", str(path)], check=False)
            else:
                # Linux (Raspberry Pi)
                subprocess.run(["aplay", str(path)], check=False)
        except Exception as e:
            print(f"  [audio error: {e}]")
    threading.Thread(target=_run, daemon=True).start()


def is_in_square(face_loc, frame_shape):
    """
    Given a face bounding box and the frame shape, decide whether the face
    is "in the square" — i.e. large enough and roughly centered.
    """
    top, right, bottom, left = face_loc
    fh, fw = frame_shape[:2]

    face_w = right - left
    if face_w / fw < MIN_FACE_WIDTH_FRACTION:
        return False

    face_cx = (left + right) / 2
    frame_cx = fw / 2
    if abs(face_cx - frame_cx) / fw > CENTER_TOLERANCE_FRACTION:
        return False

    return True


def pick_primary_face(face_locations, frame_shape):
    """
    Of all faces detected, pick the one we should consider for recognition:
    the largest face that's in the square. Returns its index, or None.
    """
    fw = frame_shape[1]
    candidates = []
    for i, loc in enumerate(face_locations):
        if is_in_square(loc, frame_shape):
            top, right, bottom, left = loc
            area = (right - left) * (bottom - top)
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

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: could not open webcam.")
        sys.exit(1)

    # State for the "confirm over multiple frames" logic.
    last_match_id = None
    consecutive_count = 0
    last_greeted = {}  # student_id -> timestamp, for a short re-greet cooldown

    print("\nAttendance loop running. Press Q in the window to quit.\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue

        # Downscale for speed; recognize on small frame, draw on full.
        small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small, model="hog")
        primary_idx = pick_primary_face(face_locations, rgb_small.shape)

        display = frame.copy()

        # Draw the "square" — a centered rectangle showing where to stand.
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
            # Get encoding for the primary face only.
            encs = face_recognition.face_encodings(
                rgb_small, [face_locations[primary_idx]]
            )
            if encs:
                enc = encs[0]
                distances = face_recognition.face_distance(known_encodings, enc)
                best_idx = int(np.argmin(distances))
                best_dist = float(distances[best_idx])

                # Draw box on full-size frame (scale coords back up).
                top, right, bottom, left = face_locations[primary_idx]
                top, right, bottom, left = top*2, right*2, bottom*2, left*2

                if best_dist < MATCH_THRESHOLD:
                    matched = students[best_idx]
                    matched_id = matched["id"]
                    color = (0, 255, 0)
                    status_text = f"{matched['name']}  ({best_dist:.2f})"
                    status_color = color

                    # Confirm over multiple consecutive frames.
                    if matched_id == last_match_id:
                        consecutive_count += 1
                    else:
                        last_match_id = matched_id
                        consecutive_count = 1

                    if consecutive_count == CONFIRM_FRAMES:
                        # Avoid re-greeting in a tight loop (within 30s).
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
                    # Face seen, but not confidently recognized.
                    color = (0, 0, 255)
                    status_text = f"Unknown  ({best_dist:.2f})"
                    status_color = color
                    last_match_id = None
                    consecutive_count = 0

                cv2.rectangle(display, (left, top), (right, bottom), color, 2)
        else:
            # No face in the square.
            last_match_id = None
            consecutive_count = 0

        cv2.putText(display, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        cv2.imshow("Attendance — press Q to quit", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
