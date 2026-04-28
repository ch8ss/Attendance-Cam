# Preschool Attendance System — Phase 1 (Laptop Prototype)

A face-recognition attendance system. Kid stands in a square in front of the
camera, system recognizes them, plays "Good morning, NAME", marks them present
in a database. Teacher views/edits attendance via a small web dashboard.

This is **Phase 1**: it runs entirely on your laptop using its built-in webcam
and speakers. Once it works, we'll deploy it to a Raspberry Pi at the school.

---

## One-time setup (Mac, ~5 minutes)

You need Python 3.10 or 3.11 (not 3.12+). Check yours:

```bash
python3 --version
```

### 1. Install system dependency for `dlib`

```bash
brew install cmake
```

(If you don't have Homebrew: install it from https://brew.sh first.)

### 2. Create a virtual environment for the project

```bash
cd attendance_system
python3 -m venv venv
source venv/bin/activate
```

You should now see `(venv)` at the start of your terminal prompt. Any time
you come back to work on this project, run `source venv/bin/activate` again.

### 3. Install Python libraries

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

The `face_recognition` install can take 2-5 minutes — it compiles `dlib` from
source. Be patient. If it fails, the most common cause is a missing `cmake`
(see step 1).

### 4. Initialize the database

```bash
python db.py
```

You should see `Database ready at: .../attendance.db`.

### 5. Grant camera + microphone permissions (Mac)

The first time you run a script that opens the webcam, macOS will pop up a
permission dialog. Allow Terminal (or your IDE) to access the camera.
You may need to do the same for microphone if you hear no sound.

---

## Daily usage

Always activate the venv first:
```bash
source venv/bin/activate
```

### Enroll students (one-time per kid)

```bash
python enroll.py
```

Enter their name and class. The webcam will open. Stand in front of it (or
have the kid stand), and press **SPACE** to capture each sample. Try to get 5
samples with slight variations — turn your head a tiny bit, change expression
slightly. Press **Q** when done.

The system needs at least 3 samples to save the student.

Re-run for each kid. To test: enroll yourself first.

### Run the attendance loop

In the morning, run:
```bash
python recognize.py
```

A window opens showing the camera feed with a yellow square in the center.
Have someone enrolled stand so their face fills the square. After ~5 confirmed
frames, the system marks them present and plays the greeting. Press **Q** to quit.

### View the attendance register

In a separate terminal:
```bash
source venv/bin/activate
python dashboard.py
```

Open http://localhost:5000 in your browser. You'll see today's attendance
grouped by class. You can:
- Pick a different date
- Manually mark a student present or absent (overrides the camera)
- Click "Flag remaining as absent" at the end of the morning to mark everyone
  who hasn't shown up as absent for the day

To view the dashboard from your phone (same WiFi), find your laptop's IP:
```bash
ipconfig getifaddr en0
```
Then open `http://THAT_IP:5000` on your phone.

---

## File overview

```
attendance_system/
├── requirements.txt   # Python dependencies
├── db.py              # SQLite schema + helpers
├── enroll.py          # Add new students
├── recognize.py       # Main attendance loop (camera)
├── dashboard.py       # Web register (Flask)
├── attendance.db      # Created on first run — the database
└── audio/             # Created on first enroll — greeting WAV files
```

## Tunable settings (in `recognize.py`)

- `MATCH_THRESHOLD` (default 0.5) — lower = stricter, fewer false positives
- `MIN_FACE_WIDTH_FRACTION` (default 0.20) — how close the kid must be
- `CONFIRM_FRAMES` (default 5) — how many frames to confirm before greeting

## Troubleshooting

**`face_recognition` install fails:** install `cmake` (`brew install cmake`),
make sure you're on Python 3.10/3.11, and try again.

**No webcam window opens:** check macOS Privacy & Security → Camera, allow
Terminal.

**No greeting audio plays:** test pyttsx3 separately:
```python
import pyttsx3; e = pyttsx3.init(); e.say("hello"); e.runAndWait()
```
On Mac, pyttsx3 uses the built-in `say` command — should just work.

**Recognition is bad:** re-enroll with more samples, in similar lighting to
where the system will run. Lower `MATCH_THRESHOLD` to 0.45 if you're getting
false matches; raise it to 0.55 if real students aren't being recognized.

---

## What's NOT in Phase 1 (intentionally)

- Auto-flagging absences at a specific time (manual button on the dashboard
  for now; we'll add a cron job once running on the Pi)
- Multi-camera support (only need one camera per the design)
- Authentication on the dashboard (assumed to run on a trusted local network)
- Re-enrollment / photo updates flow (delete + re-enroll for now)

These are all easy additions for Phase 2 once the core works.
