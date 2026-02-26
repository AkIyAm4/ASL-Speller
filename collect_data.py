"""
╔══════════════════════════════════════════════════════════════╗
║           STEP 1 — COLLECT YOUR OWN HAND SIGN DATA          ║
╚══════════════════════════════════════════════════════════════╝

What this script does:
  - Opens your webcam
  - For each letter (A–Y, skipping J and Z), it asks you to
    hold the sign while it records landmark data
  - Saves everything to a file called "asl_data.pkl"

Why we collect landmarks instead of raw images:
  - MediaPipe gives us 21 (x, y, z) points per hand = 63 numbers
  - These 63 numbers are WAY smaller than a full image (which can
    be 200x200x3 = 120,000 numbers)
  - Smaller input = faster training = easier for a beginner model
  - Also works across different lighting conditions and skin tones

Controls during collection:
  SPACE  → start/stop recording for the current letter
  N      → skip to next letter
  Q      → quit and save what you have so far

How many samples to collect:
  - Aim for 100–200 per letter for decent accuracy
  - More = better, but 100 is enough to start
"""

import cv2
import mediapipe as mp
import pickle
import os

# ── MediaPipe Tasks setup (same as your working code) ─────────────────────────
model_path = 'hand_landmarker.task'
BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.5,
)

# ── Skeleton connections (same as your working code) ──────────────────────────
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

# ── Letters we'll collect ──────────────────────────────────────────────────────
# J and Z require motion (drawing in the air) so we skip them
LETTERS = [l for l in "ABCDEFGHIKLMNOPQRSTUVWXY"]  # 24 letters
SAMPLES_PER_LETTER = 150

# ── Feature extraction ─────────────────────────────────────────────────────────
def extract_features(landmarks):
    """
    Convert 21 landmarks into 63 numbers, normalized to the wrist.
    Subtracting the wrist means the model learns the SHAPE of the sign,
    not where the hand is on screen.
    """
    wrist = landmarks[0]
    features = []
    for lm in landmarks:
        features.extend([
            lm.x - wrist.x,
            lm.y - wrist.y,
            lm.z - wrist.z,
        ])
    return features


# ── Load existing data if resuming ────────────────────────────────────────────
DATA_FILE = "asl_data.pkl"
if os.path.exists(DATA_FILE):
    with open(DATA_FILE, "rb") as f:
        saved = pickle.load(f)
    all_data   = saved["data"]
    all_labels = saved["labels"]
    print(f"Resuming — loaded {len(all_data)} existing samples.")
else:
    all_data   = []
    all_labels = []
    print("Starting fresh data collection.")


# ── Main collection loop ───────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:
    for letter_idx, letter in enumerate(LETTERS):

        existing = all_labels.count(letter)
        if existing >= SAMPLES_PER_LETTER:
            print(f"  [{letter}] Already have {existing} samples, skipping.")
            continue

        needed    = SAMPLES_PER_LETTER - existing
        collected = 0
        recording = False

        print(f"\n── Letter {letter} ({letter_idx+1}/{len(LETTERS)}) ──")
        print(f"   Need {needed} more samples. Press SPACE to start recording.")

        while collected < needed:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            # Same detection flow as your working code
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            result    = landmarker.detect_for_video(mp_image, timestamp)

            hand_found = False

            if result.hand_landmarks:
                hand_found = True
                for landmarks in result.hand_landmarks:
                    # Draw skeleton exactly like your working code
                    for connection in CONNECTIONS:
                        start_idx, end_idx = connection
                        p1 = landmarks[start_idx]
                        p2 = landmarks[end_idx]
                        x1, y1 = int(p1.x * w), int(p1.y * h)
                        x2, y2 = int(p2.x * w), int(p2.y * h)
                        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    for point in landmarks:
                        x, y = int(point.x * w), int(point.y * h)
                        cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)

                    # Save features if recording
                    if recording:
                        features = extract_features(landmarks)
                        all_data.append(features)
                        all_labels.append(letter)
                        collected += 1

            # ── Draw UI ───────────────────────────────────────────────────────
            cv2.rectangle(frame, (0, 0), (w, 90), (20, 20, 20), -1)

            cv2.putText(frame, f"Sign: {letter}", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 180), 2)
            cv2.putText(frame, f"Collected: {collected}/{needed}  (total existing: {existing})",
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

            if recording:
                cv2.circle(frame, (w - 30, 30), 15, (0, 0, 255), -1)
                cv2.putText(frame, "REC", (w - 70, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                status = "SPACE to record" if hand_found else "Show your hand!"
                cv2.putText(frame, status, (w - 220, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

            # Progress bar at bottom
            cv2.rectangle(frame, (0, h - 18), (w, h), (30, 30, 30), -1)
            prog_w = int(w * (collected / needed))
            cv2.rectangle(frame, (0, h - 18), (prog_w, h), (0, 255, 180), -1)
            cv2.putText(frame, "SPACE=record  N=next letter  Q=quit",
                        (10, h - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

            cv2.imshow("ASL Data Collection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                recording = not recording
                print(f"   Recording {'ON' if recording else 'OFF'}")
            elif key == ord('n'):
                print(f"   Skipped {letter}")
                break
            elif key == ord('q'):
                print("\nSaving and quitting...")
                with open(DATA_FILE, "wb") as f:
                    pickle.dump({"data": all_data, "labels": all_labels}, f)
                cap.release()
                cv2.destroyAllWindows()
                exit(0)

        print(f"   Done with {letter}: collected {collected} new samples")

# ── Save everything ────────────────────────────────────────────────────────────
with open(DATA_FILE, "wb") as f:
    pickle.dump({"data": all_data, "labels": all_labels}, f)

cap.release()
cv2.destroyAllWindows()

# ── Summary ────────────────────────────────────────────────────────────────────
from collections import Counter
counts = Counter(all_labels)
print("\n╔══ Data Collection Complete ══╗")
print(f"  Total samples : {len(all_data)}")
print(f"  Total classes : {len(counts)}")
print(f"  Samples/letter: {dict(sorted(counts.items()))}")
print(f"\n  Saved to: {DATA_FILE}")
print("  Next step: run python train_models.py")
