"""
╔══════════════════════════════════════════════════════════════╗
║              STEP 3 — ASL SPELLER                            ║
╚══════════════════════════════════════════════════════════════╝

Automatically loads whichever model won the comparison.
You can also override with:  python asl_speller.py --model rf
                              python asl_speller.py --model nn

Controls:
  Hold sign  1.5s  → confirm letter
  No hand    2.0s  → add SPACE
  D key            → delete last character
  C key            → clear sentence
  M key            → switch between RF and NN live
  Q key            → quit
"""

import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
import sys
from collections import Counter

# ── Load model info ────────────────────────────────────────────────────────────
try:
    with open("best_model_info.pkl", "rb") as f:
        info = pickle.load(f)
    CLASSES = info["classes"]
    default_winner = info["winner"]
except FileNotFoundError:
    print("ERROR: best_model_info.pkl not found. Run train_models.py first!")
    exit(1)

# ── Parse optional --model argument ───────────────────────────────────────────
force_model = None
if "--model" in sys.argv:
    idx = sys.argv.index("--model")
    if idx + 1 < len(sys.argv):
        force_model = sys.argv[idx + 1].lower()  # "rf" or "nn"

active_model_type = force_model or default_winner

# ── Load Random Forest ─────────────────────────────────────────────────────────
try:
    with open("model_rf.pkl", "rb") as f:
        rf_saved = pickle.load(f)
    rf_clf = rf_saved["model"]
    rf_loaded = True
except FileNotFoundError:
    rf_clf = None
    rf_loaded = False

# ── Load Neural Network ────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn

    class ASLNet(nn.Module):
        def __init__(self, input_size, num_classes):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_size, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, 128),        nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(128, 64),         nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(64, len(CLASSES)),
            )
        def forward(self, x):
            return self.network(x)

    nn_saved  = torch.load("model_nn.pt", map_location="cpu")
    nn_model  = ASLNet(nn_saved["input_size"], nn_saved["num_classes"])
    nn_model.load_state_dict(nn_saved["model_state"])
    nn_model.eval()
    nn_loaded = True
except Exception:
    nn_model  = None
    nn_loaded = False

if not rf_loaded and not nn_loaded:
    print("ERROR: No models found. Run train_models.py first!")
    exit(1)

# Fall back if chosen model isn't available
if active_model_type == "neural_network" and not nn_loaded:
    active_model_type = "random_forest"
if active_model_type == "random_forest" and not rf_loaded:
    active_model_type = "neural_network"

print(f"Active model: {active_model_type}")

# ── Prediction function ────────────────────────────────────────────────────────
def predict(features_np, model_type):
    """Returns (letter, confidence) using the specified model."""
    if model_type == "random_forest" and rf_loaded:
        proba = rf_clf.predict_proba(features_np)[0]
        idx   = np.argmax(proba)
        return CLASSES[idx], proba[idx]

    elif model_type == "neural_network" and nn_loaded:
        with torch.no_grad():
            t     = torch.tensor(features_np, dtype=torch.float32)
            out   = nn_model(t)
            proba = torch.softmax(out, dim=1).numpy()[0]
            idx   = np.argmax(proba)
        return CLASSES[idx], proba[idx]

    return None, 0.0


# ── Feature extraction ─────────────────────────────────────────────────────────
def extract_features(landmarks):
    wrist = landmarks[0]
    row   = []
    for lm in landmarks:
        row.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
    return np.array(row, dtype=np.float32).reshape(1, -1)


# ── Speller ────────────────────────────────────────────────────────────────────
class Speller:
    def __init__(self, hold_time=1.5, space_time=2.0):
        self.sentence      = ""
        self.last_letter   = None
        self.letter_start  = None
        self.confirmed     = False
        self.hold_time     = hold_time
        self.space_time    = space_time
        self.no_hand_start = None

    def update(self, letter):
        now = time.time()
        if letter is None:
            if self.no_hand_start is None:
                self.no_hand_start = now
            elif now - self.no_hand_start > self.space_time:
                if self.sentence and self.sentence[-1] != ' ':
                    self.sentence += ' '
                self.no_hand_start = None
            self.last_letter = None; self.letter_start = None; self.confirmed = False
            return
        self.no_hand_start = None
        if letter != self.last_letter:
            self.last_letter = letter; self.letter_start = now; self.confirmed = False
        elif not self.confirmed and (now - self.letter_start) >= self.hold_time:
            self.sentence += letter
            self.confirmed = True

    def progress(self):
        if self.last_letter is None or self.confirmed:
            return 0.0
        return min((time.time() - self.letter_start) / self.hold_time, 1.0)

    def delete(self):  self.sentence = self.sentence[:-1]
    def clear(self):   self.sentence = ""


# ── MediaPipe Tasks setup ──────────────────────────────────────────────────────
MODEL_PATH = "hand_landmarker.task"
BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

lm_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.5,
)

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)
]

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    global active_model_type
    speller     = Speller(hold_time=1.5, space_time=2.0)
    pred_buffer = []
    SMOOTH_N    = 8
    cap         = cv2.VideoCapture(0)

    with HandLandmarker.create_from_options(lm_options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            result    = landmarker.detect_for_video(mp_image, timestamp)

            raw_letter = None
            confidence = 0.0

            if result.hand_landmarks:
                for landmarks in result.hand_landmarks:
                    # Draw skeleton
                    for s, e in CONNECTIONS:
                        p1, p2 = landmarks[s], landmarks[e]
                        cv2.line(frame,
                                 (int(p1.x*w), int(p1.y*h)),
                                 (int(p2.x*w), int(p2.y*h)),
                                 (160, 160, 160), 2)
                    for pt in landmarks:
                        cv2.circle(frame, (int(pt.x*w), int(pt.y*h)), 5, (255,255,255), -1)

                    features   = extract_features(landmarks)
                    raw_letter, confidence = predict(features, active_model_type)
                    if confidence < 0.45:
                        raw_letter = None

            # Smooth predictions across last N frames
            pred_buffer.append(raw_letter)
            if len(pred_buffer) > SMOOTH_N:
                pred_buffer.pop(0)
            valid  = [p for p in pred_buffer if p is not None]
            letter = Counter(valid).most_common(1)[0][0] if valid else None

            speller.update(letter)

            # ── HUD ───────────────────────────────────────────────────────────
            cv2.rectangle(frame, (0, h-150), (w, h), (15,15,15), -1)

            # Current letter (big)
            sign_text = letter if letter else '–'
            cv2.putText(frame, sign_text, (18, h-85),
                        cv2.FONT_HERSHEY_SIMPLEX, 3.2, (0,255,180), 5, cv2.LINE_AA)

            # Model label
            model_label = "RF 🌲" if active_model_type == "random_forest" else "NN 🧠"
            cv2.putText(frame, model_label, (w-110, h-110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,0), 2)

            # Confidence bar
            if confidence > 0 and raw_letter:
                bar_x, bar_y, bar_w, bar_h = 110, h-75, w-130, 16
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (50,50,50), -1)
                col = (0,255,180) if confidence > 0.75 else (0,200,255) if confidence > 0.55 else (0,120,255)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x+int(bar_w*confidence), bar_y+bar_h), col, -1)
                cv2.putText(frame, f"conf {confidence:.0%}", (bar_x, bar_y-4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150,150,150), 1)

            # Hold progress bar
            prog = speller.progress()
            if prog > 0:
                bar_x, bar_y, bar_w, bar_h = 110, h-50, w-130, 16
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (50,50,50), -1)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x+int(bar_w*prog), bar_y+bar_h), (255,200,0), -1)
                cv2.putText(frame, "hold...", (bar_x, bar_y-4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,200,0), 1)

            # Sentence
            display = speller.sentence if speller.sentence else '...'
            cv2.putText(frame, display, (18, h-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255), 2, cv2.LINE_AA)

            # Top hint bar
            cv2.rectangle(frame, (0,0), (w,26), (15,15,15), -1)
            cv2.putText(frame,
                        "Hold 1.5s=letter | Pause 2s=SPACE | D=del | C=clear | M=switch model | Q=quit",
                        (8,18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140,140,140), 1)

            cv2.imshow("ASL Speller", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                speller.delete()
            elif key == ord('c'):
                speller.clear()
            elif key == ord('m'):
                # Switch between models live
                if active_model_type == "random_forest" and nn_loaded:
                    active_model_type = "neural_network"
                elif active_model_type == "neural_network" and rf_loaded:
                    active_model_type = "random_forest"
                print(f"Switched to: {active_model_type}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()