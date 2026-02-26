# ASL Speller

A real-time American Sign Language (ASL) fingerspelling program that uses your webcam to detect hand signs and spell out words letter by letter. Built with MediaPipe, OpenCV, scikit-learn, and PyTorch.

---

## What it does

- Tracks your hand in real time using MediaPipe
- Recognizes ASL hand signs for letters A through Y (J and Z, for now, are excluded as they require motion)
- Spells out words on screen as you sign each letter
- Trains two machine learning models on your own hand data and compares their accuracy:
  - Random Forest (scikit-learn)
  - Neural Network (PyTorch)

---

## Requirements

- Python 3.9 to 3.12
- A working webcam
- Windows (the setup script is a .bat file)

---

## Setup

1. Clone the repository
   ```
   git clone https://github.com/AkIyAm4/ASL-Speller.git
   cd ASL-Speller
   ```

2. Double-click `setup.bat`

   This will automatically:
   - Create a virtual environment
   - Install all required packages
   - Download the MediaPipe hand landmark model

That is all. No manual pip installs needed.

---

## How to use

Run the scripts in this order:

**Step 1 — Collect your own hand sign data**
```
python collect_data.py
```
Hold each letter sign in front of your webcam and press SPACE to record. Aim for around 150 samples per letter. You can stop and resume at any time — progress is saved automatically.

**Step 2 — Train both models and compare them**
```
python train_models.py
```
This trains a Random Forest and a Neural Network on your collected data, prints a comparison of their accuracy, and saves both models. The better one is automatically selected for the speller.

**Step 3 — Run the speller**
```
python asl_speller.py
```

### Controls

| Key            | Action                        |
|----------------|-------------------------------|
| Hold sign 1.5s | Confirm the letter            |
| No hand 2s     | Add a space                   |
| D              | Delete last character         |
| C              | Clear the whole sentence      |
| M              | Switch between RF and NN live |
| Q              | Quit                          |

---

## Project structure

```
ASL-Speller/
├── setup.bat            - Auto setup script
├── collect_data.py      - Webcam data collection
├── train_models.py      - Train and compare ML models
├── asl_speller.py       - Main speller application
├── requirements.txt     - Package list
└── .gitignore
```

---

## Notes

- The models are trained on your hands specifically, which gives better accuracy than a generic pre-trained model
- J and Z are not supported because they require drawing a motion path in the air rather than holding a static pose
- If accuracy feels off for certain letters, collect more samples for those letters and retrain

---

## Credits

The original hand tracking code that this project is built on was written by [Naourr](https://github.com/Naourr).
You can find the original repository here: [Hand-Gestures-thang](https://github.com/Naourr/Hand-Gestures-thang)

---

## Built with

- [MediaPipe](https://developers.google.com/mediapipe) - Hand landmark detection
- [OpenCV](https://opencv.org/) - Webcam and drawing
- [scikit-learn](https://scikit-learn.org/) - Random Forest
- [PyTorch](https://pytorch.org/) - Neural Network
