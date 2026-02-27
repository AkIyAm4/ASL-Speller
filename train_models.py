"""
╔══════════════════════════════════════════════════════════════╗
║         STEP 2 — TRAIN & COMPARE BOTH ML MODELS              ║
╚══════════════════════════════════════════════════════════════╝

What this script does:
  1. Loads your collected data (asl_data.pkl)
  2. Trains a Random Forest classifier
  3. Trains a Neural Network (using PyTorch)
  4. Compares their accuracy and saves both models
  5. Saves the better one as "best_model.pkl" for the speller

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Requirements:
  pip install scikit-learn torch numpy
"""

import pickle
import numpy as np
import time

# scikit-learn: a beginner-friendly ML library with many classic algorithms
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# PyTorch: the most popular deep learning library
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ══════════════════════════════════════════════════════════════════════════════
# STEP A — LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("Loading data...")
try:
    with open("asl_data.pkl", "rb") as f:
        saved = pickle.load(f)
except FileNotFoundError:
    print("ERROR: asl_data.pkl not found. Run collect_data.py first!")
    exit(1)

X = np.array(saved["data"],   dtype=np.float32)  # shape: (num_samples, 63)
y_raw = np.array(saved["labels"])                 # shape: (num_samples,)  e.g. ['A','A','B',...]

# ── Label encoding ─────────────────────────────────────────────────────────────
# Neural networks work with numbers, not strings.
# LabelEncoder converts: ['A','B','C',...] → [0, 1, 2, ...]
le = LabelEncoder()
y  = le.fit_transform(y_raw)   # y is now integers
CLASSES = list(le.classes_)    # e.g. ['A','B','C',...]
NUM_CLASSES = len(CLASSES)

print(f"  Samples   : {len(X)}")
print(f"  Features  : {X.shape[1]}  (21 landmarks × 3 coordinates)")
print(f"  Classes   : {NUM_CLASSES}  → {CLASSES}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP B — SPLIT INTO TRAINING SET AND TEST SET
# ══════════════════════════════════════════════════════════════════════════════
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% for testing
    random_state=42,     # "random seed" — makes the split reproducible
    stratify=y           # ensures each letter is proportionally represented
)

print(f"\n  Training samples : {len(X_train)}")
print(f"  Test samples     : {len(X_test)}")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 1 — RANDOM FOREST
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*55)
print("  MODEL 1: RANDOM FOREST")
print("═"*55)

rf_start = time.time()

rf_model = RandomForestClassifier(
    n_estimators=200,    # number of trees — more trees = better but slower
    max_depth=None,      # let trees grow as deep as needed
    random_state=42,
    n_jobs=-1            # use all CPU cores
)

print("  Training...")
rf_model.fit(X_train, y_train)

rf_time = time.time() - rf_start
rf_preds = rf_model.predict(X_test)
rf_acc   = accuracy_score(y_test, rf_preds)

print(f"  Training time : {rf_time:.1f}s")
print(f"  Test accuracy : {rf_acc:.2%}")
print("\n  Per-letter breakdown:")
print(classification_report(y_test, rf_preds, target_names=CLASSES))


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 2 — NEURAL NETWORK (PyTorch)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*55)
print("  MODEL 2: NEURAL NETWORK (PyTorch)")
print("═"*55)

# ── Device ────────────────────────────────────────────────────────────────────
# Use GPU if available (much faster), otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Using device: {device}")

# ── Convert numpy arrays to PyTorch Tensors ───────────────────────────────────
# PyTorch operates on "tensors" — like numpy arrays but GPU-compatible
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_test_t  = torch.tensor(y_test,  dtype=torch.long)

# DataLoader batches and shuffles data automatically each epoch
train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=32,
    shuffle=True    # shuffle each epoch so the model doesn't memorize order
)

# ── Define the Neural Network architecture ────────────────────────────────────
class ASLNet(nn.Module):
    """
    nn.Module is PyTorch's base class for all neural networks.
    We define our layers in __init__ and the forward pass in forward().
    """
    def __init__(self, input_size, num_classes):
        super(ASLNet, self).__init__()

        # nn.Sequential is a container that chains layers in order
        self.network = nn.Sequential(
            # Layer 1: 63 inputs → 256 neurons
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),         # normalize for stable training
            nn.ReLU(),                   # activation: f(x) = max(0, x)
            nn.Dropout(0.3),             # randomly drop 30% of neurons

            # Layer 2: 256 → 128
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Layer 3: 128 → 64
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Output layer: 64 → 24 (one score per letter)
            # No activation here — CrossEntropyLoss applies softmax internally
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        """Defines how data flows through the network."""
        return self.network(x)


# ── Initialize model, loss function, optimizer ───────────────────────────────
model     = ASLNet(input_size=X.shape[1], num_classes=NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()              # loss function for classification
optimizer = optim.Adam(model.parameters(), lr=0.001)   # Adam optimizer
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
# scheduler: halve the learning rate every 20 epochs (fine-tunes near the end)

NUM_EPOCHS = 60
nn_start = time.time()

print(f"  Training for {NUM_EPOCHS} epochs...\n")
print(f"  {'Epoch':<8} {'Loss':<12} {'Train Acc':<14} {'Test Acc'}")
print(f"  {'─'*5:<8} {'─'*8:<12} {'─'*9:<14} {'─'*8}")

best_test_acc  = 0.0
best_nn_state  = None

for epoch in range(1, NUM_EPOCHS + 1):
    # ── Training phase ────────────────────────────────────────────────────────
    model.train()   # enable dropout and batchnorm training behavior
    total_loss   = 0.0
    train_correct = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()           # clear gradients from last step
        outputs = model(X_batch)        # forward pass
        loss    = criterion(outputs, y_batch)   # compute loss
        loss.backward()                 # backpropagation: compute gradients
        optimizer.step()                # update weights

        total_loss    += loss.item() * len(X_batch)
        preds          = outputs.argmax(dim=1)
        train_correct += (preds == y_batch).sum().item()

    scheduler.step()   # adjust learning rate

    # ── Evaluation phase ──────────────────────────────────────────────────────
    model.eval()    # disable dropout for evaluation
    with torch.no_grad():   # don't track gradients (saves memory)
        test_out  = model(X_test_t.to(device))
        test_preds = test_out.argmax(dim=1).cpu()
        test_acc  = (test_preds == y_test_t).float().mean().item()

    avg_loss  = total_loss / len(X_train)
    train_acc = train_correct / len(X_train)

    # Save the best model weights seen so far
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_nn_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Print every 5 epochs
    if epoch % 5 == 0 or epoch == 1:
        print(f"  {epoch:<8} {avg_loss:<12.4f} {train_acc:<14.2%} {test_acc:.2%}"
              + (" ← best" if test_acc == best_test_acc else ""))

nn_time = time.time() - nn_start
print(f"\n  Training time  : {nn_time:.1f}s")
print(f"  Best test acc  : {best_test_acc:.2%}")

# Load best weights back into model
model.load_state_dict(best_nn_state)
model.eval()

# Detailed report
with torch.no_grad():
    nn_preds = model(X_test_t.to(device)).argmax(dim=1).cpu().numpy()
print("\n  Per-letter breakdown:")
print(classification_report(y_test, nn_preds, target_names=CLASSES))


# ══════════════════════════════════════════════════════════════════════════════
# STEP C — COMPARE AND SAVE
# ══════════════════════════════════════════════════════════════════════════════
print("═"*55)
print("  COMPARISON SUMMARY")
print("═"*55)
print(f"  Random Forest  : {rf_acc:.2%}  (trained in {rf_time:.1f}s)")
print(f"  Neural Network : {best_test_acc:.2%}  (trained in {nn_time:.1f}s)")

winner = "neural_network" if best_test_acc >= rf_acc else "random_forest"
print(f"\n  Winner: {'Neural Network 🧠' if winner == 'neural_network' else 'Random Forest 🌲'}")
print("═"*55)

# Save Random Forest
with open("model_rf.pkl", "wb") as f:
    pickle.dump({"model": rf_model, "classes": CLASSES, "label_encoder": le}, f)
print("\n  Saved: model_rf.pkl")

# Save Neural Network
torch.save({
    "model_state": best_nn_state,
    "classes": CLASSES,
    "input_size": X.shape[1],
    "num_classes": NUM_CLASSES,
}, "model_nn.pt")
print("  Saved: model_nn.pt")

# Save the winner as "best_model" for the speller to auto-load
with open("best_model_info.pkl", "wb") as f:
    pickle.dump({"winner": winner, "classes": CLASSES}, f)
print(f"  Saved: best_model_info.pkl  (winner = {winner})")


print("\n  Next step: run python asl_speller.py")

