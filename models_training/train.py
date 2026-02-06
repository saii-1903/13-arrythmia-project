
import os
import sys
import json
import psycopg2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from pathlib import Path
from collections import Counter

# Fix path to imports
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

try:
    from models_training.data_loader import (
        get_rhythm_label_idx, RHYTHM_CLASS_NAMES, ECTOPY_CLASS_NAMES, 
        get_ectopy_label_idx, extract_fixed_window, WINDOW_SEC
    )
    from models_training.models import CNNTransformerClassifier
except ImportError:
    # Fallback if running from within models_training
    from data_loader import (
        get_rhythm_label_idx, RHYTHM_CLASS_NAMES, ECTOPY_CLASS_NAMES, 
        get_ectopy_label_idx, extract_fixed_window, WINDOW_SEC
    )
    from models import CNNTransformerClassifier

# =============================================================================
# FOCAL LOSS - Handles extreme imbalance (Step 3)
# =============================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# =============================================================================
# STEP 1 & 2: EVENT-BASED DATASET & WINDOW NORMALIZATION
# =============================================================================
class EventDataset(Dataset):
    def __init__(self, task="rhythm"):
        self.X = []
        self.y = []
        self.task = task
        
        print(f"🚀 Loading Event-Based Dataset for {task.upper()}...")
        
        # Connection
        conn = psycopg2.connect(
            dbname="ecg_analysis", 
            user="ecg_user", 
            password="sais", 
            host="127.0.0.1"
        )
        cur = conn.cursor()
        
        # STEP 1A: Query from optimized ecg_segments table
        cur.execute("SELECT signal, events_json, segment_fs FROM ecg_segments")
        rows = cur.fetchall()
        
        TARGET_FS = 250
        WINDOW_SAMPLES = int(TARGET_FS * 2) # STEP 2: 2 seconds fixed (500 samples)

        for row in rows:
            # STEP 1B: Python logic (Event extraction)
            signal_raw, events_raw, fs = row
            
            # Robust parsing
            if isinstance(signal_raw, str):
                signal = np.array(json.loads(signal_raw), dtype=np.float32)
            else:
                signal = np.array(signal_raw, dtype=np.float32)
                
            if isinstance(events_raw, str):
                events = json.loads(events_raw)
            else:
                events = events_raw or []

            for event in events:
                # 🔒 WORKSTREAM 3: ONLY CARDIOLOGIST EVENTS
                if (
                    event.get("annotation_source") == "cardiologist"
                    and event.get("annotation_status") == "confirmed"
                ):
                    # 🔒 WORKSTREAM 1: MAKE WINDOWING EXPLICIT (MANDATORY)
                    window = extract_fixed_window(
                        signal,
                        fs,
                        event["start_time"],
                        event["end_time"]
                    )
                    
                    # 🔍 VERIFICATION (MANDATORY)
                    assert window.shape[0] == int(WINDOW_SEC * fs), f"Window mismatch: {window.shape[0]} != {int(WINDOW_SEC * fs)}"
                
                    # 🚨 Mandatory: Use event_type, NEVER pattern_label (as per instructions)
                    etype = event["event_type"]
                    
                    if task == "rhythm":
                        l_idx = get_rhythm_label_idx(etype)
                    else:
                        l_idx = get_ectopy_label_idx(etype)
                    
                    if l_idx is not None:
                        self.X.append(window.astype(np.float32))
                        self.y.append(l_idx)
        
        conn.close()
        print(f"✅ Successfully loaded {len(self.X)} events.")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Return (B, 1, L) for CNN
        return torch.from_numpy(self.X[idx]).unsqueeze(0), torch.tensor(self.y[idx], dtype=torch.long)

# =============================================================================
# STEP 3: CLASS BALANCING & TRAINING LOOP
# =============================================================================
def run_training(task="rhythm", epochs=10):
    dataset = EventDataset(task=task)
    if len(dataset) == 0:
        print("⛔ No events found for training. Did you run the migration?")
        return

    # STEP 3: CLASS BALANCING (WeightedRandomSampler)
    labels = dataset.y
    counts = Counter(labels)
    num_classes = len(RHYTHM_CLASS_NAMES if task=="rhythm" else ECTOPY_CLASS_NAMES)
    
    # Calculate inverse frequencies for sampling
    class_weights = []
    for i in range(num_classes):
        c = counts.get(i, 0)
        class_weights.append(1.0 / c if c > 0 else 0.0)
    
    sample_weights = [class_weights[l] for l in labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(labels) * 2, replacement=True)
    
    loader = DataLoader(dataset, batch_size=32, sampler=sampler)
    
    # Initialize Model
    model = CNNTransformerClassifier(num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Loss & Optimizer
    # Smooth the alpha for Focal Loss to prevent exploding gradients on rare classes
    alpha = torch.tensor(np.sqrt(class_weights), dtype=torch.float32).to(device)
    criterion = FocalLoss(alpha=alpha)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    print(f"\n🚀 Starting Training on {device}...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        correct = 0
        
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            
        avg_loss = total_loss / len(loader)
        acc = correct / (len(labels) * 2) # Weighted sampler size
        print(f"Epoch {epoch:02d} | Loss: {avg_loss:.4f} | Est. Acc: {acc:.4f}")

    # Save
    save_path = f"best_model_{task}_event_based.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n⭐ Training Complete! Model saved to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="rhythm", choices=["rhythm", "ectopy"])
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    
    run_training(task=args.task, epochs=args.epochs)
