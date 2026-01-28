
# Implementation Plan: Remove Sinus Rhythm from Retraining

## 1. Objective
Eliminate "Sinus Rhythm" and related sinus classes from the retraining pipeline (`retrain.py`). The model should strictly learn to differentiate between arrhythmias. Sinus detection will be delegated to the existing rule-based system (regularity, P-wave, HR bounds) or the frozen baseline model.

## 2. Core Constraints
*   **Sinus must be REMOVED**, not just down-weighted.
*   **Retraining Only**: This change applies to `retrain.py` and the models it produces. The inference pipeline (if it shares `data_loader.py`) must still validly handle Sinus appropriately (e.g. by not passing it to this model, or the system handling it before-hand).
*   **Label Space**: The model's output layer must resize to $N_{total} - N_{sinus}$.

## 3. Proposed Changes

### A. Modify `models_training/data_loader.py`

**Current State**:
`CLASS_NAMES` includes "Sinus Rhythm" (0), "Sinus Bradycardia" (1), "Sinus Tachycardia" (2).

**Change**:
Define a specific list for retraining that excludes these.

```python
# In data_loader.py

# ... existing CLASS_NAMES ...

# NEW: Strict Arrhythmia Classes for Retraining (No Sinus)
RETRAIN_CLASS_NAMES = [
    name for name in CLASS_NAMES 
    if "Sinus" not in name and name != "Artifact" # Optional: Check if Artifact should stay? Usually Artifact is also filtered or handled separately.
]

# Create a mapping from Original Name -> New Retrain Index
RETRAIN_CLASS_INDEX = {name: i for i, name in enumerate(RETRAIN_CLASS_NAMES)}

def get_retrain_label(original_label_name):
    """Returns new index (0..K) or None if class should be skipped."""
    if original_label_name in RETRAIN_CLASS_INDEX:
        return RETRAIN_CLASS_INDEX[original_label_name]
    return None
```
*Note: We assume "Sinus Bradycardia" and "Sinus Tachycardia" are also excluded as they are handled by HR bounds.*

### B. Modify `models_training/retrain.py`

**1. Update `ECGRawDatasetSQL` Query**
Filter out Sinus classes at the SQL level to save time transferring data.

```python
# In models_training/retrain.py -> ECGRawDatasetSQL.__init__

# Modify the WHERE clause
query = """
    SELECT segment_id, arrhythmia_label, raw_signal, patient_id, admission_id
    FROM ecg_features_annotatable
    WHERE raw_signal IS NOT NULL
      AND arrhythmia_label IS NOT NULL
      AND arrhythmia_label != 'Unlabeled'
      AND corrected_by IS NOT NULL 
      -- NEW FILTERS
      AND arrhythmia_label NOT IN ('Sinus Rhythm', 'Sinus Bradycardia', 'Sinus Tachycardia', 'SR', 'SB', 'ST')
      -- Alternatively, rely on Python-side filtering if normalization is complex
"""
```

**2. Update `ECGRawDatasetSQL` Processing Loop**
Ensure we only keep samples that map to our new label space.

```python
# Inside the processing loop
l_clean = normalize_label(label)

# Check if it belongs to our new separate Retraining list
if l_clean in RETRAIN_CLASS_INDEX:
    label_idx = RETRAIN_CLASS_INDEX[l_clean]
    self.samples.append((seg_id, label_idx, patient_id, admission_id))
    # ... signal processing ...
else:
    # Explicitly skip Sinus or other excluded classes
    continue
```

**3. Update Model Initialization**
Pass the correct number of classes.

```python
# In retrain_model()

# Use the new list
from data_loader import RETRAIN_CLASS_NAMES

num_classes = len(RETRAIN_CLASS_NAMES)
print(f"Retraining on {num_classes} Arrhythmia classes (Sinus Removed)")

model = CNNTransformerClassifier(num_classes=num_classes).to(device)

# Ensure logging uses the correct names
# ...
torch.save({
    # ...
    "class_names": RETRAIN_CLASS_NAMES # Save the specific classes this model knows
}, ckpt_path)
```

## 4. Verification Plan

1.  **Run `retrain.py` (Dry Run)**:
    *   Verify "Sinus Rhythm" count is 0 in the "CLASS DISTRIBUTION" log.
    *   Verify `num_classes` is reduced (e.g., from 37 to 34).
    *   Check that "Sinus Accuracy" is no longer logged (as the class doesn't exist).
2.  **Inspect Model**:
    *   Ensure final layer size matches `len(RETRAIN_CLASS_NAMES)`.

## 5. Summary of Impact
*   **Data**: ~40k Sinus samples dropped. Dataset size drastically reduced (focusing on the ~5k useful arrhythmia examples).
*   **Training**: Faster epochs, gradients dominated by arrhythmia differentiation.
*   **Inference**:
    *   The `decision_engine` must strictly gate logic:
        *   `if is_sinus_by_rules_or_baseline(): return "Sinus Rhythm"`
        *   `else: predict_with_retrained_model()` (which only outputs arrhythmias).
