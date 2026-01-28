
# Implementation Plan: Split Models with Hard Architecture Guards (Change 3)

## 1. Objective
Split retraining into two specialized tasks: **Rhythm** (pathology detection) and **Ectopy** (beat-level events). Implement hard logic gates to prevent cross-leakage and ensure clinical validity.

## 2. Model Specifications & Class Lists

### Model A: Rhythm Task
*   **Classes**:
    ```python
    RHYTHM_CLASS_NAMES = [
        "Atrial Fibrillation", "Atrial Flutter", "Supraventricular Tachycardia",
        "Ventricular Tachycardia", "Ventricular Fibrillation", "Junctional Rhythm",
        "Idioventricular Rhythm", "1st Degree AV Block", "2nd Degree AV Block Type 1",
        "2nd Degree AV Block Type 2", "3rd Degree AV Block", "Bundle Branch Block",
        "Artifact", "Pause"
    ]
    ```
*   **Safety Gate**: Must reject **any** segment labeled with ectopy (PVC, PAC, etc.) even if it occurs during an arrhythmia. Rhythm model should learn pure morphologies.

### Model B: Ectopy Task
*   **Classes**:
    ```python
    ECTOPY_CLASS_NAMES = ["None", "PVC", "PAC", "Run"]
    ```
*   **Safety Gate**: Must ignore rhythm semantics. AF segments without PVCs map to `None`. Sinus segments without PVCs map to `None`.

## 3. Required Code Changes

### A. Modify `models_training/data_loader.py`

**1. Task-Specific Mapper Logic**
Implementing strict rejection for Rhythm and semantic-blindness for Ectopy.

```python
# Rhythm Mapper
def get_rhythm_label_idx(original_label_name):
    label = normalize_label(original_label_name)
    
    # HARD REJECTION: No ectopy in Rhythm model
    ectopy_terms = ["PVC", "PAC", "Bigeminy", "Trigeminy", "Couplet", "Run", "NSVT"]
    if any(term.upper() in label.upper() for term in ectopy_terms):
        return None
        
    # Handle Remaining Composites (if any)
    if " + " in label:
        return get_rhythm_label_idx(label.split(" + ")[0])

    return RHYTHM_CLASS_INDEX.get(label, None)

# Ectopy Mapper
def get_ectopy_label_idx(original_label_name):
    label = normalize_label(original_label_name).upper()
    
    if "PVC" in label or "BIGEMINY" in label or "TRIGEMINY" in label or "COUPLET" in label:
        return ECTOPY_CLASS_INDEX["PVC"]
    if "PAC" in label:
        return ECTOPY_CLASS_INDEX["PAC"]
    if "RUN" in label or "NSVT" in label:
        return ECTOPY_CLASS_INDEX["Run"]
        
    return ECTOPY_CLASS_INDEX["None"]
```

### B. Modify `models_training/retrain.py`

**1. CLI Task Selection**
Add `--task=rhythm|ectopy`.

**2. SQL Optimization Logic**
*   `rhythm`: Exclude Sinus strings in SQL (`NOT IN ('Sinus Rhythm', ...)`).
*   `ectopy`: **Include everything** (need Sinus to learn `None` baseline).

**3. Startup Assertions (Mandatory)**
```python
if task == "rhythm":
    # Ensure no ectopy snuck into names
    for name in RHYTHM_CLASS_NAMES:
        assert not any(t in name for t in ["PVC", "PAC", "Bigeminy"]), f"Leak! {name}"
elif task == "ectopy":
    assert "None" in ECTOPY_CLASS_NAMES
    assert ECTOPY_CLASS_INDEX["None"] == 0
```

**4. Checkpoint Management**
Automatically switch between `best_model_rhythm.pth` and `best_model_ectopy.pth`.

## 4. Execution Workflow

1.  **`data_loader.py` Update**: Implement class lists and the two guarded mapping functions.
2.  **`retrain.py` Update**:
    *   Wire up the `--task` argument.
    *   Update `ECGRawDatasetSQL` to use the task-specific mapper.
    *   Add startup assertions.
3.  **Verification**: Dry-run both tasks to ensure class distributions are correct (Rhythm distribution has 0 PVC/Sinus; Ectopy distribution shows "None" as the majority class).

## 5. Summary of Guards
*   **Guard 1**: `normalize_label` is always the entry point.
*   **Guard 2**: `get_rhyth_label_idx` identifies ectopy terms and returns `None` immediately.
*   **Guard 3**: `get_ectopy_label_idx` reduces all rhythm contexts (Sinus/AF/VT) to `None` unless ectopy keywords exist.
