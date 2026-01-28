
# Implementation Plan: Remove Composite Classes "Sinus + X"

## 1. Objective
Eliminate all composite labels (e.g., "Sinus Bradycardia + PVC", "Atrial Fibrillation + PVC") from the retraining label space. This prevents the model from learning "mixed" classes.

## 2. Core Constraints
*   **No Composite Labels**: `RETRAIN_CLASS_NAMES` must not contain any class with " + ".
*   **Data Handling**:
    *   **Sinus + X**: Mapped to "Sinus" -> **DROPPED** (since Sinus is excluded).
    *   **Arrhythmia + X**: Mapped to "Arrhythmia" -> **KEPT** (e.g., "AF + PVC" -> "AF").
    *   *Rationale*: An AF segment with a PVC is still AF. Discarding it wastes valuable arrhythmia data. Mapping it to AF makes the model robust to PVC artifacts during AF types.

## 3. Proposed Changes

### A. Modify `models_training/data_loader.py`

**1. Define Composites Exclusion**
Exclude any class name containing `" + "`.

```python
# In data_loader.py

RETRAIN_CLASS_NAMES = [
    c for c in CLASS_NAMES 
    if c not in EXCLUDED_PURE_SINUS 
    and " + " not in c  # NEW: Remove composites
]
```

**2. Update Mapping Logic (`get_retrain_label_idx`)**
Implement the "Split and Map" strategy.

```python
def get_retrain_label_idx(original_label_name):
    # 1. Handle Composites: "Base + Event" -> "Base"
    if " + " in original_label_name:
        parts = original_label_name.split(" + ")
        base_rhythm = parts[0] # e.g., "Atrial Fibrillation" or "Sinus Bradycardia"
        
        # Check if base is mappable
        # Note: If base is "Sinus Bradycardia", it will fail the lookup below (Correct).
        # If base is "Atrial Fibrillation", it will succeed (Correct).
        return get_retrain_label_idx(base_rhythm)

    # 2. Standard Lookup
    if original_label_name in RETRAIN_CLASS_INDEX:
        return RETRAIN_CLASS_INDEX[original_label_name]
    
    return None
```

### B. Verify Impact
*   **Sinus Bradycardia + PVC** -> Maps to `Sinus Bradycardia` -> **Excluded**. (Correct)
*   **Atrial Fibrillation + PVC** -> Maps to `Atrial Fibrillation` -> **Included** (index of AF). (Correct)
*   **PVC Bigeminy** -> No `+` -> **Included** (for now, until Change 3).
*   **Artifact** -> No `+` -> **Included**.

## 4. Verification
1.  **Script**: Update `verify_retrain_setup.py` to assert:
    *   No class in `RETRAIN_CLASS_NAMES` contains `+`.
    *   `get_retrain_label_idx("Atrial Fibrillation + PVC")` returns the index of AF.
    *   `get_retrain_label_idx("Sinus Bradycardia + PVC")` returns `None`.

## 5. Summary
This change strictly enforces "No Composites" in the output space, while intelligently preserving arrhythmia data where the base rhythm is pathological.
