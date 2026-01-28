
# Implementation Plan: Remove Composite Classes and Handle Recursion

## 1. Objective
Remove composite classes (`Base + Event`) from the retraining set while ensuring robust mapping and handling of ectopy classes temporarily.

## 2. Core Constraints
*   **Normalization First**: `get_retrain_label_idx` must normalize inputs before attempting to split strings to handle variations like "AF+PVC".
*   **Recursive Split**: Logic must split on " + " and recursively check the base rhythm.
*   **Documentation Contract**: Explicitly note that pure ectopy classes (PVC, PAC) remain *for now* but are slated for removal in Change 3.

## 3. Proposed Changes (Revised)

### A. Modify `models_training/data_loader.py`

**1. Explicit Documentation (Architectural Contract)**
Add the mandatory comment block.

```python
# NOTE:
# Ectopy-only classes (PVC, PAC, Bigeminy, Trigeminy) are
# intentionally retained for CHANGE-2.
# These will be moved to a separate Ectopy Model in CHANGE-3.
RETRAIN_CLASS_NAMES = [
    c for c in CLASS_NAMES 
    if c not in EXCLUDED_PURE_SINUS 
    and " + " not in c
]
```

**2. Robust Mapping Logic (`get_retrain_label_idx`)**
Implement the safe recursion pattern.

```python
def get_retrain_label_idx(original_label_name):
    """
    Returns index for retraining.
    1. Normalizes label.
    2. Handles composites via recursion ("AF + PVC" -> "AF").
    3. Returns None for excluded classes (Sinus).
    """
    # 1. Normalize first (Critical Fix #1)
    label = normalize_label(original_label_name)

    # 2. Handle composites
    if " + " in label:
        base = label.split(" + ")[0]
        # Recursively get index for base (e.g., "Atrial Fibrillation")
        # This handles nested normalization if needed, though normalize_label should cover it.
        return get_retrain_label_idx(base)

    # 3. Base case check
    if label in RETRAIN_CLASS_INDEX:
        return RETRAIN_CLASS_INDEX[label]
        
    return None
```

### B. Verify Impact
*   **Safe Recursion**: `Atrial Fibrillation + PVC` -> `Atrial Fibrillation` -> [Lookup] -> Index.
*   **Sinus Dropped**: `Sinus Bradycardia + PVC` -> `Sinus Bradycardia` -> [Lookup Fail] -> None.
*   **Ectopy Kept**: `PVC Bigeminy` -> [Lookup] -> Index (kept for Change 2).

## 4. Verification Steps
1.  **Script**: `verify_retrain_setup.py` updates:
    *   Test `get_retrain_label_idx("AF+PVC")` (Un-normalized input) -> Should return AF index.
    *   Test `get_retrain_label_idx("Sinus Bradycardia + PVC")` -> Should return None.
    *   Assert `PVC Bigeminy` is in `RETRAIN_CLASS_NAMES`.
    
## 5. Summary
This plan strictly adheres to the "Normalization First" rule and documents the architectural state of Ectopy classes.
