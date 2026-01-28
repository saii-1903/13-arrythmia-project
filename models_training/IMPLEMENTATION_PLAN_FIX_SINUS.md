
# Implementation Plan: Fix Sinus Removal Strategy

## 1. Objective
Refine the strategy for removing Sinus Rhythm from the retraining dataset to ensure correctness, safety, and proper handling of "Artifact" and label normalization.

## 2. Changes Required

### A. Fix `RETRAIN_CLASS_NAMES` Logic (Keep Artifact, Handle Composites)
**Correction**: The previous plan/logic might have filterd Artifact or was ambiguous. We must strictly **KEEP** "Artifact" and **REMOVE** only pure Sinus classes.
**Nuance**: The user suggested `if not name.startswith("Sinus")`. However, `CLASS_NAMES` contains "Sinus Bradycardia + PVC".
*   If we use `startswith("Sinus")`, we lose PVC data occurring during bradycardia.
*   **Decision**: We will use an explicit Blocklist to remove specific Sinus classes ("Sinus Rhythm", "Sinus Bradycardia", "Sinus Tachycardia") but **KEEP** composites ("Sinus Bradycardia + PVC") and **KEEP** "Artifact".
*   *Note*: The user's prompt emphasized "Keep Artifact".

```python
# In data_loader.py

EXCLUDED_PURE_SINUS = {
    "Sinus Rhythm", 
    "Sinus Bradycardia", 
    "Sinus Tachycardia"
}

RETRAIN_CLASS_NAMES = [
    c for c in CLASS_NAMES 
    if c not in EXCLUDED_PURE_SINUS
    # Implicitly keeps "Artifact" and "Sinus Bradycardia + PVC"
]
```

### B. Python-Side Filtering (Safety Gate)
**Correction**: Do not rely solely on SQL. SQL usually has raw strings. Verification must happen *after* normalization in Python.
**Action**:
1.  Keep SQL filter (as an optimization to avoid fetching 40k rows).
2.  **Add strict Python check** in `__getitem__` or the loading loop.

```python
# In retrain.py loop

label_clean = normalize_label(raw_label)

# 1. Check if it's a target class
label_idx = get_retrain_label_idx(label_clean)

# 2. Explicit sanity check (Double Safety)
if label_idx is None:
    continue
    
if RETRAIN_CLASS_NAMES[label_idx] in EXCLUDED_PURE_SINUS:
    raise ValueError(f"FATAL: Sinus class {RETRAIN_CLASS_NAMES[label_idx]} leaked into dataset!")
```

### C. Add Guards and Assertions
**Correction**: "Explicitly enforce NO SINUS at dataset level".
**Action**: Add assertions in `retrain.py` startup.

```python
# In retrain.py (before training)

print("🛡️ Verifying Class List Integrity...")
forbidden_terms = ["Sinus Rhythm", "Sinus Bradycardia", "Sinus Tachycardia"]
for name in RETRAIN_CLASS_NAMES:
    if name in forbidden_terms:
        raise RuntimeError(f"FATAL: {name} found in RETRAIN_CLASS_NAMES. Exiting.")
    # Extra check for Artifact
    if name == "Artifact":
        print("✅ Artifact is PRESENT in training classes (Correct).")

print("✅ Integrity Check Passed: No pure Sinus classes.")
```

## 3. Execution Steps
1.  **Edit `data_loader.py`**: Ensure `RETRAIN_CLASS_NAMES` construction is explicit (using the Set Logic) and does not accidentally drop Artifact or Composites (unless specifically desired, but assuming we want PVCs).
2.  **Edit `retrain.py`**:
    *   Insert assertions at the start of `retrain_model`.
    *   Ensure the Python loop filters based on the *normalized* label mapping.
3.  **Verify**: Run `python models_training/retrain.py --dry-run` (or similar) to check the assertions pass and distribution log shows 0 Sinus.

## 4. Why this matches the User Request
*   **Artifact Kept**: Logic explicitly preserves it.
*   **Python Filter**: We filter after `normalize_label`.
*   **Assertions**: Added startup guards.
