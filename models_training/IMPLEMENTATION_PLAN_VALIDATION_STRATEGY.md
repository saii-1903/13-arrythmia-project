
# Implementation Plan: Fix Validation Strategy (Change 6)

## 1. Objective
Eliminate data leakage in single-patient (or few-patient) scenarios by enforcing a strict **Time-Based Split**, replacing random record-level splitting.

## 2. Current Issue
*   Current logic relies on `sklearn.model_selection.train_test_split(shuffle=True)` when patient IDs are insufficient.
*   For continuous ECG (Holter/Bedside), adjacent segments are highly correlated. Random shuffling mixes "future" segments into training and "past" segments into validation, inflating accuracy (autocorrelation leakage).

## 3. Proposed Changes (`models_training/retrain.py`)

### A. Update `ECGRawDatasetSQL`
1.  **Modify SQL Queries**: Include `segment_start_s` (or `segment_index`) in the `SELECT` clause.
2.  **Update Storage**: Append `segment_start_s` to the `self.samples` tuple list.
    *   New Tuple Structure: `(seg_id, label_idx, patient_id, admission_id, c_weight, segment_start_s)`

### B. Update Splitting Logic in `retrain_model`
1.  **Check Patient Count**: Determine if `len(unique_patients) <= 1`.
2.  **Implement Time-Split Branch**:
    *   **Sort**: Sort all samples explicitly by `segment_start_s`.
    *   **Cutoff**: Calculate split index at 70% (0.7 * N).
    *   **Assign**: 
        *   `Train` = First 70% (Historical data).
        *   `Val` = Last 30% (Future data).
    *   **Logging**: Explicitly print "⚠️ Single patient detected. Using TIME-BASED split (No Shuffling) to prevent leakage."
3.  **Preserve Patient-Split**: Keep existing logic for multi-patient scenarios (which is safe).

## 4. Verification
1.  **Dry Run**: Run with the single patient DB.
2.  **Check Indices**: Verify `train_idx` set contains only earlier timestamps compared to `val_idx`.
3.  **Check Overlap**: Assert max(train_time) <= min(val_time).

## 5. Why this is Non-Negotiable
*   Ensures the model is tested on its ability to *generalize to the future*, not its ability to interpolate between known seconds (which is trivial and useless).
