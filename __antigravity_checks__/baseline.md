# Baseline Analysis - 2026-02-03

## SQL Interactions - `database/db_service.py`
- `get_segment_list()`: Summary counts by filename.
- `get_segment_data(segment_id)`: Detailed segment info including `raw_signal`, `features_json`, `corrected_by`, etc.
- `update_annotation(...)`: Updates `arrhythmia_label`, `ectopy_label`, `annotation_type`, `mistake_target`, and `used_for_training`.
  - **Annotation Logic**:
    - `doc_rhy != mod_rhy` -> `mistake_target = 'RHYTHM'`
    - `doc_ect != mod_ect` and `doc_rhy == mod_rhy` -> `mistake_target = 'ECTOPY'`
    - `FALSE_NEGATIVE` if model missed an arrhythmia (thought it was Sinus).
    - `FALSE_POSITIVE` if model detected something wrong or different.
- `save_model_prediction(...)`: Updates `model_pred_label` and `model_pred_probs`.
- `get_all_corrected()`: Exports segments where `arrhythmia_label` is not NULL/Unlabeled.

## Training Logic - `models_training/retrain.py`
- **Data Selection**:
  - `raw_signal IS NOT NULL`
  - `arrhythmia_label IS NOT NULL` AND `!= 'Unlabeled'`
  - `corrected_by IS NOT NULL`
  - `annotation_type IN ('FALSE_POSITIVE', 'FALSE_NEGATIVE', 'BORDERLINE')`
  - `used_for_training = FALSE OR NULL`
  - `mistake_target = TASK` (RHYTHM or ECTOPY)
- **Weighting**:
  - `FALSE_NEGATIVE`: 1.5
  - `FALSE_POSITIVE`: 1.0
  - `BORDERLINE`: 0.7
- **Classes**:
  - `RHYTHM_CLASS_NAMES`: 15 classes (excludes Sinus variants and Composites).
  - `ECTOPY_CLASS_NAMES`: 4 classes (None, PVC, PAC, Run).

## Currently Active File - `models_training/calibration.py`
- Implements `TemperatureScaling` for calibrating logits.
- Uses `_ECELoss` for evaluation.
- Optimizes `temperature` using `LBFGS` on NLL of validation set.
