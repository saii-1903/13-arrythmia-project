
# Implementation Plan: Split into Rhythm and Ectopy Models (Change 3)

## 1. Objective
Decouple rhythm detection from ectopy detection to simplify the learning objective and improve convergence.

## 2. Models Definition

### Model A: Rhythm Model
*   **Purpose**: Detect primary rhythm pathologies.
*   **Classes**:
    *   Pathologies: AF, Flutter, SVT, VT, VF, Junctional, Idioventricular, AV Blocks (1st, 2nd-T1, 2nd-T2, 3rd), BBB, Pause.
    *   Reference: Artifact.
*   **Exclusions**: Sinus Rhythm, Sinus Bradycardia, Sinus Tachycardia, and all pure Ectopy (PVC, PAC, Bigeminy, Trigeminy).

### Model B: Ectopy Model
*   **Purpose**: Detect beat-level abnormalities regardless of base rhythm.
*   **Classes**:
    1.  `None` (No ectopy present)
    2.  `PVC` (Includes PVCs, Couplets, Bigeminy/Trigeminy)
    3.  `PAC` (Includes PACs, Bigeminy)
    4.  `Run` (Atrial Run, Ventricular Run, NSVT)

## 3. Proposed Changes

### A. Modify `models_training/data_loader.py`

**1. Define Task-Specific Class Names**
```python
RHYTHM_CLASS_NAMES = [
    "Atrial Fibrillation", "Atrial Flutter", "Supraventricular Tachycardia",
    "Ventricular Tachycardia", "Ventricular Fibrillation", "Junctional Rhythm",
    "Idioventricular Rhythm", "1st Degree AV Block", "2nd Degree AV Block Type 1",
    "2nd Degree AV Block Type 2", "3rd Degree AV Block", "Bundle Branch Block",
    "Artifact", "Pause"
]

ECTOPY_CLASS_NAMES = ["None", "PVC", "PAC", "Run"]
```

**2. Implement Task-Specific Label Mapping**
*   **`get_rhythm_label_idx(label)`**:
    *   Normalize -> Split on " + " -> Take Base.
    *   If Base is in `RHYTHM_CLASS_NAMES` -> Return Index.
    *   Else -> Return `None`.
*   **`get_ectopy_label_idx(label)`**:
    *   Normalize.
    *   If label contains "PVC" or "Bigeminy/Trigeminy" (with PVC) -> "PVC".
    *   If label contains "PAC" -> "PAC".
    *   If label contains "Run" or "NSVT" -> "Run".
    *   Else -> "None".

### B. Modify `models_training/retrain.py`

**1. Add `--task` Argument**
*   Possible values: `rhythm`, `ectopy`.

**2. Dynamic Configuration**
*   Based on task, set:
    *   `CLASS_NAMES_TO_USE` (either `RHYTHM_CLASS_NAMES` or `ECTOPY_CLASS_NAMES`).
    *   `LABEL_MAP_FUNC` (task-specific mapper).
    *   `CHECKPOINT_NAME` (e.g., `best_model_rhythm.pth`).
    *   `SQL_FILTER`:
        *   Rhythm Task: `WHERE arrhythmia_label != 'Sinus Rhythm'...`
        *   Ectopy Task: `WHERE 1=1` (Can train on Sinus segments to learn "None").

**3. Safety Assertions**
*   Update guardrails to verify the specific task's class list.

## 4. Execution Workflow

1.  **Update `data_loader.py`**: Establish the new list and mapping functions.
2.  **Update `retrain.py`**: Implement the task-switching logic.
3.  **Train Rhythm Model**:
    ```bash
    python retrain.py --task rhythm --epochs 30
    ```
4.  **Train Ectopy Model**:
    ```bash
    python retrain.py --task ectopy --epochs 30
    ```

## 5. Verification Plan
*   **Rhythm Mode**: Verify "PVCs" segments are dropped or mapped to `None`.
*   **Ectopy Mode**: Verify "Sinus Rhythm" segments map to index 0 (`None`). Verify "AF + PVC" maps to index 1 (`PVC`).
*   **Storage**: Confirm two separate `.pth` files are generated.
