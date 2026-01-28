
# Implementation Plan: Balance Mechanisms & Mistake-Only Policy (Changes 4 & 5)

## 1. Objective
Refine the retraining instability by removing over-correction in class balancing and strictly limiting training data to model mistakes and borderline cases.

## 2. Change 4: Balancing Mechanism Optimization

### A. Remove Weighted Sampler
*   **Action**: In `retrain.py`, remove the `WeightedRandomSampler` from both `rhythm` and `ectopy` training loops. 
*   **Replacement**: Use standard `shuffle=True` in the `DataLoader`.
*   **Rationale**: The combination of Focal Loss + Oversampling + Heavy Weights causes the model to memorize rare classes rather than learn general patterns.

### B. Cap Class Weights
*   **Action**: In `retrain_model()`, clamp the calculated class weights.
*   **Logic**: `class_weights = torch.clamp(class_weights, max=5.0)`.
*   **Rationale**: Prevents individual rare samples from having 30x-40x the gradient impact of a standard segment, which destabilizes the loss landscape.

## 3. Change 5: Mistake-Only Data Policy

### A. Database Preparation
*   **Constraint**: The `ecg_features_annotatable` table currently lacks an `annotation_type` column.
*   **Action**: Create a migration/utility script to:
    1.  Add `annotation_type` (VARCHAR).
    2.  Populate it:
        *   `FALSE_NEGATIVE`: `model_pred_label` was Sinus-like, but `arrhythmia_label` is Pathology.
        *   `FALSE_POSITIVE`: `model_pred_label` was Pathology, but `arrhythmia_label` is Sinus/Normal.
        *   `BORDERLINE`: Segments with doctor notes indicating uncertainty.
    *   *Self-Correction*: I will use a SQL-level derivation if I cannot permanently alter the schema, but adding the column is better for the "non-negotiable" rule.

### B. Update `retrain.py` SQL Query
*   **Action**: Modify the `ECGRawDatasetSQL` query to strictly filter by `annotation_type`.
*   **Query**:
    ```sql
    WHERE raw_signal IS NOT NULL
      AND corrected_by IS NOT NULL
      AND annotation_type IN ('FALSE_POSITIVE', 'FALSE_NEGATIVE', 'BORDERLINE')
    ```
*   **Note**: This applies to both Rhythm and Ectopy tasks. Correct predictions ("REDUNDANT" data) are now strictly excluded.

## 4. Execution Workflow
1.  **DB Schema Update**: Add and populate `annotation_type`.
2.  **Code Correction**:
    *   Modify `retrain.py` to use the new filter.
    *   Remove Sampler and clamp weights.
3.  **Verification**: 
    *   Log the new dataset size (should be significantly smaller, focusing on "repair").
    *   Dry-run to ensure loss stability.

## 5. Summary of Stability Impact
*   **Change 4**: Flattens the loss landscape, preventing "fake 1.000 accuracy" on memorized noise.
*   **Change 5**: Forces the model to only update gradients on its own failures, preventing gradient collapse from 40,000 "already solved" Sinus segments.
