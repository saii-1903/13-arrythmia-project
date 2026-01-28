import psycopg2
import json
from typing import List, Dict, Any

# ---------------------------------------
# PostgreSQL Connection Settings
# ---------------------------------------
PSQL_CONN_PARAMS = {
    "dbname": "ecg_analysis",
    "user": "ecg_user",
    "password": "sais",         # <-- your password
    "host": "127.0.0.1",
    "port": "5432"
}

def _connect():
    """Create a new PostgreSQL connection."""
    return psycopg2.connect(**PSQL_CONN_PARAMS)

# =====================================================================
# FETCH LIST OF FILES
# =====================================================================
def get_segment_list() -> List[Dict[str, Any]]:
    conn = None
    try:
        conn = _connect()
        cur = conn.cursor()
        cur.execute("""
            SELECT filename,
                   COUNT(*) AS segment_count,
                   SUM(CASE WHEN arrhythmia_label IS NULL
                               OR arrhythmia_label='Unlabeled'
                            THEN 1 ELSE 0 END) AS unlabeled_count
            FROM ecg_features_annotatable
            GROUP BY filename
            ORDER BY filename;
        """)

        rows = cur.fetchall()
        return [
            {
                "filename": r[0],
                "segment_count": r[1],
                "unlabeled_count": r[2]
            }
            for r in rows
        ]
    except:
        return []
    finally:
        if conn:
            conn.close()

# =====================================================================
# FETCH A SINGLE SEGMENT
# =====================================================================
def get_segment_data(segment_id: int):
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    segment_id,
                    filename,
                    segment_index,
                    segment_start_s,
                    segment_duration_s,
                    arrhythmia_label,
                    arrhythmia_text_notes,
                    r_peaks_in_segment,
                    features_json,
                    cardiologist_notes,
                    corrected_by,
                    corrected_at,
                    training_round,
                    raw_signal,
                    pr_interval,
                    segment_fs,
                    dataset_source
                FROM ecg_features_annotatable
                WHERE segment_id = %s
                """,
                (segment_id,),
            )
            row = cur.fetchone()
            if not row:
                return None

            cols = [
                "segment_id",
                "filename",
                "segment_index",
                "segment_start_s",
                "segment_duration_s",
                "arrhythmia_label",
                "arrhythmia_text_notes",
                "r_peaks_in_segment",
                "features_json",
                "cardiologist_notes",
                "corrected_by",
                "corrected_at",
                "training_round",
                "raw_signal",
                "pr_interval",
                "segment_fs",
                "dataset_source",
            ]

            data = {cols[i]: row[i] for i in range(len(cols))}
            return data

    except Exception as e:
        print("DB ERROR get_segment_data:", e)
        return None
    finally:
        conn.close()


# =====================================================================
# UPDATE SEGMENT ANNOTATION
# =====================================================================
def update_annotation(
    segment_id: int, 
    # Doctor Inputs
    doctor_rhythm_label: str,
    doctor_ectopy_label: str,
    r_peaks, # Not used for logic but stored
    notes: str, 
    corrected_by: str = "Cardiologist",
    # Model Inputs (Required for logic)
    model_rhythm_label: str = None,
    model_ectopy_label: str = None,
    doctor_uncertain: bool = False
) -> bool:
    """
    Updates manual annotation and classifies the mistake for retraining.
    Strictly follows 'Correct Annotation Logic' (Step 4 of user request).
    """
    conn = None
    try:
        conn = _connect()
        cur = conn.cursor()

        r_str = ",".join(map(str, r_peaks)) if r_peaks else ""

        if doctor_rhythm_label is None or doctor_ectopy_label is None:
            raise ValueError("Doctor must explicitly provide rhythm and ectopy labels")

        if model_rhythm_label is None or model_ectopy_label is None:
             raise ValueError("Model rhythm and ectopy labels must be provided")

        doc_rhy = str(doctor_rhythm_label)
        doc_ect = str(doctor_ectopy_label)
        mod_rhy = str(model_rhythm_label)
        mod_ect = str(model_ectopy_label)

        annotation_type = "BORDERLINE"
        mistake_target = None

        # --- STEP 1: Determine annotation_type ---
        
        if doctor_uncertain:
            annotation_type = "BORDERLINE"
            mistake_target = "RHYTHM" # Assign to rhythm for safety if uncertain, or could be None?
            # User said: "if doctor_uncertain: annotation_type = BORDERLINE"
            # And "mistake_target derived from what was wrong".
            # If uncertain, we assume the model might be wrong but we don't know why.
            # Let's derive target from mismatch if exists, else None.
            if doc_rhy != mod_rhy: mistake_target = "RHYTHM"
            elif doc_ect != mod_ect: mistake_target = "ECTOPY"
            else: mistake_target = None

        # Exact Match (Everything Correct)
        elif doc_rhy == mod_rhy and doc_ect == mod_ect:
             annotation_type = "CONFIRMED_CORRECT"
             mistake_target = None
        
        else:
             # Mismatch Exists
             # Priority 1: Rhythm Mismatch
             if doc_rhy != mod_rhy:
                 if mod_rhy == "Sinus Rhythm": # Model missed an arrhythmia
                     annotation_type = "FALSE_NEGATIVE"
                 else:
                     annotation_type = "FALSE_POSITIVE" # Wrong classification
             
             # Priority 2: Ectopy Mismatch (if Rhythm was correct)
             elif doc_ect != mod_ect:
                 annotation_type = "FALSE_NEGATIVE" # Missed event logic usually FN 
                 # (Though could be FP "False Alarm", user asked to treat mismatches carefully.
                 # "Missed PVC/PAC -> ECTOPY" implies FN. 
                 # Let's stick to the prompt heuristic: "missed ectopy -> FALSE_NEGATIVE"
                 # BUT prompt specifically said "Do NOT collapse all mismatches into FN".
                 # Let's refine: 
                 if mod_ect == "None" and doc_ect != "None":
                     annotation_type = "FALSE_NEGATIVE" # Missed event
                 elif mod_ect != "None" and doc_ect == "None":
                      annotation_type = "FALSE_POSITIVE" # False Alarm
                 else:
                      annotation_type = "FALSE_POSITIVE" # Wrong Class (PVC vs Run)

        # --- STEP 2: Determine mistake_target ---
        
        if doc_rhy != mod_rhy:
            mistake_target = "RHYTHM"  # Primary Rhythm Error
        elif doc_ect != mod_ect:
            mistake_target = "ECTOPY"  # Secondary Ectopy Error
        else:
            mistake_target = None # Should fit CONFIRMED case

        # --- STEP 4: Persist (FINAL UPDATE) ---
        cur.execute("""
            UPDATE ecg_features_annotatable
            SET 
                arrhythmia_label = %s,    -- Doctor Rhythm
                ectopy_label = %s,        -- Doctor Ectopy (New Column)
                r_peaks_in_segment = %s,
                arrhythmia_text_notes = %s,
                corrected_by = %s,
                corrected_at = CURRENT_TIMESTAMP,
                
                -- Classification Fields
                annotation_type = %s,
                mistake_target = %s,
                used_for_training = FALSE, -- Reset Logic
                
                -- Optional: Store Model's view too if we wanted, but not in schema yet
                model_pred_label = %s,      -- Store Rhythm Model
                model_ectopy_label = %s     -- Store Ectopy Model
            WHERE segment_id = %s;
        """, (
            doc_rhy, doc_ect, 
            r_str, notes, corrected_by, 
            annotation_type, mistake_target, 
            mod_rhy, mod_ect,
            segment_id
        ))

        conn.commit()
        return cur.rowcount > 0

        conn.commit()
        return cur.rowcount > 0

    except Exception as e:
        print("DB ERROR update_annotation:", e)
        return False
    finally:
        if conn:
            conn.close()

# =====================================================================
# SAVE MODEL PREDICTION (for XAI UI)
# =====================================================================
def save_model_prediction(segment_id: int, pred_label: str, probs_list):
    conn = None
    try:
        conn = _connect()
        cur = conn.cursor()

        cur.execute("""
            UPDATE ecg_features_annotatable
            SET model_pred_label = %s,
                model_pred_probs = %s
            WHERE segment_id = %s;
        """, (pred_label, json.dumps(probs_list), segment_id))

        conn.commit()

    except Exception as e:
        print("DB ERROR save_model_prediction:", e)
    finally:
        if conn:
            conn.close()
    return True
# =====================================================================
# FIND FIRST SEGMENT WITH raw_signal
# =====================================================================
def get_min_segment_id_with_signal() -> int:
    conn = None
    try:
        conn = _connect()
        cur = conn.cursor()

        cur.execute("""
            SELECT MIN(segment_id)
            FROM ecg_features_annotatable
            WHERE raw_signal IS NOT NULL;
        """)

        row = cur.fetchone()
        return int(row[0]) if row and row[0] else 0

    except Exception as e:
        print("DB ERROR get_min_segment_id_with_signal:", e)
        return 0
    finally:
        if conn:
            conn.close()

# =====================================================================
# GENERIC fetch_one() used by your app.py
# =====================================================================
def fetch_one(sql: str, params=None):
    conn = None
    try:
        conn = _connect()
        cur = conn.cursor()
        cur.execute(sql, params)
        return cur.fetchone()
    except Exception as e:
        print("DB fetch_one error:", e)
        return None
    finally:
        if conn:
            conn.close()

# =====================================================================
# Find first segment for a newly uploaded JSON
# =====================================================================
def get_first_segment_id_by_filename(filename_key: str) -> int:
    conn = None
    try:
        conn = _connect()
        cur = conn.cursor()

        cur.execute("""
            SELECT segment_id
            FROM ecg_features_annotatable
            WHERE filename = %s
            ORDER BY segment_index ASC
            LIMIT 1;
        """, (filename_key,))

        row = cur.fetchone()
        return row[0] if row else 0

    except Exception as e:
        print("DB ERROR get_first_segment_id_by_filename:", e)
        return 0

    finally:
        if conn:
            conn.close()

# =====================================================================
# GET ALL CORRECTED SEGMENTS (For Export)
# =====================================================================
def get_all_corrected() -> List[Dict[str, Any]]:
    conn = None
    try:
        conn = _connect()
        cur = conn.cursor()
        cur.execute("""
            SELECT segment_id,
                   filename,
                   segment_index,
                   arrhythmia_label,
                   model_pred_label,
                   features_json,
                   raw_signal,
                   segment_fs,
                   dataset_source
            FROM ecg_features_annotatable
            WHERE raw_signal IS NOT NULL
              AND arrhythmia_label IS NOT NULL
              AND arrhythmia_label != 'Unlabeled';
        """)
        
        rows = cur.fetchall()
        cols = [
            "segment_id", "filename", "segment_index", "arrhythmia_label",
            "model_pred_label", "features_json", "raw_signal", "segment_fs", "dataset_source"
        ]
        
        results = []
        for r in rows:
            results.append({cols[i]: r[i] for i in range(len(cols))})
            
        return results

    except Exception as e:
        print("DB ERROR get_all_corrected:", e)
        return []
    finally:
        if conn:
            conn.close()
