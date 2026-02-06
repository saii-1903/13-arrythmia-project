
import psycopg2
import sys
from pathlib import Path

# Fix path to imports
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

try:
    from database.db_service import PSQL_CONN_PARAMS
except ImportError:
    PSQL_CONN_PARAMS = {
        "dbname": "ecg_analysis",
        "user": "ecg_user",
        "password": "sais",
        "host": "127.0.0.1",
        "port": "5432"
    }

def verify_schema():
    print("Verifying Database Schema...")
    
    required_tables = {
        "ecg_features_annotatable": [
            # --- Base Labels ---
            "arrhythmia_label", 
            "ectopy_label", 
            "dataset_source", 
            "used_for_training",
            
            # --- Model Outputs (For Comparison) ---
            "model_pred_label",
            "model_ectopy_label",

            # --- "Presidency" / Cardiologist Override Columns (CRITICAL) ---
            "is_verified",          # Boolean: Has a human checked this?
            "mistake_target",       # Text: 'RHYTHM', 'ECTOPY', or NULL
            "annotation_type",      # Text: 'CONFIRMED', 'FALSE_POSITIVE', etc.
            "cardiologist_notes"    # Text: Optional notes from the doctor
        ],
        "ecg_segments": [
            "events_json",
            "patient_id",
            "segment_state",
            "signal"            # The actual column name is 'signal' (jsonb)
        ]
    }
    
    try:
        conn = psycopg2.connect(**PSQL_CONN_PARAMS)
        cur = conn.cursor()
        
        all_good = True
        
        for table, required_cols in required_tables.items():
            print(f"\nTable: {table}")
            
            # Check existence
            cur.execute("SELECT to_regclass(%s)", (table,))
            if not cur.fetchone()[0]:
                print(f"  MISSING TABLE: {table}")
                all_good = False
                continue
                
            # Get columns
            cur.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = %s
            """, (table,))
            existing_cols = {row[0] for row in cur.fetchall()}
            
            # Check required columns
            for col in required_cols:
                if col in existing_cols:
                    print(f"  Found column: {col}")
                else:
                    print(f"  MISSING column: {col}")
                    all_good = False
            
            # Get Row Count
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            count = cur.fetchone()[0]
            print(f"  Row Count: {count}")

        if all_good:
            print("\nDATABASE SCHEMA IS READY.")
        else:
            print("\nDATABASE SCHEMA HAS ISSUES. Run migrations.")
            
    except Exception as e:
        print(f"Connection Error: {e}")
    finally:
        if 'conn' in locals() and conn:
            conn.close()

if __name__ == "__main__":
    verify_schema()
