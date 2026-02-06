
import psycopg2
import json
import numpy as np
import sys
from pathlib import Path

# Fix path to imports
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

try:
    from database.db_service import PSQL_CONN_PARAMS
    from decision_engine.rhythm_orchestrator import RhythmOrchestrator
    from xai.xai import explain_segment, explain_decision
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def migrate_sample(limit=20):
    print(f"🚀 Starting Migration Sample (Limit: {limit})...")
    
    orchestrator = RhythmOrchestrator()
    
    try:
        conn = psycopg2.connect(**PSQL_CONN_PARAMS)
        cur = conn.cursor()
        
        # 1. Fetch old data
        cur.execute("""
            SELECT 
                segment_id, 
                raw_signal, 
                segment_fs, 
                features_json, 
                filename, 
                segment_index
            FROM ecg_features_annotatable
            WHERE raw_signal IS NOT NULL
            LIMIT %s
        """, (limit,))
        
        rows = cur.fetchall()
        print(f"✓ Found {len(rows)} segments for migration.")
        
        for row in rows:
            seg_id, signal_data, fs, features_raw, filename, seg_idx = row
            
            # Convert signal to numpy
            if isinstance(signal_data, str):
                signal = np.array(list(map(float, signal_data.split(","))))
            else:
                signal = np.array(signal_data)
                
            features = features_raw if isinstance(features_raw, dict) else json.loads(features_raw or "{}")
            
            # 2. Run Decision Engine
            # We need ML predictions first
            ml_results = explain_segment(signal, features)
            
            # Orchestrate
            # Note: SQI result can be derived or mocked if missing.
            # Usually features_json has some SQI if calculated.
            sqi = {"is_acceptable": True} # Default for migration
            
            decision = orchestrator.decide(
                ml_prediction=ml_results.get("rhythm", {}),
                clinical_features=features,
                sqi_result=sqi,
                segment_index=seg_idx
            )
            
            # 3. Generate XAI Narrative
            narrative = explain_decision(decision)
            
            # 4. Save to NEW table ecg_segments
            cur.execute("""
                INSERT INTO ecg_segments (
                    segment_id,
                    signal,
                    features,
                    segment_state,
                    background_rhythm,
                    events_json,
                    filename,
                    segment_index,
                    segment_fs
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (segment_id) DO UPDATE SET
                    segment_state = EXCLUDED.segment_state,
                    background_rhythm = EXCLUDED.background_rhythm,
                    events_json = EXCLUDED.events_json,
                    filename = EXCLUDED.filename,
                    segment_index = EXCLUDED.segment_index,
                    segment_fs = EXCLUDED.segment_fs;
            """, (
                seg_id,
                json.dumps(signal.tolist()),
                json.dumps(features),
                decision.segment_state.value,
                decision.background_rhythm,
                json.dumps([e.to_dict() for e in decision.events]),
                filename,
                seg_idx,
                fs
            ))
            
            print(f"  [+] Migrated Segment {seg_id} | Result: {decision.background_rhythm}")
            
        conn.commit()
        print("✅ Migration Sample Complete!")
        
    except Exception as e:
        print(f"❌ Error during migration: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    migrate_sample(limit=20)
