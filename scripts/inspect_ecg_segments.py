
import psycopg2
import sys

PSQL_CONN_PARAMS = {
    "dbname": "ecg_analysis",
    "user": "ecg_user",
    "password": "sais",
    "host": "127.0.0.1",
    "port": "5432"
}

def list_columns():
    print("Listing columns for 'ecg_segments'...")
    try:
        conn = psycopg2.connect(**PSQL_CONN_PARAMS)
        cur = conn.cursor()
        
        cur.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'ecg_segments';
        """)
        
        rows = cur.fetchall()
        for col, dtype in rows:
            print(f" - {col} ({dtype})")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals() and conn:
            conn.close()

if __name__ == "__main__":
    list_columns()
