
import psycopg2
import sys

PSQL_CONN_PARAMS = {
    "dbname": "ecg_analysis",
    "user": "ecg_user",
    "password": "sais",
    "host": "127.0.0.1",
    "port": "5432"
}

def check_owner():
    print("Checking Table Ownership...")
    try:
        conn = psycopg2.connect(**PSQL_CONN_PARAMS)
        cur = conn.cursor()
        
        cur.execute("""
            SELECT tablename, tableowner 
            FROM pg_tables 
            WHERE tablename = 'ecg_segments' OR tablename = 'ecg_features_annotatable';
        """)
        
        rows = cur.fetchall()
        for table, owner in rows:
            print(f"Table: {table} | Owner: {owner}")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals() and conn:
            conn.close()

if __name__ == "__main__":
    check_owner()
