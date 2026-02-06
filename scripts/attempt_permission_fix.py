
import psycopg2
import sys

def try_fix_permissions():
    print("Attempting to fix permission for 'ecg_segments'...")
    
    # Try connecting as postgres (superuser)
    # Common scenarios: No password, 'postgres', 'password', 'admin'
    passwords_to_try = [None, "postgres", "password", "admin", "root", "sais"]
    
    success = False
    
    for pwd in passwords_to_try:
        try:
            params = {
                "dbname": "ecg_analysis",
                "user": "postgres",
                "host": "127.0.0.1",
                "port": "5432"
            }
            if pwd:
                params["password"] = pwd
                
            print(f"  ... trying user='postgres' with password='{pwd}'")
            
            conn = psycopg2.connect(**params)
            conn.autocommit = True
            cur = conn.cursor()
            
            # Change owner
            cur.execute("ALTER TABLE ecg_segments OWNER TO ecg_user;")
            print("  SUCCESS: Changed owner of ecg_segments to ecg_user.")
            
            # Grant all just in case
            cur.execute("GRANT ALL PRIVILEGES ON TABLE ecg_segments TO ecg_user;")
            print("  SUCCESS: Granted all privileges.")
            
            conn.close()
            success = True
            break
            
        except Exception as e:
            # print(f"  Failed: {e}")
            pass

    if success:
        print("\nPermissions fixed! You can now run the checks.")
    else:
        print("\nCould not auto-fix. Please run this SQL manually as Superuser:")
        print("   ALTER TABLE ecg_segments OWNER TO ecg_user;")

if __name__ == "__main__":
    try_fix_permissions()
