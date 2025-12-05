import sqlite3
import time
from contextlib import contextmanager
from src.helpers import load_config

config = load_config()

max_requests = config['rate_limiting']['max_requests']
time_window = config['rate_limiting']['time_window']
class RateLimit:

    def __init__(self, db_path = 'rate_limits.db'):
        self.requests = {}
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create table if it doesn't exist"""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rate_limits (
                    ip_address TEXT,
                    request_time REAL,
                    PRIMARY KEY (ip_address, request_time)
                )
            """)
            # Create index for faster queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ip_time 
                ON rate_limits(ip_address, request_time)
            """)

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
    
    def is_allowed(self, ip_address, max_requests=max_requests, time_window=time_window):
        """Check if request is allowed"""
        current_time = time.time()
        cutoff_time = current_time - time_window
        
        with self._get_connection() as conn:
            # STEP 1: Clean old requests (older than time_window)
            conn.execute("""
                DELETE FROM rate_limits 
                WHERE ip_address = ? AND request_time < ?
            """, (ip_address, cutoff_time))
            
            # STEP 2: Count recent requests
            cursor = conn.execute("""
                SELECT COUNT(*) FROM rate_limits 
                WHERE ip_address = ? AND request_time > ?
            """, (ip_address, cutoff_time))
            
            request_count = cursor.fetchone()[0]
            
            # STEP 3: Check if over limit
            if request_count >= max_requests:
                # Get oldest timestamp to calculate wait time
                cursor = conn.execute("""
                    SELECT MIN(request_time) FROM rate_limits 
                    WHERE ip_address = ?
                """, (ip_address,))
                
                oldest_timestamp = cursor.fetchone()[0]
                wait_time = int((oldest_timestamp + time_window) - current_time)
                return False, wait_time
            
            # STEP 4: Add new request
            conn.execute("""
                INSERT INTO rate_limits (ip_address, request_time) 
                VALUES (?, ?)
            """, (ip_address, current_time))
        
        return True, None
    
    def cleanup_old_records(self, days=7):
        """Clean up records older than X days (maintenance)"""
        cutoff = time.time() - (days * 24 * 3600)
        with self._get_connection() as conn:
            conn.execute("DELETE FROM rate_limits WHERE request_time < ?", (cutoff,))




