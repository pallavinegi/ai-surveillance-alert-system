import sqlite3
import datetime


DB_NAME = "surveillance.db"

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            camera_id TEXT,
            alert_type TEXT,
            confidence REAL,
            snapshot_path TEXT,
            latitude REAL,
            longitude REAL
        )
    """)
    
    cursor.execute("""
       CREATE TABLE IF NOT EXISTS mail_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        subject TEXT,
        recipient TEXT,
        status TEXT
    )
    """)
    conn.commit()
    conn.close()
    
def log_mail(subject, recipient, status):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO mail_logs (timestamp, subject, recipient, status)
        VALUES (?, ?, ?, ?)
    """, (
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        subject,
        recipient,
        status
    ))


    conn.commit()
    conn.close()
