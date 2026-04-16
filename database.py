import sqlite3
import os
from dotenv import load_dotenv

load_dotenv()

# sqlite db path - stored in project folder locally, /home/user/app on hf spaces
DB_PATH = os.getenv('DB_PATH', os.path.join(os.path.dirname(__file__), 'finetunex.db'))


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename VARCHAR(255),
                base_model VARCHAR(100),
                total_examples INTEGER,
                epochs INTEGER,
                final_loss REAL,
                lora_rank INTEGER,
                adapter_path VARCHAR(500),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        cursor.close()
        conn.close()
        print("database ready!")
    except Exception as e:
        print(f"db error: {e}")


def save_run(filename, base_model, total_examples, epochs, final_loss, lora_rank, adapter_path):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO training_runs (filename, base_model, total_examples, epochs, final_loss, lora_rank, adapter_path) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (filename, base_model, total_examples, epochs, final_loss, lora_rank, adapter_path)
        )
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"error saving run: {e}")


def get_history():
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM training_runs ORDER BY created_at DESC LIMIT 20")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        results = []
        for row in rows:
            row_dict = dict(row)
            if row_dict.get("created_at"):
                row_dict["created_at"] = str(row_dict["created_at"])
            results.append(row_dict)
        return results
    except Exception as e:
        print(f"error getting history: {e}")
        return []
