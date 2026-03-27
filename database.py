import mysql.connector
import os

# mysql for storing training run history

def get_connection():
    return mysql.connector.connect(
        host=os.getenv("MYSQL_HOST", "localhost"),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", ""),
        database=os.getenv("MYSQL_DATABASE", "finetunex")
    )


def init_db():
    # create db if it doesnt exist yet
    conn = mysql.connector.connect(
        host=os.getenv("MYSQL_HOST", "localhost"),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", "")
    )
    cursor = conn.cursor()
    cursor.execute("CREATE DATABASE IF NOT EXISTS finetunex")
    conn.close()

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS training_runs (
            id INT AUTO_INCREMENT PRIMARY KEY,
            filename VARCHAR(255),
            base_model VARCHAR(100),
            total_examples INT,
            epochs INT,
            final_loss FLOAT,
            lora_rank INT,
            adapter_path VARCHAR(500),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def save_run(filename, base_model, total_examples, epochs, final_loss, lora_rank, adapter_path):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO training_runs (filename, base_model, total_examples, epochs, final_loss, lora_rank, adapter_path) VALUES (%s, %s, %s, %s, %s, %s, %s)",
        (filename, base_model, total_examples, epochs, final_loss, lora_rank, adapter_path)
    )
    conn.commit()
    conn.close()


def get_history():
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM training_runs ORDER BY created_at DESC LIMIT 20")
    rows = cursor.fetchall()
    conn.close()

    for row in rows:
        if row.get("created_at"):
            row["created_at"] = row["created_at"].strftime("%Y-%m-%d %H:%M:%S")
    return rows
