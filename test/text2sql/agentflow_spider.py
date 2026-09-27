# Spider Text-to-SQL evaluation script
# Execution Accuracy: 0.350 (20-sample eval with Qwen2.5-7B)
import sqlite3, re, os, sys

DB_DIR = 'data/spider/database'

def get_schema(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute('SELECT name FROM sqlite_master WHERE type=\'table\'')
    tables = [r[0] for r in cur.fetchall()]
    lines = []
    for t in tables:
        cur.execute(f'PRAGMA table_info({t})')
        cols = [f'{r[1]} ({r[2]})' for r in cur.fetchall()]
        lines.append(f'Table {t}: {chr(44).join(cols)}')
    conn.close()
    return chr(10).join(lines)

def execute_sql(sql, db_path):
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        conn.close()
        return rows
    except Exception as e:
        return [("ERROR", str(e))]

