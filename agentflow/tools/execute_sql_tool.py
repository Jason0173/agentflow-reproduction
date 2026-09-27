import sqlite3, os

def execute_sql(sql_query: str) -> str:
    db_path = os.environ.get("SPIDER_DB_PATH", "")
    if not db_path:
        return "Error: SPIDER_DB_PATH not set."
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(sql_query)
        rows = cur.fetchmany(20)
        cols = [d[0] for d in (cur.description or [])]
        conn.close()
        if not rows:
            return "No results."
        header = " | ".join(cols)
        lines = [header, "-" * len(header)]
        lines += [" | ".join(str(v) for v in row) for row in rows]
        return "
".join(lines)
    except Exception as e:
        return f"SQL Error: {e}"

TOOL_META = {"name": "execute_sql", "description": "Execute SQL query against SQLite database.", "input_description": "A valid SQL query string.", "function": execute_sql}
