"""Unit tests for the Spider evaluation helpers. No model or API key needed."""

import importlib.util
import sqlite3
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "test" / "text2sql" / "spider_eval.py"
spec = importlib.util.spec_from_file_location("spider_eval", SCRIPT)
spider_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(spider_eval)


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "concert.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE singer (singer_id INTEGER PRIMARY KEY, name TEXT, age INTEGER);
        INSERT INTO singer VALUES (1, 'Joe', 52), (2, 'Ann', 29), (3, 'Tim', 29);
        """
    )
    conn.commit()
    conn.close()
    return str(path)


def test_get_schema_lists_tables_and_columns(db):
    assert spider_eval.get_schema(db) == "Table singer: singer_id (INTEGER), name (TEXT), age (INTEGER)"


def test_extract_sql_prefers_fenced_block():
    text = "Plan...\nSELECT wrong FROM x\n```sql\nSELECT name FROM singer\n```\nDone."
    assert spider_eval.extract_sql(text) == "SELECT name FROM singer"


def test_extract_sql_falls_back_to_last_select_line():
    text = "First try: SELECT 1\nselect name from singer where age > 30\nThat is the answer."
    assert spider_eval.extract_sql(text) == "select name from singer where age > 30"


def test_extract_sql_returns_text_when_no_sql():
    assert spider_eval.extract_sql("  I don't know  ") == "I don't know"


def test_execute_sql_reports_errors_instead_of_raising(db):
    rows = spider_eval.execute_sql("SELECT nope FROM singer", db)
    assert rows[0][0] == "ERROR"


def test_results_match_ignores_row_order(db):
    gold = "SELECT name FROM singer WHERE age = 29 ORDER BY name"
    pred = "SELECT name FROM singer WHERE age < 30 ORDER BY name DESC"
    assert spider_eval.results_match(pred, gold, db)


def test_results_match_detects_wrong_rows(db):
    assert not spider_eval.results_match("SELECT name FROM singer", "SELECT name FROM singer WHERE age > 30", db)


def test_results_match_fails_on_broken_prediction(db):
    assert not spider_eval.results_match("SELEC name FROM singer", "SELECT name FROM singer", db)
