import sqlite3
import json
from datetime import datetime
from dotenv import load_dotenv
import csv

load_dotenv(override=True)

DB = "accounts.db"


with sqlite3.connect(DB) as conn:
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE IF NOT EXISTS accounts (name TEXT PRIMARY KEY, account TEXT)')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            datetime DATETIME,
            type TEXT,
            message TEXT
        )
    ''')
    cursor.execute('CREATE TABLE IF NOT EXISTS market (date TEXT PRIMARY KEY, data TEXT)')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS shares (
            EXCH_ID TEXT,
            SEGMENT TEXT,
            SECURITY_ID TEXT,
            ISIN TEXT,
            INSTRUMENT TEXT,
            UNDERLYING_SECURITY_ID TEXT,
            UNDERLYING_SYMBOL TEXT,
            SYMBOL_NAME TEXT,
            DISPLAY_NAME TEXT,
            INSTRUMENT_TYPE TEXT,
            SERIES TEXT,
            LOT_SIZE INTEGER
        )
    ''')
    conn.commit()

##! Test query in terminal:
##* sqlite3 accounts.db "SELECT * FROM shares WHERE UNDERLYING_SYMBOL is 'BBOX';"


# --- QUERY CLASS ---
class DatabaseQueries:
    DB = DB

    @staticmethod
    def write_account(name, account_dict):
        json_data = json.dumps(account_dict)
        with sqlite3.connect(DatabaseQueries.DB) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO accounts (name, account)
                VALUES (?, ?)
                ON CONFLICT(name) DO UPDATE SET account=excluded.account
            ''', (name.lower(), json_data))
            conn.commit()

    @staticmethod
    def read_account(name):
        with sqlite3.connect(DatabaseQueries.DB) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT account FROM accounts WHERE name = ?', (name.lower(),))
            row = cursor.fetchone()
            return json.loads(row[0]) if row else None

    @staticmethod
    def write_log(name: str, type: str, message: str):
        now = datetime.now().isoformat()
        with sqlite3.connect(DatabaseQueries.DB) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO logs (name, datetime, type, message)
                VALUES (?, datetime('now'), ?, ?)
            ''', (name.lower(), type, message))
            conn.commit()

    @staticmethod
    def read_log(name: str, last_n=10):
        with sqlite3.connect(DatabaseQueries.DB) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT datetime, type, message FROM logs 
                WHERE name = ? 
                ORDER BY datetime DESC
                LIMIT ?
            ''', (name.lower(), last_n))
            return reversed(cursor.fetchall())

    @staticmethod
    def write_market(date: str, data: dict) -> None:
        data_json = json.dumps(data)
        with sqlite3.connect(DatabaseQueries.DB) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO market (date, data)
                VALUES (?, ?)
                ON CONFLICT(date) DO UPDATE SET data=excluded.data
            ''', (date, data_json))
            conn.commit()

    @staticmethod
    def read_market(date: str) -> dict | None:
        with sqlite3.connect(DatabaseQueries.DB) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT data FROM market WHERE date = ?', (date,))
            row = cursor.fetchone()
            return json.loads(row[0]) if row else None

    @staticmethod
    def get_equity_script(underlying_symbol: str):
        with sqlite3.connect(DatabaseQueries.DB) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM scripts WHERE EXCH_ID = 'NSE' AND UNDERLYING_SYMBOL = ? AND INSTRUMENT = 'EQUITY'
            ''', (underlying_symbol,))
            row = cursor.fetchone()
            return dict(row) if row else None
        
    @staticmethod
    def get_security_id(underlying_symbol: str):
        script = DatabaseQueries.get_equity_script(underlying_symbol)
        return script["SECURITY_ID"] if script else None
    
    @staticmethod
    def get_scripts_name_symbols():
        with sqlite3.connect(DatabaseQueries.DB) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT UNDERLYING_SYMBOL, SYMBOL_NAME, DISPLAY_NAME FROM scripts WHERE EXCH_ID = "NSE" AND INSTRUMENT = "EQUITY"')
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

if __name__ == "__main__":
    pass