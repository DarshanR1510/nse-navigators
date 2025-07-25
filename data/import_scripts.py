import sqlite3
import csv


DB = "accounts.db"

def import_scripts_from_csv(csv_path: str):
    """
    Import filtered trading scripts data from a CSV file into the scripts table.
    Only rows with EXCH_ID == 'NSE' and INSTRUMENT == 'EQUITY' and only selected columns are inserted.
    Args:
        csv_path (str): Path to the CSV file.
    """
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        scripts_rows = [
            (
                row['EXCH_ID'],
                row['SEGMENT'],
                row['SECURITY_ID'],
                row['ISIN'],
                row['INSTRUMENT'],
                row['UNDERLYING_SECURITY_ID'],
                row['UNDERLYING_SYMBOL'],
                row['SYMBOL_NAME'],
                row['DISPLAY_NAME'],
                row['INSTRUMENT_TYPE'],
                row['SERIES'],
                float(row['LOT_SIZE']) if row['LOT_SIZE'] else None
            )
            for row in reader 
            if row['EXCH_ID'] == 'NSE' 
            and row['INSTRUMENT'] == 'EQUITY'
            and not any(char.isdigit() for char in row['UNDERLYING_SYMBOL'])
        ]
        index_rows = [
            (
                row['EXCH_ID'],
                row['SEGMENT'],
                row['SECURITY_ID'],
                row['ISIN'],
                row['INSTRUMENT'],
                row['UNDERLYING_SECURITY_ID'],
                row['UNDERLYING_SYMBOL'],
                row['SYMBOL_NAME'],
                row['DISPLAY_NAME'],
                row['INSTRUMENT_TYPE'],
                row['SERIES'],
                float(row['LOT_SIZE']) if row['LOT_SIZE'] else None
            )
            for row in reader 
            if row['EXCH_ID'] == 'NSE' 
                and row['INSTRUMENT'] == 'INDEX' 
                and (
                'NIFTY' in row['UNDERLYING_SYMBOL'].upper() 
                or 'VIX' in row['UNDERLYING_SYMBOL'].upper()
            )   
        ]
    with sqlite3.connect(DB) as conn:
        cursor = conn.cursor()
        # Delete rows only if tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='scripts'")
        if cursor.fetchone():
            cursor.execute("DELETE FROM scripts")
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='indexes'")
        if cursor.fetchone():
            cursor.execute("DELETE FROM indexes")
        
        cursor.executemany('''
            INSERT INTO scripts VALUES (
                ?,?,?,?,?,?,?,?,?,?,?,?
            )
        ''', scripts_rows)
        cursor.executemany('''
            INSERT INTO indexes VALUES (
                ?,?,?,?,?,?,?,?,?,?,?,?
            )
        ''', index_rows)
        conn.commit()

if __name__ == "__main__":
    import_scripts_from_csv("utils/api-scrip-master-detailed.csv")
