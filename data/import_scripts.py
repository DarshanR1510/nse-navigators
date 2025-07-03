import sqlite3
import csv
from data.database import DB


def import_scripts_from_csv(csv_path: str):
    """
    Import filtered trading scripts data from a CSV file into the scripts table.
    Only rows with SEGMENT == 'E' and only selected columns are inserted.
    Args:
        csv_path (str): Path to the CSV file.
    """
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = [
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
            for row in reader if row['SEGMENT'] == 'E'
        ]
    with sqlite3.connect(DB) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM scripts")
        cursor.executemany('''
            INSERT INTO scripts VALUES (
                ?,?,?,?,?,?,?,?,?,?,?,?
            )
        ''', rows)
        conn.commit()

if __name__ == "__main__":
    import_scripts_from_csv("./api-scrip-master-detailed.csv")
