import sqlite3
import redis

# Connect to SQLite
conn = sqlite3.connect('accounts.db')
cursor = conn.cursor()

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, db=0)

# Fetch all scripts
cursor.execute("SELECT SECURITY_ID, UNDERLYING_SYMBOL, SYMBOL_NAME, DISPLAY_NAME FROM scripts WHERE EXCH_ID = 'NSE' AND INSTRUMENT = 'EQUITY'")
rows = cursor.fetchall()

# Store in Redis
for id, symbol, name, display in rows:
    if symbol:
        r.hset('symbol_index', symbol.lower(), symbol)
    if id:
        r.hset('id_index', id.lower(), symbol)
    if name:
        r.hset('name_index', name.lower(), symbol)
    if display:
        r.hset('display_index', display.lower(), symbol)

print("Loaded all symbols into Redis.")