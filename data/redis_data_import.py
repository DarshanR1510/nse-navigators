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

print(f"Fetched {len(rows)} rows from SQLite.")
# # Store in Redis
for security_id, symbol, name, display in rows:
    # Normalize missing values to None or empty string
    security_id = security_id if security_id is not None else ""
    symbol = symbol if symbol is not None else ""
    name = name if name is not None else ""
    display = display if display is not None else ""
    
    if symbol:  # Only store if symbol is present
        r.hset(f"symbol:{symbol.lower()}", mapping={
            "security_id": int(security_id),
            "symbol": symbol,
            "name": name,
            "display": display
        })

print("Loaded all symbols into Redis.")


# Example usage
data = r.hgetall('symbol:reliance')
print({k.decode(): v.decode() for k, v in data.items()})

print(f"{(r.hget('symbol:reliance', 'symbol')).decode()}")
print(f"{(r.hget('symbol:reliance', 'security_id')).decode()}")
print(f"{(r.hget('symbol:reliance', 'name')).decode()}")
print(f"{(r.hget('symbol:reliance', 'display')).decode()}")