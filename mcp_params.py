import os
from dotenv import load_dotenv

load_dotenv(override=True)

brave_env = {"BRAVE_API_KEY": os.getenv("BRAVE_API_KEY")}

market_mcp = ({"command": "uv", "args": ["run", "market_server.py"]})


# The full set of MCP servers for the trader: Accounts, Push Notification and the Market
trader_mcp_server_params = [
    {"command": "uv", "args": ["run", "accounts_server.py"]},
    {"command": "uv", "args": ["run", "market_server.py"]},
    {"command": "uv", "args": ["run", "push_server.py"]}
]

# The full set of MCP servers for the researcher: Fetch, Brave Search and Memory
researcher_mcp_server_params = [
    {"command": "uvx", "args": ["mcp-server-fetch"]},
    {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-brave-search"], "env": brave_env},
    {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-memory"]}
]