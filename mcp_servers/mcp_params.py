import os
from dotenv import load_dotenv

load_dotenv(override=True)

brave_env = {"BRAVE_API_KEY": os.getenv("BRAVE_API_KEY")}
serper_env = {"SERPER_API_KEY": os.getenv("SERPER_API_KEY")}

# The full set of MCP servers for the trader: Accounts, Push Notification and the Market
trader_mcp_server_params = [
    {"command": "python", "args": ["-m", "mcp_servers.accounts_server"]},
    {"command": "python", "args": ["-m", "mcp_servers.market_server"]},
    {"command": "python", "args": ["-m", "mcp_servers.memory_server"]},
    {"command": "python", "args": ["-m", "mcp_servers.risk_server"]},
    {"command": "python", "args": ["-m", "mcp_servers.push_server"]},
]

# The full set of MCP servers for the researcher: Fetch, Brave Search and Memory
researcher_mcp_server_params = [
    {"name": "fetch", "command": "uvx", "args": ["mcp-server-fetch"]},
    {"name": "serper-search", "command": "npx", "args": ["-y", "serper-search-scrape-mcp-server"], "env": serper_env},
    {"name": "brave-search", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-brave-search"], "env": brave_env},
    {"name": "memory", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-memory"]}
]