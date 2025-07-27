import os
from dotenv import load_dotenv

load_dotenv(override=True)

serper_env = {"SERPER_API_KEY": os.getenv("SERPER_API_KEY")}
brave_env = {"BRAVE_API_KEY": os.getenv("BRAVE_API_KEY")}



researcher_mcp_server_params = [
    {"name": "fetch", "command": "uvx", "args": ["mcp-server-fetch"]},
    {"name": "serper-search", "command": "npx", "args": ["-y", "serper-search-scrape-mcp-server"], "env": serper_env},
    {"name": "brave-search", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-brave-search"], "env": brave_env}
]

memory_mcp_server_params = [
    {"command": "python", "args": ["-m", "mcp_servers.memory_server"]},
    {"name": "memory", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-memory"]},
]

fundamental_mcp_server_params = [
    {"command": "python", "args": ["-m", "mcp_servers.fundamental_server"]},
]

technical_mcp_server_params = [
    {"command": "python", "args": ["-m", "mcp_servers.technical_server"]},
]

risk_mcp_server_params = [
    {"command": "python", "args": ["-m", "mcp_servers.risk_server"]},
]

accounts_mcp_server_params = [
    {"command": "python", "args": ["-m", "mcp_servers.accounts_server"]},
]

push_mcp_server_params = [
    {"command": "python", "args": ["-m", "mcp_servers.push_server"]},
]

decision_mcp_server_params = [
    {"command": "python", "args": ["-m", "mcp_servers.decision_server"]},
]

execution_mcp_server_params = [
    {"command": "python", "args": ["-m", "mcp_servers.execution_server"]},
    push_mcp_server_params[0]
]
