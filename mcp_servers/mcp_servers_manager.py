import asyncio
import logging
from typing import Dict, List, Tuple
from agents.mcp import MCPServerStdio
from mcp_servers.mcp_params import (
    researcher_mcp_server_params,
    fundamental_mcp_server_params,
    technical_mcp_server_params,
    risk_mcp_server_params,
    memory_mcp_server_params,
    accounts_mcp_server_params,
    decision_mcp_server_params,
    execution_mcp_server_params
)

class MCPServerManager:
    """Manages MCP server connections with pooling"""
    
    _instance = None
    _pools: Dict[str, List[MCPServerStdio]] = {}
    _locks: Dict[str, asyncio.Lock] = {}
    _logger = logging.getLogger(__name__)

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._locks = {}
        return cls._instance

    async def get_servers(self, server_type: str, params_list: list) -> List[MCPServerStdio]:
        """Get or create server connections from pool with proper locking"""
        if server_type not in self._locks:
            self._locks[server_type] = asyncio.Lock()
            
        async with self._locks[server_type]:
            if server_type not in self._pools:
                self._pools[server_type] = []
                for params in params_list:
                    try:
                        server = MCPServerStdio(
                            params, 
                            client_session_timeout_seconds=60,
                            cache_tools_list=True
                        )
                        await server.connect()
                        self._pools[server_type].append(server)
                    except Exception as e:
                        self._logger.error(f"Server connection failed - {server_type}: {e}")
            return self._pools[server_type]

    async def get_researcher_servers(self) -> Tuple[List[MCPServerStdio], List[MCPServerStdio]]:
        """Get researcher and memory servers"""
        researcher = await self.get_servers('researcher', researcher_mcp_server_params)
        memory = await self.get_servers('memory', memory_mcp_server_params)
        return researcher, memory


    async def get_fundamental_servers(self) -> List[MCPServerStdio]:
        """Get fundamental analysis servers"""
        return await self.get_servers('fundamental', fundamental_mcp_server_params)


    async def get_technical_servers(self) -> List[MCPServerStdio]:
        """Get technical analysis servers"""
        return await self.get_servers('technical', technical_mcp_server_params)


    async def get_risk_servers(self) -> List[MCPServerStdio]:
        """Get risk management servers"""
        return await self.get_servers('risk', risk_mcp_server_params)


    async def get_memory_servers(self) -> List[MCPServerStdio]:
        """Get memory servers"""
        return await self.get_servers('memory', memory_mcp_server_params)


    async def get_account_servers(self) -> List[MCPServerStdio]:
        """Get account management servers"""
        return await self.get_servers('account', accounts_mcp_server_params)


    async def get_decision_servers(self) -> List[MCPServerStdio]:
        """Get decision servers"""
        return await self.get_servers('decision', decision_mcp_server_params)


    async def get_execution_servers(self) -> List[MCPServerStdio]:
        """Get execution servers"""
        return await self.get_servers('execution', execution_mcp_server_params)
    

mcp_manager = MCPServerManager()

# Convenience functions that use the manager internally
async def setup_researcher_mcp_servers():
    return await mcp_manager.get_researcher_servers()


async def setup_fundamental_mcp_servers():
    return await mcp_manager.get_fundamental_servers()


async def setup_technical_mcp_servers():
    return await mcp_manager.get_technical_servers()


async def setup_risk_mcp_servers():
    return await mcp_manager.get_risk_servers()


async def setup_memory_mcp_servers():
    return await mcp_manager.get_memory_servers()


async def setup_account_mcp_servers():
    return await mcp_manager.get_account_servers()


async def setup_decision_mcp_servers():
    return await mcp_manager.get_decision_servers()


async def setup_execution_mcp_servers():
    return await mcp_manager.get_execution_servers()





# import asyncio
# from agents.mcp import MCPServerStdio
# from mcp_servers.mcp_params import (
#     researcher_mcp_server_params,
#     fundamental_mcp_server_params,
#     technical_mcp_server_params,
#     risk_mcp_server_params,
#     memory_mcp_server_params,
#     accounts_mcp_server_params,
#     execution_mcp_server_params
# )

# async def setup_researcher_mcp_servers():
#     """Setup and connect researcher MCP servers."""
#     researcher_servers = []
#     memory_servers = []
    
#     # Setup researcher servers
#     for params in researcher_mcp_server_params:
#         server = MCPServerStdio(params, client_session_timeout_seconds=30)
#         await server.connect()
#         researcher_servers.append(server)
    
#     # Setup memory servers
#     for params in memory_mcp_server_params:
#         server = MCPServerStdio(params, client_session_timeout_seconds=30)
#         await server.connect()
#         memory_servers.append(server)
    
#     return researcher_servers, memory_servers


# async def setup_fundamental_mcp_servers():
#     """Setup and connect fundamental analysis MCP servers."""
#     servers = []
#     for params in fundamental_mcp_server_params:
#         server = MCPServerStdio(params, client_session_timeout_seconds=30)
#         await server.connect()
#         servers.append(server)
#     return servers


# async def setup_technical_mcp_servers():
#     """Setup and connect technical analysis MCP servers."""
#     servers = []
#     for params in technical_mcp_server_params:
#         server = MCPServerStdio(params, client_session_timeout_seconds=30)
#         await server.connect()
#         servers.append(server)
#     return servers


# async def setup_risk_mcp_servers():
#     """Setup and connect risk management MCP servers."""
#     servers = []
#     for params in risk_mcp_server_params:
#         server = MCPServerStdio(params, client_session_timeout_seconds=30)
#         await server.connect()
#         servers.append(server)
#     return servers


# async def setup_memory_mcp_servers():
#     """Setup and connect memory MCP servers."""
#     servers = []
#     for params in memory_mcp_server_params:
#         server = MCPServerStdio(params, client_session_timeout_seconds=30)
#         await server.connect()
#         servers.append(server)
#     return servers


# async def setup_account_mcp_servers():
#     """Setup and connect account management MCP servers."""
#     servers = []
#     for params in accounts_mcp_server_params:
#         server = MCPServerStdio(params, client_session_timeout_seconds=30)
#         await server.connect()
#         servers.append(server)
#     return servers


# async def setup_execution_mcp_servers():
#     """Setup and connect execution MCP servers."""
#     servers = []
#     for params in execution_mcp_server_params:
#         server = MCPServerStdio(params, client_session_timeout_seconds=30)
#         await server.connect()
#         servers.append(server)
#     return servers