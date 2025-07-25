import mcp
from mcp.client.stdio import stdio_client
from mcp import StdioServerParameters
from agents import FunctionTool
import json

params = StdioServerParameters(command="python", args=["-m", "mcp_servers.accounts_server"], env=None)

async def list_accounts_tools():
    async with stdio_client(params) as streams:
        async with mcp.ClientSession(*streams) as session:
            await session.initialize()
            tools_result = await session.list_tools()
            return tools_result.tools
        
async def call_accounts_tool(tool_name, tool_args):
    async with stdio_client(params) as streams:
        async with mcp.ClientSession(*streams) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, tool_args)
            return result
            
            
async def read_accounts_resource(name):
    async with stdio_client(params) as streams:
        async with mcp.ClientSession(*streams) as session:
            await session.initialize()
            result = await session.read_resource(f"accounts://accounts_server/{name}")
            return result.contents[0].text
        

async def read_strategy_resource(name):    
    async with stdio_client(params) as streams:
        async with mcp.ClientSession(*streams) as session:
            await session.initialize()
            result = await session.read_resource(f"accounts://strategy/{name}")
            return result.contents[0].text


async def get_accounts_tools_openai():
    openai_tools = []
    for tool in await list_accounts_tools():
        schema = {**tool.inputSchema, "additionalProperties": False}
        openai_tool = FunctionTool(
            name=tool.name,
            description=tool.description,
            params_json_schema=schema,
            on_invoke_tool=lambda ctx, args, toolname=tool.name: call_accounts_tool(toolname, json.loads(args))
                
        )
        openai_tools.append(openai_tool)
    return openai_tools


# import mcp
# from mcp.client.stdio import stdio_client
# from mcp import StdioServerParameters
# from agents import FunctionTool
# import json
# from typing import Dict
# import asyncio
# import logging

# logger = logging.getLogger(__name__)
# params = StdioServerParameters(command="python", args=["-m", "mcp_servers.accounts_server"], env=None)


# class AccountsClient:
#     """Client for accounts server using shared session"""
    
#     _mcp_sessions: Dict[str, mcp.ClientSession] = {}
#     _lock = asyncio.Lock()

#     @classmethod
#     async def get_mcp_session(cls, trader_name: str) -> mcp.ClientSession:
#         """Get or create MCP session for trader"""
#         async with cls._lock:

#             print(f"current MCP sessions: {list(cls._mcp_sessions.keys())}")
            
#             if trader_name not in cls._mcp_sessions:
#                 streams = await stdio_client(params).__aenter__()
#                 session = await mcp.ClientSession(*streams).__aenter__()
#                 await session.initialize()
#                 cls._mcp_sessions[trader_name] = session

#             logger.info(f"Created new MCP session for trader {trader_name}")
#             logger.info(f"Current MCP sessions: {list(cls._mcp_sessions.keys())}")
#             return cls._mcp_sessions[trader_name]


# async def call_accounts_tool(tool_name: str, tool_args: dict, trader_name: str):
#     """Call account tool using trader's session"""
#     session = await AccountsClient.get_mcp_session(trader_name)
#     return await session.call_tool(tool_name, tool_args)

# async def list_accounts_tools(trader_name: str):
#     """List tools using trader's session"""
#     session = await AccountsClient.get_mcp_session(trader_name)
#     tools_result = await session.list_tools()
#     return tools_result.tools


# async def read_accounts_resource(name: str) -> str:
#     """Read account resource using trader's session"""
#     try:
#         session = await AccountsClient.get_mcp_session(name)
#         result = await session.read_resource(f"accounts://accounts_server/{name}")
#         return result.contents[0].text
#     except Exception as e:
#         logger.error(f"Error reading account resource for {name}: {e}")
#         raise


# # async def read_strategy_resource(name: str) -> str:
# #     """Read strategy resource using trader's session"""
# #     # try:
# #     session = await AccountsClient.get_mcp_session(name)
# #     result = await session.read_resource(f"accounts://strategy/{name}")
# #     return result.contents[0].text
# #     # except Exception as e:
# #     #     logger.error(f"Error reading strategy resource for {name}: {e}")
# #     #     raise

# async def read_strategy_resource(name):
#     async with stdio_client(params) as streams:
#         async with mcp.ClientSession(*streams) as session:
#             await session.initialize()
#             result = await session.read_resource(f"accounts://strategy/{name}")
#             return result.contents[0].text


# async def get_accounts_tools_openai(trader_name: str):
#     """Get OpenAI tools using trader's session"""
#     openai_tools = []
#     for tool in await list_accounts_tools(trader_name):
#         schema = {**tool.inputSchema, "additionalProperties": False}
#         openai_tool = FunctionTool(
#             name=tool.name,
#             description=tool.description,
#             params_json_schema=schema,
#             on_invoke_tool=lambda ctx, args, tn=tool.name: call_accounts_tool(tn, json.loads(args), trader_name)
#         )
#         openai_tools.append(openai_tool)
#     return openai_tools