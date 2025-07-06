import os
import json
from contextlib import AsyncExitStack
from mcp_servers.accounts_client import read_accounts_resource, read_strategy_resource
from memory.agent_memory import AgentMemory
from trade_agents.tracers import make_trace_id
from agents import Agent, Tool, Runner, OpenAIChatCompletionsModel, trace
from openai import AsyncOpenAI
from dotenv import load_dotenv
from agents.mcp import MCPServerStdio
from risk_management.trade_execution import execute_buy, execute_sell
from dataclasses import asdict
from utils.redis_client import main_redis_client
from trade_agents.templates import researcher_instructions, trader_instructions, trade_message, rebalance_message, research_tool
from mcp_servers.mcp_params import trader_mcp_server_params, researcher_mcp_server_params

load_dotenv(override=True)


deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")


DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

MAX_TURNS = 30

deepseek_client = AsyncOpenAI(base_url=DEEPSEEK_BASE_URL, api_key=deepseek_api_key)
gemini_client = AsyncOpenAI(base_url=GEMINI_BASE_URL, api_key=google_api_key)


def get_model(model_name: str):
    if "gemini" in model_name:
        return OpenAIChatCompletionsModel(model=model_name, openai_client=gemini_client)
    elif "deepseek" in model_name:
        return OpenAIChatCompletionsModel(model=model_name, openai_client=deepseek_client)
    else:
        return model_name


async def get_researcher(mcp_servers, model_name) -> Agent:
    researcher = Agent(
        name="Researcher",
        instructions=researcher_instructions(),
        model=get_model(model_name),
        mcp_servers=mcp_servers,
    )
    return researcher


async def get_researcher_tool(mcp_servers, model_name) -> Tool:    
    researcher = await get_researcher(mcp_servers, model_name)
    return researcher.as_tool(
            tool_name="Researcher",
            tool_description=research_tool()
        )


class Trader:
    def __init__(self, name: str, lastname="Trader", model_name="gpt-4o-mini", position_manager=None):
        self.name = name
        self.lastname = lastname
        self.agent = None
        self.model_name = model_name
        self.do_trade = True
        self.position_manager = position_manager
        self.position_manager = None 
        self.stop_loss_manager = None
        self.agent_memory = AgentMemory(self.name)       
        self.redis_client = main_redis_client        

    async def trade(self, symbol: str, entry_price: float, stop_loss: float, target: float, quantity: int = None, rationale: str = "", position_type: str = "LONG", sector: str = None):
        """
        Unified trade execution using trade_execution.py
        """
        from data.accounts import Account
        account = Account.get(self.name)
        
        result = execute_buy(
            agent_name=self.name,
            account=account,
            position_manager=self.position_manager,
            symbol=symbol,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target=target,
            quantity=quantity,
            rationale=rationale,
            position_type=position_type,
            sector=sector
        )
        if result.get("success"):
            self.stop_loss_manager.set_stop_loss(
                symbol=symbol,
                stop_price=stop_loss,
                quantity=quantity,
                agent_name=self.name,
                trailing=True,  # True for trailing
                trail_percent=20  # set if trailing
            )
        await self.after_trade(result)
        return result

    async def close_position(self, symbol: str, execution_price: float, rationale: str = "Close position"):
        from data.accounts import Account
        account = Account.get(self.name)
        result = execute_sell(
            agent_name=self.name,
            account=account,
            position_manager=self.position_manager,
            symbol=symbol,
            quantity=self.position_manager.positions[symbol].quantity if symbol in self.position_manager.positions else 0,
            rationale=rationale,
            execution_price=execution_price
        )
        await self.after_trade(result)
        return result


    async def create_agent(self, trader_mcp_servers, researcher_mcp_servers) -> Agent:
        tool = await get_researcher_tool(researcher_mcp_servers, self.model_name)
        self.agent = Agent(
            name=self.name,
            instructions=trader_instructions(self.name),
            model=get_model(self.model_name),
            tools=[tool],
            mcp_servers=trader_mcp_servers
        )
        return self.agent
    

    async def get_account_report(self) -> str:
        account = await read_accounts_resource(self.name)
        account_json = json.loads(account)
        account_json.pop("portfolio_value_time_series", None)
        return json.dumps(account_json)

    async def after_trade(self, trade_details: dict):        
        self.agent_memory.log_trade(trade_details)        
        positions_dict = {symbol: asdict(pos) for symbol, pos in self.position_manager.positions.items()}
        self.agent_memory.store_active_positions(positions_dict)        
        context = {"summary": "Trade executed", "details": trade_details}
        self.agent_memory.store_daily_context(context)

    async def before_decision(self):        
        context = self.agent_memory.get_daily_context()
        positions = self.agent_memory.get_active_positions()
        watchlist = self.agent_memory.get_watchlist()
        return {
            "context": context,
            "positions": positions,
            "watchlist": watchlist
        }
    

    async def run_agent(self, trader_mcp_servers, researcher_mcp_servers):
        self.agent = await self.create_agent(trader_mcp_servers, researcher_mcp_servers)
        account = await self.get_account_report()
        strategy = await read_strategy_resource(self.name)
        memory_context = await self.before_decision()

        message = trade_message(
        self.name, strategy, account,
        positions=memory_context["positions"],
        watchlist=memory_context["watchlist"],
        context=memory_context["context"]
    ) if self.do_trade else rebalance_message(
        self.name, strategy, account,
        positions=memory_context["positions"],
        watchlist=memory_context["watchlist"],
        context=memory_context["context"]
    )
        

        await Runner.run(self.agent, message, max_turns=MAX_TURNS)


    async def run_with_mcp_servers(self):
        async with AsyncExitStack() as stack:
            trader_mcp_servers = [await stack.enter_async_context(MCPServerStdio(params, client_session_timeout_seconds=30)) for params in trader_mcp_server_params]
            async with AsyncExitStack() as stack:
                researcher_mcp_servers = [await stack.enter_async_context(MCPServerStdio(params, client_session_timeout_seconds=30)) for params in researcher_mcp_server_params]
                await self.run_agent(trader_mcp_servers, researcher_mcp_servers)
    

    async def run_with_trace(self):
        trace_name = f"{self.name}-trading" if self.do_trade else f"{self.name}-rebalancing"
        trace_id = make_trace_id(f"{self.name.lower()}")
        with trace(trace_name, trace_id=trace_id):
            await self.run_with_mcp_servers()
    

    async def run(self):
        try:
            await self.run_with_trace()
        except Exception as e:
            print(f"Error running trader {self.name}: {e}")
        self.do_trade = not self.do_trade

