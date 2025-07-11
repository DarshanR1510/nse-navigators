import logging
from mcp.server.fastmcp import FastMCP
from data.accounts import Account
from risk_management.trade_execution import execute_buy, execute_sell
from market_tools.market import get_symbol_price_impl
from trading_floor import position_managers


mcp = FastMCP("accounts_server")
logger = logging.getLogger(__name__)


@mcp.tool()
async def get_balance(name: str) -> float:
    """Get the cash balance of the given account name.

    Args:
        name: The name of the account holder
    """
    return Account.get(name).balance

@mcp.tool()
async def get_holdings(name: str) -> dict[str, int]:
    """Get the holdings of the given account name.

    Args:
        name: The name of the account holder
    """
    return Account.get(name).holdings

@mcp.tool()
async def buy_shares(name: str, symbol: str, quantity: int, stop_loss: float, rationale: str) -> float:
    """Buy shares of a stock.

    Args:
        name: The name of the account holder
        symbol: The symbol of the stock
        quantity: The quantity of shares to buy
        stop_loss: The stop loss price for the purchase (Use m_calculate_position_size to determine this)
        rationale: The rationale for the purchase and fit with the account's strategy
    """
    account = Account.get(name)    
    normalized_name = name.strip().lower()

    entry_price = get_symbol_price_impl(symbol)
    target = (entry_price - stop_loss) * 2 + entry_price #1:2 risk-reward ratio
    
    position_manager = position_managers.get(normalized_name)

    success, msg = execute_buy(
        agent_name=name,
        account=account,
        position_manager=position_manager,
        symbol=symbol,
        entry_price=entry_price,
        stop_loss=stop_loss,
        target=target,
        quantity=quantity,
        rationale=rationale,
    )
    return msg


@mcp.tool()
async def sell_shares(name: str, symbol: str, quantity: int, rationale: str) -> float:
    """Sell shares of a stock.

    Args:
        name: The name of the account holder        
        symbol: The symbol of the stock
        quantity: The quantity of shares to sell
        rationale: The rationale for the sale and fit with the account's strategy
    """
    account = Account.get(name)        
    execution_price = get_symbol_price_impl(symbol)
    position_manager = position_managers.get(name.strip().lower())
    logger.info(f"Position manager in sell for {name}: {position_manager}")

    success, msg = execute_sell(
        agent_name=name,
        account=account,
        position_manager=position_manager,
        symbol=symbol,
        quantity=quantity,
        rationale=rationale,
        execution_price=execution_price
    )
    return msg

@mcp.tool()
async def change_strategy(name: str, strategy: str) -> str:
    """At your discretion, if you choose to, call this to change your investment strategy for the future.

    Args:
        name: The name of the account holder
        strategy: The new strategy for the account
    """
    return Account.get(name).change_strategy(strategy)

@mcp.resource("accounts://accounts_server/{name}")
async def read_account_resource(name: str) -> str:
    account = Account.get(name.lower())
    return account.report()

@mcp.resource("accounts://strategy/{name}")
async def read_strategy_resource(name: str) -> str:
    account = Account.get(name.lower())
    return account.get_strategy()

if __name__ == "__main__":
    mcp.run(transport='stdio')