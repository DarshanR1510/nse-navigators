import logging
from mcp.server.fastmcp import FastMCP
from data.accounts import Account
from risk_management.trade_execution import execute_buy, execute_sell
from market_tools.market import get_symbol_price_impl
from trading_floor import position_managers
from risk_management.risk_management_tools import t_set_stop_loss_order
from data.database import DatabaseQueries


mcp = FastMCP("execution_server")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@mcp.tool()
async def buy_shares(name: str, symbol: str, quantity: int, stop_loss: float, target_price: float, rationale: str) -> str:
    """Buy shares of a stock.

    Args:
        name: The name of the account holder (trader)
        symbol: The symbol of the stock
        entry_price: The entry price for the purchase
        target_price: The target price for the purchase
        quantity: The quantity of shares to buy
        stop_loss: The stop loss price for the purchase 
        rationale: The rationale for the purchase and fit with the account's strategy
    """

    try:
        normalized_name = name.lower().strip()
        entry_price = get_symbol_price_impl(symbol)
        position_manager = position_managers.get(normalized_name)

        if not position_manager:
            error_msg = f"No position manager found for trader {normalized_name}"
            logger.error(error_msg)
            return error_msg

        logger.info(f"Executing buy for {normalized_name} using position manager: {position_manager}")

        success, msg = execute_buy(
            trader_name=normalized_name,
            symbol=symbol,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target=target_price,
            quantity=quantity,
            rationale=rationale,
            position_manager=position_manager,
        )

        if success:
            logging.info(f"Execution via MCP to buy {symbol} got {success}, {msg}")
            DatabaseQueries.write_log(normalized_name, "response", f"Buy {symbol}")

        return msg
        
    except Exception as e:
        error_msg = f"Error executing buy: {str(e)}"
        logger.error(error_msg)
        return error_msg


@mcp.tool()
async def sell_shares(name: str, symbol: str, quantity: int, rationale: str) -> str:
    """Sell shares of a stock.

    Args:
        name: The name of the account holder        
        symbol: The symbol of the stock
        quantity: The quantity of shares to sell
        rationale: The rationale for the sale and fit with the account's strategy
    """
    normalized_name = name.lower().strip()
    account = Account.get(normalized_name)
    execution_price = get_symbol_price_impl(symbol)
    position_manager = position_managers.get(normalized_name)
    logger.info(f"Position manager in sell for {normalized_name}: {position_manager}")

    success, msg = execute_sell(
        trader_name=normalized_name,
        account=account,
        position_manager=position_manager,
        symbol=symbol,
        quantity=quantity,
        rationale=rationale,
        execution_price=execution_price
    )
    return msg


@mcp.tool()
async def m_set_stop_loss_order(symbol: str, stop_price: float, quantity: int, trader_name: str):
    """
    Set a stop loss order for a specific stock symbol in the trader's portfolio.
    Args:
        symbol: The trading symbol for which to set the stop loss.
        stop_price: The stop loss price to trigger a sell.
        quantity: The number of shares/contracts to protect.
        trader_name: The name of the trader placing the stop loss order.
    Returns:
        Confirmation of stop loss order placement.
    """
    return await t_set_stop_loss_order(symbol, stop_price, quantity, trader_name=trader_name)

if __name__ == "__main__":
    mcp.run(transport='stdio')