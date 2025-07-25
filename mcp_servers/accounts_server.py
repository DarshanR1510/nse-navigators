import logging
from mcp.server.fastmcp import FastMCP
from data.accounts import Account


mcp = FastMCP("accounts_server")
logger = logging.getLogger(__name__)

print("Starting accounts server...")


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
    try:
        account = Account.get(name.lower())
        strategy = account.get_strategy()
        
        if not strategy:
            logger.error(f"Failed to initialize strategy for {name}")
            return ""
        return strategy
    
    except Exception as e:
        logger.error(f"Error reading strategy for {name}: {e}")
        raise

if __name__ == "__main__":
    mcp.run(transport='stdio')