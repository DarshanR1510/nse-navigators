import os
from pydantic import BaseModel
import json
from dotenv import load_dotenv
from datetime import datetime
from data.database import DatabaseQueries
from market_tools.market import get_symbol_price_impl, get_prices_with_cache
import time
import logging
from data.schemas import IST
from trade_agents.strategies import warren_strategy, george_strategy, ray_strategy, cathie_strategy


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv(override=True)

# SPREAD = 0.002


all_holdings = []

class Transaction(BaseModel):
    symbol: str
    quantity: int
    price: float
    timestamp: str
    rationale: str

    def total(self) -> float:
        return self.quantity * self.price
    
    def __repr__(self):
        return f"{abs(self.quantity)} shares of {self.symbol} at {self.price} each."


class Account(BaseModel):
    name: str
    balance: float
    strategy: str
    holdings: dict[str, int]
    transactions: list[Transaction]
    portfolio_value_time_series: list[tuple[str, float]]

    @classmethod
    def get(cls, name: str):
        fields = DatabaseQueries.read_account(name.lower())        
        if not fields:
            fields = cls._initialize_new_account(name)
            # fields = {
            #     "name": name.lower(),
            #     "balance": float(os.getenv("INITIAL_BALANCE", 500000.0)),
            #     "strategy": "",
            #     "holdings": {},
            #     "transactions": [],
            #     "portfolio_value_time_series": []
            # }
            DatabaseQueries.write_account(name, fields)
        
        elif not fields.get("strategy"):
            fields = cls._initialize_new_account(name, existing_fields=fields)

        return cls(**fields)
    

    @classmethod
    def _initialize_new_account(cls, name: str, existing_fields=None) -> dict:
        """Initialize account with proper strategy based on trader name"""
        strategies = {
            "warren": warren_strategy,
            "george": george_strategy,
            "ray": ray_strategy,
            "cathie": cathie_strategy,
        }

        fields = existing_fields or {
            "name": name.lower(),
            "balance": float(os.getenv("INITIAL_BALANCE", 500000.0)),
            "holdings": {},
            "transactions": [],
            "portfolio_value_time_series": []
        }
        
        # Set strategy based on trader name
        fields["strategy"] = strategies.get(name.lower(), "")
        
        # Save to database
        DatabaseQueries.write_account(name.lower(), fields)
        logger.info(f"Initialized account {name} with strategy")
        
        return fields
    
    
    def save(self):
        DatabaseQueries.write_account(self.name.lower(), self.model_dump())


    def reset(self, strategy: str):
        self.balance = float(os.getenv("INITIAL_BALANCE", 500000.0))
        self.strategy = strategy
        self.holdings = {}
        self.transactions = []
        self.portfolio_value_time_series = []
        self.save()

        print("Something Dummy")
        logger.info(f"Reset account {self.name} with strategy length: {len(strategy)}")
        print(f"Account {self.name} has been reset with strategy: {self.strategy}")


    def deposit(self, amount: float):
        """ Deposit funds into the account. """
        if amount <= 0:
            raise ValueError("Deposit amount must be positive.")
        self.balance += amount
        print(f"Deposited ₹{amount}. New balance: ₹{self.balance}")
        self.save()
        

    def withdraw(self, amount: float):
        """ Withdraw funds from the account, ensuring it doesn't go negative. """
        if amount > self.balance:
            raise ValueError("Insufficient funds for withdrawal.")
        self.balance -= amount
        print(f"Withdrew ₹{amount}. New balance: ₹{self.balance}")
        self.save()


    def calculate_portfolio_value(self) -> float:
        """
        Calculate the total value of the user's portfolio using the global cached prices.
        """
        total_value = self.balance
        symbols = list(self.holdings.keys())
        prices = get_prices_with_cache(symbols)  # Use the global cache from market.py

        for symbol, quantity in self.holdings.items():
            price = prices.get(symbol.upper(), 0.0)
            total_value += price * quantity            
        return total_value


    def calculate_profit_loss(self, portfolio_value: float):
        """ Calculate profit or loss from the initial spend. """        
        return portfolio_value - float(os.getenv("INITIAL_BALANCE", 500000.0))


    def get_holdings(self):
        """ Report the current holdings of the user. """
        return self.holdings


    def get_profit_loss(self):
        """ Report the user's profit or loss at any point in time. """
        return self.calculate_profit_loss()


    def list_transactions(self):
        """ List all transactions made by the user. """
        return [transaction.model_dump() for transaction in self.transactions]
    
    
    def report(self) -> str:
        """ Return a json string representing the account.  """
        portfolio_value = self.calculate_portfolio_value()
        self.portfolio_value_time_series.append((datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"), portfolio_value))
        self.save()
        pnl = self.calculate_profit_loss(portfolio_value)
        data = self.model_dump()
        data["total_portfolio_value"] = portfolio_value
        data["total_profit_loss"] = pnl
        DatabaseQueries.write_log(self.name, "account", f"Retrieved account details")
        return json.dumps(data)
    

    def get_strategy(self) -> str:
        """ Return the strategy of the account """
        if not self.strategy:
            fields = self._initialize_new_account(self.name)
            self.strategy = fields["strategy"]
            self.save()                  
        return self.strategy
    

    def change_strategy(self, strategy: str) -> str:
        """ At your discretion, if you choose to, call this to change your investment strategy for the future """
        self.strategy = strategy
        self.save()
        DatabaseQueries.write_log(self.name, "account", f"Changed strategy")
        return "Changed strategy"


# Example of usage:
if __name__ == "__main__":
    account = Account("John Doe")
    account.deposit(1000)
    print(f"Current Holdings: {account.get_holdings()}")
    print(f"Total Portfolio Value: {account.calculate_portfolio_value()}")
    print(f"Profit/Loss: {account.get_profit_loss()}")
    print(f"Transactions: {account.list_transactions()}")