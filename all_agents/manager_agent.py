from typing import Dict, Any, Set, List
from datetime import datetime
from agents import Runner, trace
from mcp_servers.accounts_client import read_accounts_resource
from memory import memory_tools
from mcp_servers.mcp_servers_manager import setup_memory_mcp_servers, mcp_manager
from data.schemas import SelectedSymbols, RiskLimits, SymbolsAnalysis
from data.accounts import Account
from data.database import DatabaseQueries
from agents.agent import Agent
from trade_agents.models import get_model
from trade_agents.templates import manager_instructions, manager_decision_instructions
from risk_management.position_manager import PositionManager
from all_agents.base_trade_agent import AgentConfig, BaseTradeAgent
import logging
import traceback
import json
import os
from data.schemas import IST

class ManagerAgent(BaseTradeAgent):
    """
    Manager Agent responsible for high-level trading decisions and portfolio monitoring.
    Has two main roles:
    1. Strategic decision making for trading workflow
    2. Portfolio monitoring and oversight
    """

    def __init__(self, config: AgentConfig):
        """Initialize base configuration"""
        super().__init__(config)
        self.position_manager = None
        self.watchlist = None
        self.context = None
        self.account = None
        self.positions = None
        self.mcp_servers = None
        self.decision_agent = None
        self.monitor_agent = None
        self.logger = logging.getLogger(__name__)
        self.risk_limits = RiskLimits
        self.check_market_hours = os.getenv("CHECK_MARKET_HOURS", "False") == "False"
        self.VALID_DECISIONS: Set[str] = {"RESEARCH", "MONITOR", "REBALANCE"}


    @classmethod
    async def create(cls, config: AgentConfig, position_manager: PositionManager):
        """Factory method to create ManagerAgent instance with proper initialization"""
        
        instance = cls(config)        
        
        watchlist = await memory_tools.get_watchlist(config.name)
        context = memory_tools.get_overall_market_context()
        
        account = Account.get(config.name)
        account_report = account.report()
        
        BaseTradeAgent.__init__(instance, config)
        await instance.initialize(position_manager, watchlist, context, json.loads(account_report))
        return instance


    async def initialize(self,
                        position_manager: PositionManager,
                        watchlist: dict,
                        context: dict,
                        account: Account):
        
        # Core components
        self.position_manager = position_manager
        self.watchlist = watchlist
        self.context = context
        self.account = account
        self.positions = await memory_tools.get_positions(self.name)
        
        # Setup MCP servers and agents
        self.mcp_servers = await mcp_manager.get_memory_servers()
        self.decision_agent = self._create_decision_agent()
        self.monitor_agent = self._create_monitor_agent()

        # Configuration
        self.logger = logging.getLogger(__name__)
        self.risk_limits = RiskLimits
        self.check_market_hours = os.getenv("CHECK_MARKET_HOURS", "False") == "False"
        self.VALID_DECISIONS: Set[str] = {"RESEARCH", "MONITOR", "REBALANCE"}


    def _create_decision_agent(self) -> Agent:
        """Creates agent specifically for strategic decisions"""
        
        return Agent(
            name=f"{self.name}_manager_decision",
            instructions=manager_decision_instructions(),
            model=get_model(self.model_name),
            mcp_servers=self.mcp_servers            
        )


    def _create_monitor_agent(self) -> Agent:
        """Creates agent specifically for monitoring"""
        return Agent(
            name=f"{self.name}_manager_monitor",
            instructions=manager_instructions(
                self.name, 
                self.strategy,
                self.account,
                self.positions,
                self.watchlist,
                self.context
            ),
            model=get_model(self.model_name),
            mcp_servers=self.mcp_servers
        )


    async def decide_next_action(self, context: Dict[str, Any]) -> str:
        """High-level strategic decision making for trading workflow"""
        
        try:
            if not await self._perform_pre_workflow_checks():
                self.logger.warning("Pre-workflow checks failed, defaulting to MONITOR")
                return "MONITOR"

            decision_context = await self._prepare_decision_context(context)            
            serialized_context = self._serialize_context(decision_context)        
            
            decision = await self._get_llm_decision(serialized_context)                

            if "```json" in decision.lower():
                decision = decision.replace("```JSON", "```json")
                decision = decision.split("```json")[1].split("```")[0].strip()
            
            elif "```" in decision:
                decision = decision.split("```")[1].split("```")[0].strip()
                # Remove "JSON\n" if present at the start
                if decision.startswith("JSON\n"):
                    decision = decision[len("JSON\n"):].strip()

            if decision == "RESEARCH" and not await self._should_seek_new_positions(context.get("portfolio_status", {})):
                self.logger.info("Research decision overridden due to position constraints")
                return "MONITOR"

            return decision

        except Exception as e:
            self.logger.error(f"Decision making failed: {str(e)}")
            return "MONITOR"


    async def monitor_portfolio(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Portfolio monitoring and risk oversight"""
        try:
            monitoring_context = await self._prepare_monitoring_context(context)
            
            monitoring_result = await Runner.run(
                self.monitor_agent,
                input=json.dumps(monitoring_context),
                session=self.session,
                max_turns=15
            )

            return self._process_monitoring_result(monitoring_result)

        except Exception as e:
            self.logger.error(f"Portfolio monitoring failed: {str(e)}")
            return {"status": "ERROR", "error": str(e)}


    async def _get_llm_decision(self, serialized_context: str) -> str:
        """Get decision from LLM based on serialized context"""
        try:
            # Run agent with cached decision agent
            if not hasattr(self, '_cached_decision_agent'):
                self._cached_decision_agent = self._create_decision_agent()
                
            response = await Runner.run(
                self._cached_decision_agent,
                input=serialized_context,
                session=self.session,
                max_turns=15
            )

            if not response or not hasattr(response, 'final_output'):
                raise ValueError("Invalid response from LLM")

            # Extract decision from response
            output = response.final_output
            decision = ""

            # Handle dictionary response
            if isinstance(output, dict):
                decision = output.get("decision", "").strip().upper()
            
            # Handle string response
            elif isinstance(output, str):
                try:
                    # Try parsing as JSON first
                    parsed = json.loads(output)
                    decision = parsed.get("decision", "").strip().upper()
                except json.JSONDecodeError:
                    # If not JSON, use string directly
                    decision = output.strip().upper()

            # Validate decision
            if not decision or decision not in self.VALID_DECISIONS:
                self.logger.warning(f"Invalid decision received: {decision}")
                return "MONITOR"

            self.logger.info(f"LLM decision made: {decision}")
            DatabaseQueries.write_log(self.name, "generation", f"LLM has decided to do {decision}")
            return decision

        except Exception as e:
            self.logger.error(f"LLM decision making failed: {str(e)}")
            return "MONITOR"
            

    async def _prepare_decision_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare context for decision making"""
        try:
            DatabaseQueries.write_log(self.name, "generation", f"Preparing decision context for {self.name}")
            logging.info(f"Preparing decision context for {self.name}")
            return {
                "current_state": {
                    "portfolio": {
                        "status": context.get("portfolio_status", {}),
                        "risk_level": (await self._calculate_portfolio_value_risk())["total_risk"],
                        "positions": await memory_tools.get_positions(self.name),
                        "utilization": context.get("portfolio_status", {}).get("portfolio_utilization", 0)
                    },
                    "market": {
                        "context": context.get("market_context", {}),
                        "hours_active": not self.check_market_hours or (9 <= datetime.now(IST).hour <= 16)
                    },
                    "account": {
                        "cash_available": context.get("portfolio_status", {}).get("cash_available", 0)
                    }
                },
                "constraints": {
                    "strategy": self.strategy,
                    "max_positions_can_open": self.risk_limits.max_open_positions
                },
                "opportunities": {
                    "watchlist": self.watchlist
                }
            }
        except Exception as e:
            self.logger.error(f"Failed to prepare decision context: {str(e)}")
            raise


    async def _prepare_monitoring_context(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Prepare context for portfolio monitoring"""
        try:
            base_context = context or {}
            
            monitoring_context = {
                "portfolio": {
                    "positions": await memory_tools.get_positions(self.name),
                    "risk_level": (await self._calculate_portfolio_value_risk())["total_risk"],
                    "total_value": (await self._calculate_portfolio_value_risk())["total_portfolio_value"],
                    "cash_available": base_context.get("portfolio_status", {}).get("cash_available", 0),
                    "utilization": base_context.get("portfolio_status", {}).get("portfolio_utilization", 0)
                },
                "market": {
                    "conditions": base_context.get("market_context", {}),
                    "hours_active": not self.check_market_hours or (9 <= datetime.now(IST).hour <= 16),
                    "volatility_level": base_context.get("market_context", {}).get("volatility", "MEDIUM")
                },
                "risk_metrics": {
                    "max_portfolio_risk": self.risk_limits.max_portfolio_risk,
                    "max_position_size": self.risk_limits.max_position_size,                    
                },
                "strategy": {
                    "name": self.strategy,
                    "watchlist": self.watchlist,
                    "max_positions": self.risk_limits.max_open_positions
                },
                "monitoring_timestamp": datetime.now(IST).isoformat()
            }

            # Add position-specific metrics
            position_metrics = []
            for position in monitoring_context["portfolio"]["positions"]:
                metrics = {
                    "symbol": position.symbol,
                    "quantity": position.quantity,
                    "entry_price": position.entry_price,
                    "current_price": position.current_price,
                    "pnl": position.calculate_pnl() if hasattr(position, 'calculate_pnl') else None,
                    "stop_loss": position.stop_loss,
                    "target": position.target,
                    "hold_duration": (datetime.now(IST) - position.entry_date).days if hasattr(position, 'entry_date') else 0
                }
                position_metrics.append(metrics)
            
            monitoring_context["portfolio"]["position_metrics"] = position_metrics
            
            return monitoring_context

        except Exception as e:
            self.logger.error(f"Failed to prepare monitoring context: {str(e)}")
            raise


    def _process_monitoring_result(self, monitoring_result: Dict[str, Any]) -> Dict[str, Any]:
        """Process and validate monitoring results"""
        try:
            if not monitoring_result:
                raise ValueError("Empty monitoring result received")

            # Extract key metrics
            processed_result = {
                "status": "SUCCESS",
                "timestamp": datetime.now(IST).isoformat(),
                "portfolio_health": {
                    "risk_level": monitoring_result.get("risk_level", "UNKNOWN"),
                    "alerts": monitoring_result.get("alerts", []),
                    "recommendations": monitoring_result.get("recommendations", [])
                },
                "market_conditions": monitoring_result.get("market_conditions", {}),
                "position_updates": []
            }

            # Process position-specific updates
            positions = monitoring_result.get("positions", {})
            for symbol, data in positions.items():
                position_update = {
                    "symbol": symbol,
                    "status": data.get("status", "UNKNOWN"),
                    "risk_score": data.get("risk_score", 0),
                    "action_needed": data.get("action_needed", False),
                    "suggested_actions": data.get("suggested_actions", [])
                }
                processed_result["position_updates"].append(position_update)

            # Add overall recommendations if any
            if monitoring_result.get("immediate_actions_required"):
                processed_result["urgent_actions"] = monitoring_result["immediate_actions_required"]

            return processed_result

        except Exception as e:
            self.logger.error(f"Failed to process monitoring result: {str(e)}")
            return {
                "status": "ERROR",
                "error": str(e),
                "timestamp": datetime.now(IST).isoformat()
            }

   
    async def _calculate_portfolio_value_risk(self) -> dict:
        """Calculate current portfolio risk percentage"""
        try:
            active_positions = await memory_tools.get_positions(self.name)
            
            if not active_positions:
                return {"total_risk": 0.0, "total_portfolio_value": 0.0}

            # Calculate total portfolio value
            account = Account.get(self.name)
            total_portfolio_value = account.calculate_portfolio_value()

            # Calculate VaR (Value at Risk) approximation
            total_risk = 0.0
            
            for position in active_positions.values():                
                position_value = position['entry_price'] * position['quantity']
                position_weight = position_value / total_portfolio_value
                position_risk = position_weight * 0.20
                total_risk += position_risk

            return {"total_risk": round(total_risk, 2), "total_portfolio_value": round(total_portfolio_value, 2)}

        except Exception as e:
            self.logger.error(f"Portfolio risk calculation failed: {str(e)}")
            return {"total_risk": 0.0, "total_portfolio_value": 0.0}


    async def _perform_pre_workflow_checks(self) -> bool:
        """Perform pre-workflow validation checks"""
        try:            
            # Check portfolio value and risk
            portfolio_risk = (await self._calculate_portfolio_value_risk())["total_risk"]
            if portfolio_risk > self.risk_limits.max_portfolio_risk:
                return False

            # Check market hours (basic check)
            if self.check_market_hours:
                current_time = datetime.now(IST)
                if current_time.hour < 9 or current_time.hour > 16:
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Pre-workflow check failed: {str(e)}")
            return False


    async def _should_seek_new_positions(self, portfolio_status: Dict[str, Any]) -> bool:
        """Determine if new positions should be sought"""
        try:
            # Check if we have capacity for new positions
            if portfolio_status.get("portfolio_utilization", 0) > 80:
                self.logger.info(f"\033[93m[VALIDATION] Portfolio utilization too high, skipping new positions.\033[0m")
                return False
            
            # Check if we have enough cash
            if portfolio_status.get("cash_available", 0) < 50000:  # Minimum cash requirement
                self.logger.info(f"\033[93m[VALIDATION] Not enough cash available, skipping new positions.\033[0m") 
                return False
            
            # Check if we already have max positions
            if portfolio_status.get("total_positions", 0) >= self.risk_limits.max_open_positions:  # Max positions limit
                self.logger.info(f"\033[93m[VALIDATION] Already at max positions, skipping new positions.\033[0m")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Position need assessment failed: {str(e)}")
            return False
        

    def _serialize_context(self, decision_context: Dict[str, Any]) -> str:
        """Serialize context dict to string, handling non-JSON-serializable objects."""
        
        def serialize_position(pos):
            """Convert Position dictionary or object to consistent format."""
            if isinstance(pos, dict):
                return {
                    "entry_price": pos.get("entry_price"),
                    "quantity": pos.get("quantity"),
                    "stop_loss": pos.get("stop_loss"),
                    "target": pos.get("target"),
                    "entry_date": pos.get("entry_date")
                }
            else:                
                return {
                    "entry_price": getattr(pos, "entry_price", None),
                    "quantity": getattr(pos, "quantity", None),
                    "stop_loss": getattr(pos, "stop_loss", None),
                    "target": getattr(pos, "target", None),
                    "entry_date": getattr(pos, "entry_date", None)
                }


        # Create a serializable copy of the context
        serializable_context = {
            "current_state": {
                "portfolio": {
                    "status": decision_context["current_state"]["portfolio"]["status"],
                    "risk_level": decision_context["current_state"]["portfolio"]["risk_level"],
                    "positions": {
                        symbol: serialize_position(pos) 
                        for symbol, pos in decision_context["current_state"]["portfolio"]["positions"].items()
                    },
                    "utilization": decision_context["current_state"]["portfolio"]["utilization"]
                },
                "market": {
                    "context": decision_context["current_state"]["market"]["context"],
                    "hours_active": decision_context["current_state"]["market"]["hours_active"]
                },
                "account": {
                    "cash_available": decision_context["current_state"]["account"]["cash_available"]
                }
            },
            "constraints": {
                "strategy": decision_context["constraints"]["strategy"],
                "max_positions": decision_context["constraints"]["max_positions_can_open"]
            },
            "opportunities": {
                "watchlist": list(decision_context["opportunities"]["watchlist"])  # Convert to list if it's a set
            }
        }
        
        return json.dumps(serializable_context, indent=2)