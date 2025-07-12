from typing import Dict, List
from market_tools.market import get_multiple_symbol_prices
from trade_agents.traders import Trader
from memory.agent_memory import AgentMemory
from memory.memory_tools import get_overall_market_context
import logging
from datetime import datetime, time
from utils.redis_client import main_redis_client as r
import asyncio

agent_names = ["warren", "george", "ray", "cathie"]
# agent_names = ["warren"]
all_active_positions = {}
MAX_TURNS = 30


class AgentOrchestrator:
    def __init__(self, agents: List[Trader], stop_loss_manager=None, position_managers=None):
        """
        Initialize the orchestrator with agents and supporting systems.
        
        Args:
            agents: List of trading agents (Warren, Ray, George, Cathie)
            redis_client: Redis client for caching and real-time data
        """
        
        self.agents = {agent.name: agent for agent in agents}
        self.redis_client = r
        self.logger = logging.getLogger(__name__)
            
        self.position_manager = {name: pm for name, pm in position_managers.items() if pm is not None}
        self.stop_loss_manager = stop_loss_manager

        self.risk_limits = {name: pm.risk_limits for name, pm in position_managers.items() if pm is not None}


    async def run_traders(self, market_mcp_servers, researcher_mcp_servers):
        for trader in self.agents.values():
            await trader.run(market_mcp_servers, researcher_mcp_servers)
            await asyncio.sleep(2)

    

    def run_daily_context_analysis(self) -> dict:
        """
        Collects each agent's daily context and the current market context, merges them, and performs a simple combined analysis.
        Returns a dict with agent-wise merged context.
        """

        # Get market context (shared for all agents)
        market_context = get_overall_market_context()
        if not market_context:
            market_context = {"status": "no market context found", "date": datetime.now().strftime("%Y-%m-%d")}

        merged = {}
        for agent in self.agents:
            agent_name = getattr(agent, 'name', str(agent))
            try:
                agent_mem = AgentMemory(agent_name)
                daily_context = agent_mem.get_daily_context()
            except Exception as e:
                daily_context = {"error": str(e)}
            

            merged[agent_name] = market_context + daily_context

            ## TODO: Implement market + daily context analysis OR create a new agent for this purpose

        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "merged_contexts": merged
        }



    def manage_position_monitoring(self) -> List[Dict]:
        """
        Monitors all active positions across all agents and provides actionable insights.
        
        This function checks each position for risk levels, profit/loss status,
        stop loss triggers, and generates recommendations for position management.
        
        Returns:
            List of position monitoring results with recommended actions
        """
        results = []
        for agent_name in self.agents.items():
            try:
                monitoring = self._monitor_positions(agent_name)
                time.sleep(1)
            except Exception as e:
                monitoring = {"error": str(e)}
            results.append({
                "agent": agent_name,
                "monitoring": monitoring
            })
        return results


    
    def _monitor_positions(self, agent_name: str) -> List[dict]:
        """
        Analyze and monitor all active positions for this agent.
        Returns a list of monitoring results for each position.
        """        

        results = []
        pm = self.position_manager.get(agent_name)
        positions = pm.positions if pm else {}        
        if not positions:
            return [{
                "message": f"No active positions found for {agent_name}",
                "timestamp": datetime.now().isoformat()
            }]
        
        current_price_all_positions = get_multiple_symbol_prices(list(positions.keys()))
        
        for symbol, pos in positions.items():
            current_price = current_price_all_positions.get(symbol, 0)
            entry_price = pos.get('entry_price', 0)
            quantity = pos.get('quantity', 0)
            pnl_amount = (current_price - entry_price) * quantity
            pnl_percent = ((current_price - entry_price) / entry_price) * 100 if entry_price else 0
            days_held = (datetime.now() - datetime.fromisoformat(pos.get('entry_date'))).days if pos.get('entry_date') else 0
                        
            # --- Risk logic ---
            risk_level = "Low"
            action_required = None
            reason = None

            # Example risk thresholds (customize as needed)
            if pnl_percent < -8:
                risk_level = "High"
                action_required = "Review/Exit"
                reason = "Loss exceeds -8%"
            elif pnl_percent < -4:
                risk_level = "Medium"
                action_required = "Tighten stop/review"
                reason = "Loss exceeds -4%"
            elif pnl_percent > 15:
                risk_level = "Low"
                action_required = "Consider booking profit"
                reason = "Profit exceeds 15%"
            elif pnl_percent > 8:
                risk_level = "Low"
                action_required = "Trail stop/monitor"
                reason = "Profit exceeds 8%"

            # Stop loss breach
            stop_loss_price = pos.get('stop_loss')
            if stop_loss_price and current_price <= stop_loss_price:
                risk_level = "High"
                action_required = "Immediate exit"
                reason = "Stop loss breached"
            
            results.append({
                "symbol": symbol,
                "current_price": current_price,
                "entry_price": entry_price,
                "pnl_percent": pnl_percent,
                "pnl_amount": pnl_amount,
                "risk_level": risk_level,
                "stop_loss_price": pos.get('stop_loss'),
                "target_price": pos.get('target'),
                "days_held": days_held,
                "action_required": action_required,
                "reason": reason
            })
        return results