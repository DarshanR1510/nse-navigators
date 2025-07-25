from dotenv import load_dotenv
from all_agents.base_trade_agent import AgentConfig
from all_agents.manager_agent import ManagerAgent
from all_agents.researcher_agent import ResearcherAgent
from all_agents.fundamental_agent import FundamentalAgent
from all_agents.technical_agent import TechnicalAgent
from all_agents.decision_agent import DecisionAgent
from all_agents.execution_agent import ExecutionAgent
from risk_management.position_manager import PositionManager
from data.accounts import Account
from data.schemas import SelectedSymbols, TradeCandidate, SelectedSymbol, IST
from memory import memory_tools
from market_tools.historical_data_fetcher import fetch_bulk_historical_data
from datetime import datetime
from memory.agent_memory import AgentMemory
from typing import Dict, Any
from mcp_servers.mcp_servers_manager import (
    setup_decision_mcp_servers,
    setup_fundamental_mcp_servers,
    setup_technical_mcp_servers,
    setup_execution_mcp_servers
)
from data.database import DatabaseQueries
from market_tools.market import resolve_symbol_impl
from market_tools.market import get_multiple_symbol_prices
from dataclasses import asdict
from trade_agents.performance_metrics import PerformanceTracker
from trade_agents.session_manager import SessionManager
from contextlib import asynccontextmanager
import os
import logging
import json
import traceback


load_dotenv(override=True)
logger = logging.getLogger(__name__)
trader_names = ["warren", "ray", "george", "cathie"]
    
run_agents = 2          #* Change this number to run expected number of agents
trader_names = trader_names[:run_agents]


class TradingOrchestrator:
    """Orchestrates the agentic trading workflow by coordinating specialized agents."""

    def __init__(
        self,
        trader_name: str,
        model_name: str,
        strategy: str,
        position_manager: PositionManager
    ):
        # Core attributes
        self.trader_name = trader_name
        self.model_name = model_name
        self.strategy = strategy
        self.position_manager = position_manager
        
        # Configuration
        self.portfolio_value = float(os.getenv("INITIAL_BALANCE", 500000.0))
        self.max_daily_trades = int(os.getenv("MAX_DAILY_TRADES", "5"))
        self.monitor_interval = int(os.getenv("MONITOR_INTERVAL_MINUTES", "30")) * 60
        
        # State tracking
        self.daily_trades_count = 0
        self.workflow_runs = 0
        self.last_monitor_run = None
        self.last_portfolio_check = None
        
        # Services
        self.logger = logging.getLogger(f"{__name__}.{trader_name}")
        self.agent_mem = AgentMemory(trader_name)
        self.account = Account.get(trader_name)
        self.performance_tracker = PerformanceTracker(trader_name)
        self.session = None
        # self.mcp_manager = MCPServerManager.get_instance()


    async def _initialize_agents(self) -> Dict[str, Any]:
        """Initialize all trading agents with proper configuration"""        

        return {            
            "manager": await ManagerAgent.create(
            config=self.base_config,
            position_manager=self.position_manager
            ),

            "researcher": ResearcherAgent(
                config=self.base_config
            ),
            
            "fundamental": FundamentalAgent(
                config=self.base_config,
                mcp_setup=await setup_fundamental_mcp_servers()
            ),
            
            "technical": TechnicalAgent(
                config=self.base_config,
                mcp_setup=await setup_technical_mcp_servers()
            ),                
            
            "decision": DecisionAgent(
                config=self.base_config,
                mcp_setup=await setup_decision_mcp_servers()
            ),
            
            "execution": ExecutionAgent(
                config=self.base_config,
                mcp_setup=await setup_execution_mcp_servers()
            )
        }
       
    
    async def initialize(self):
        """Initialize orchestrator with session and agents"""
        
        async with SessionManager.session_context(self.trader_name):
            self.trace_id, self.session = await SessionManager.get_session_and_trace(self.trader_name)
            
            # Create base config
            self.base_config = AgentConfig(
                name=self.trader_name,
                model_name=self.model_name,
                strategy=self.strategy,
                session=self.session,
                trace_id=self.trace_id
            )
            
            # Initialize agents
            self.agents = await self._initialize_agents()

    
    @asynccontextmanager
    async def workflow_context(self):
        """Context manager for workflow execution"""        
        try:
            await self.initialize() 
            yield self
        except Exception as e:
            self.logger.error(f"Workflow error: {str(e)}")
            raise


    async def run(self) -> Dict[str, Any]:
        """Main orchestrator workflow."""
        async with self.workflow_context():
            try:
                self.logger.info(f"🚀 Starting {self.trader_name} workflow...")
                DatabaseQueries.write_log(self.trader_name, "function", f"🚀 Starting workflow...")

                context = await self._gather_context()                
                manager_decision = await self.agents["manager"].decide_next_action(context)
                            
                # For testing purposes, we set a default decision
                # manager_decision = "RESEARCH"  
                
                workflow_results = await self._execute_workflow(manager_decision, context)
                                                
                performance_results = await self._track_performance()
                workflow_results["performance"] = performance_results

                #Final result
                logging.info(f"Workflow results: {workflow_results}")
                DatabaseQueries.write_log(self.trader_name, "response", f"Workflow completed, Monitoring positions and market...")
                return workflow_results

            except Exception as e:
                self.logger.error(f"Workflow failed: {str(e)}")
                return {"status": "ERROR", "error": str(e)}



    async def _execute_workflow(self, decision: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow based on manager's decision"""
        try:
            match decision:
                case "RESEARCH":
                    return await self._execute_research_workflow(context)
                case "MONITOR":
                    return await self._execute_monitor_workflow()
                case "REBALANCE":
                    return await self._execute_rebalance_workflow()
                case _:
                    self.logger.warning(f"Unknown decision: {decision}")
                    return await self._execute_monitor_workflow()
                    
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {str(e)}")
            return await self._handle_failure("WORKFLOW", e)


    async def _execute_research_workflow(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the research and trading workflow"""


        #! PHASE 1: Research
        DatabaseQueries.write_log(self.trader_name, "agent", f"🚀 Launching Researcher Agent...")
        research_results = await self._research_agent_run(context)
        
        if not research_results:
            return await self._handle_failure("RESEARCH", Exception("No symbols found"))
        
        research_results_with_resolved_symbols = await self._resolve_symbols_for_candidates(research_results)

        #* Researcher result test data
        # research_results_with_resolved_symbols = researcher_result_data()  


        #! PHASE 2: Analysis
        DatabaseQueries.write_log(self.trader_name, "agent", f"🕵🏻‍♂️ Fundamental & Technical Analysis Agents ...")
        analysis_results = await self._run_analysis_pipeline(research_results_with_resolved_symbols)        
        
        if not analysis_results:
            return await self._handle_failure("ANALYSIS", Exception("No analysis results"))                

        #* Analysis results test data
        # analysis_results = analysis_result_data()


        #! PHASE 3: Decision
        DatabaseQueries.write_log(self.trader_name, "agent", f"🙇🏻‍♂️ Decision agent is in cooking mode...")
        decision_results = await self._decision_agent_run(analysis_results)
        
        if not decision_results:
            return await self._handle_failure("DECISION", Exception("No decision results"))
        
        logging.info(f"[DECISION] output: {decision_results}")

        #* Decision results test data
        # decision_results = decision_result_data()

        #! PHASE 4: Execution
        if hasattr(decision_results, 'decision') and decision_results.decision == "TRADE":
            
            trade_candidate = await self._calculate_position_size(decision_results.trade_candidate)
            if not trade_candidate:
                return await self._handle_failure("POSITION_SIZING", Exception("Failed to calculate position size"))
                            
            decision_results.trade_candidate = trade_candidate            
            
            DatabaseQueries.write_log(self.trader_name, "agent", f"🕹️ Execution agent is in action ...")
            execution_results = await self._execute_trades(trade_candidate)

            if not execution_results.get("trades_executed"):
                return await self._handle_failure("EXECUTION", Exception("Trade execution failed"))
                        
            if decision_results.watchlist == {}:
                DatabaseQueries.write_log(
                    self.trader_name, "trace", 
                    f"Watchlist is empty, no symbols added"
                )
            else:
                DatabaseQueries.write_log(
                    self.trader_name, "trace",
                    f"Watchlist updated with 1 symbol"
                )

            return {
                "status": "SUCCESS",
                "research": len(research_results.selections),
                "analysis": len(analysis_results),
                "trades": execution_results,
                "watchlist": decision_results.watchlist
            }
   
        return {
            "status": "NO_TRADE",
            "watchlist": decision_results.watchlist if hasattr(decision_results, 'watchlist') else None
        }


    async def _execute_monitor_workflow(self) -> Dict[str, Any]:
        """Execute the monitoring workflow"""
        monitoring_results = await self._monitor_positions()
        return {
            "status": "MONITORING_COMPLETE",
            "results": monitoring_results
        }


    # TODO Implement portfolio rebalancing logic
    async def _execute_rebalance_workflow(self) -> Dict[str, Any]:
        """Execute the portfolio rebalancing workflow"""
        # Implementation remains the same
        return {"status": "REBALANCE_NOT_IMPLEMENTED"}


    async def _gather_context(self) -> Dict[str, Any]:
        """
        Gather current market and portfolio context for ManagerAgent.
        """
        portfolio_status = await self._analyze_portfolio_status()
        today_context = memory_tools.get_overall_market_context()

        return {
            "portfolio_status": portfolio_status,
            "overall_market_context": today_context
        }


    async def _analyze_portfolio_status(self) -> Dict[str, Any]:
        """Analyze current portfolio status"""
        try:
            # Get active positions            
            active_positions = await memory_tools.get_positions(self.trader_name)
            
            # Calculate portfolio metrics
            total_value: float = self.account.calculate_portfolio_value()
            total_pnl: float = self.account.calculate_profit_loss(total_value)
            cash_available: float = self.account.balance

            symbols = list(active_positions.keys())
            current_market_prices = get_multiple_symbol_prices(symbols)

            portfolio_status = {
                "total_positions": len(active_positions),
                "total_value": total_value,
                "total_pnl": total_pnl,
                "cash_available": cash_available,
                "portfolio_utilization": ((total_value - cash_available) / self.portfolio_value) * 100,
                "positions_details": [
                    {
                        "symbol": symbol,
                        "quantity": data["quantity"],
                        "entry_price": data["entry_price"],
                        "current_value": data["quantity"] * current_market_prices.get(symbol, 0),
                        "unrealized_pnl": round((current_market_prices.get(symbol, 0) - data["entry_price"]) * data["quantity"], 2),
                        "pnl_percentage": round(((current_market_prices.get(symbol, 0) - data["entry_price"]) / data["entry_price"]) * 100, 2)
                    }
                    for symbol, data in active_positions.items()
                ]
            }
            
            self.last_portfolio_check = datetime.now()       ##TODO: Implement this like, check only after some times     
            return portfolio_status
            
        except Exception as e:
            self.logger.error(f"Portfolio analysis failed: {str(e)}")
            return {"error": str(e)}


    async def _research_agent_run(self, today_context) -> SelectedSymbols:
        """Discover new trading opportunities using researcher agent"""
        
        try:
            # Run researcher agent
            researcher_agent = (self.agents["researcher"])         
            research_result = await researcher_agent.run(today_context)

            if research_result:                
                self.logger.info(f"{self.trader_name} has researched and selected {len(research_result.selections)} companies.")
                DatabaseQueries.write_log(self.trader_name, "agent", f"Researcher found {len(research_result.selections)} companies")
                return research_result
            
            else:
                self.logger.warning("Researcher returned no results")
                return None
            
        except Exception as e:
            traceback.print_exc()
            self.logger.error(f"Stock Research failed: {str(e)}")
            return []


    async def _run_analysis_pipeline(self, candidates: SelectedSymbols) -> Dict[str, Dict[str, Any]]:
        """Run comprehensive analysis pipeline on candidates"""
        try:            
            print(f"Running analysis pipeline for {len(candidates.selections)} candidates")            
            
            if not candidates:
                self.logger.warning("Invalid candidates format received")
                return []            

            fundamental_agent = (self.agents["fundamental"])
            technical_agent = (self.agents["technical"])

            fundamental_results = await fundamental_agent.run(candidates)
            technical_results = await technical_agent.run(candidates)

            if fundamental_results and technical_results:
                self.logger.info("Both fundamental and technical analysis completed successfully.")
            
            elif fundamental_results:
                self.logger.info("Fundamental analysis completed successfully.")

            elif technical_results:
                self.logger.info("Technical analysis completed successfully.")

            else:
                self.logger.warning("Analysis tools returned empty results")
                return []
            

            fund_dict = {a.symbol: a for a in fundamental_results.analyses}
            tech_dict = {a.symbol: a for a in technical_results.analyses}

            combined_results = {}

            for symbol in fund_dict:
                combined_results[symbol] = {
                    "fundamental_report": fund_dict[symbol].analysis,
                    "technical_report": tech_dict.get(symbol).analysis if symbol in tech_dict else None,
                    "fundamental_conviction": fund_dict[symbol].conviction_score,
                    "technical_conviction": tech_dict[symbol].conviction_score if symbol in tech_dict else None,
                }
            
            print(f"[ANALYSIS] Combined results: {json.dumps(combined_results, indent=2)}")
            self.logger.info(f"📊 [{self.trader_name}] Completed analysis for {len(combined_results)} stocks candidates")
            DatabaseQueries.write_log(self.trader_name, "generation", f"Completed analysis for {len(combined_results)} stocks candidates")
            
            return combined_results
            
        except Exception as e:
            self.logger.error(f"Analysis pipeline failed: {str(e)}")
            return []
    

    async def _decision_agent_run(self, analysis_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Run decision agent to select candidates for trading and watchlist"""
        try:
            if not analysis_results:
                self.logger.warning("No analysis results provided for decision agent")
                return {"trade_candidates": [], "watchlist": []}
            
            # Run decision agent
            decision_agent = (self.agents["decision"])
            decision_result = await decision_agent.run(analysis_results)
            
            if not decision_result:
                self.logger.warning("Decision agent returned no results")
                return {"trade_candidate": None, "watchlist": None}

            self.logger.info(
                f"🎯 [{self.trader_name}] Decision agent selected trade candidate: {getattr(decision_result.trade_candidate, 'symbol', None)} "
                f"and watchlist candidate: {getattr(decision_result.watchlist, 'symbol', None)}"
            )
            DatabaseQueries.write_log(
                self.trader_name, "agent",
                f"Decision agent selected trade candidate: {getattr(decision_result.trade_candidate, 'symbol', None)} "
                f"and watchlist candidate: {getattr(decision_result.watchlist, 'symbol', None)}"
            )
            return decision_result
            
        except Exception as e:
            self.logger.error(f"Decision agent run failed: {str(e)}")
            return {"trade_candidate": None, "watchlist": None}


    async def _execute_trades(self, trade_candidate: TradeCandidate) -> Dict[str, Any]:
        """Execute trades for validated candidates"""

        logging.info(f"Executing trades for {trade_candidate.symbol}")

        try:
            execution_results = {
                "trades_executed": [],
                "trades_failed": [],                
                "total_amount_invested": 0.0
            }     

            # Run execution agent
            execution_agent = (self.agents["execution"])
            execution_result = await execution_agent.run(trade_candidate)

            try:
                if getattr(execution_result, "execution_status", None) == "SUCCESS":
                    execution_results["trades_executed"].append({
                        "symbol": trade_candidate.symbol,
                        "quantity": trade_candidate.quantity,
                        "entry_price": trade_candidate.entry_price,
                    "stop_loss": trade_candidate.stop_loss,
                    "target_price": trade_candidate.target_price,
                    "reason": trade_candidate.reason
                })
                execution_results["total_amount_invested"] += (
                    trade_candidate.entry_price * trade_candidate.quantity
                )
                self.daily_trades_count += 1
            
                self.logger.info(f"Executed {len(execution_results['trades_executed'])} trades successfully.")
                
                return execution_results

            except Exception as e:
                self.logger.error(f"Trade execution failed for {trade_candidate.symbol}: {str(e)}")
                execution_results["trades_failed"].append({
                    "symbol": trade_candidate.symbol,
                    "error": str(e)
                })
                                
            
            
        except Exception as e:
            self.logger.error(f"Trade execution process failed: {str(e)}")
            return {"error": str(e)}
    
    
#TODO Improve this failure handling logic
    async def _handle_failure(self, phase: str, error: Exception, retries: int = 0, max_retries: int = 1) -> Dict[str, Any]:
        
        self.logger.error(f"❌ [{self.trader_name}] {phase} failed: {str(error)}")
        DatabaseQueries.write_log(self.trader_name, "error", f"{phase} failed: {str(error)}")
        
        if retries < max_retries:
            self.logger.info(f"🔄 Retrying {phase} (attempt {retries+1})...")
            return {"status": f"{phase}_RETRY", "retries": retries+1}
        
        else:
            self.logger.info(f"🔍 Navigating to monitoring after {phase} failure.")
            monitoring_results = await self._monitor_positions()
            return {
                "status": f"{phase}_FAILURE",
                "monitoring_results": monitoring_results
            }
    

    async def _monitor_positions(self) -> Dict[str, Any]:
        """
        Monitors all active positions for a single trader, executes actions if needed, and returns analytics.
        Only runs if min_interval_minutes have passed since last run.
        """
        
        now = datetime.now(IST)
        if self.last_monitor_run and (now - self.last_monitor_run).total_seconds() < 1800:  # 30 minutes
            DatabaseQueries.write_log(self.trader_name, "trace", f"Skipping monitoring for {self.trader_name} as last run was less than 30 minutes ago.")
            self.logger.info(f"Skipping monitoring for {self.trader_name}: last run was less than 30 minutes ago.")
            
            return {"status": "SKIPPED", "reason": "Last monitor run was less than 30 minutes ago."}
        
        self.last_monitor_run = now

        return self.position_manager.monitor_positions()


    async def _track_performance(self) -> Dict[str, Any]:
        """Track and update performance metrics"""
        try:
            # Get current portfolio value
            active_positions = self.position_manager.get_active_positions()
            total_value: float = self.account.calculate_portfolio_value()
            total_pnl: float = self.account.calculate_profit_loss(total_value)

            # Update performance tracker
            performance_data = {
                "timestamp": datetime.now(IST),
                "portfolio_value": round(total_value, 2),
                "total_pnl": round(total_pnl, 2),
                "active_positions": len(active_positions),
                "daily_trades": self.daily_trades_count
            }            
            
            # Get performance metrics
            metrics = self.performance_tracker.get_cached_metrics()
            
            return {
                "current_portfolio_value": round(total_value, 2),
                "total_pnl": performance_data["total_pnl"],
                "performance_metrics": asdict(metrics) if metrics else {},
                "daily_trades_count": self.daily_trades_count
            }
            
        except Exception as e:
            self.logger.error(f"Performance tracking failed: {str(e)}")
            return {"error": str(e)}
    

    async def _resolve_symbols_for_candidates(self, candidates: SelectedSymbols) -> SelectedSymbols:
        """
        Resolve symbols for the given candidates using resolve_symbol tool.
        """
        resolved_candidates = []
        all_symbols = []            #* Store all resolved symbols for storing historical data in redis
        
        for candidate in candidates.selections:
            try:
                resolved_symbol = resolve_symbol_impl(candidate.company_name)

                if resolved_symbol:
                    candidate.symbol = resolved_symbol
                    resolved_candidates.append(candidate)
                    all_symbols.append(resolved_symbol)
                
                else:
                    self.logger.warning(f"Could not resolve symbol for {candidate.company_name}")
            except Exception as e:
                self.logger.error(f"Error resolving symbol for {candidate.company_name}: {str(e)}")

        #* Fetch historical data for all resolved symbols
        if len(all_symbols) > 0:
            self.logger.info(f"Fetching historical data for {len(all_symbols)} symbols: {all_symbols}")
            fetch_bulk_historical_data(all_symbols)
        
        return SelectedSymbols(selections=resolved_candidates)    
    

    async def _calculate_position_size(self, trade_candidate: TradeCandidate) -> TradeCandidate:
        """
        Calculate position size based on entry price, stop loss, portfolio value, and trader name.
        """        
        try:
            if not trade_candidate:
                return None
            
            quantity = self.position_manager.calculate_position_size(
                trade_candidate.entry_price, 
                trade_candidate.stop_loss, 
                self.portfolio_value, 
                self.trader_name)

            return TradeCandidate(
                symbol=trade_candidate.symbol,
                entry_price=trade_candidate.entry_price,
                quantity=quantity,
                stop_loss=trade_candidate.stop_loss,
                target_price=trade_candidate.target_price,
                reason=trade_candidate.reason
            )
        
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return trade_candidate


#TODO Implement portfolio rebalancing logic
    async def _rebalance_portfolio(self):
        """
        Rebalance portfolio logic placeholder.
        """
        self.logger.info("🔄 Rebalancing portfolio...")
        # Implement rebalance logic




############################ SAMPLE TEST DATA ############################

#* SAMPLE DATA AFTER PHASE 1 - RESEARCH
def researcher_result_data():
    return SelectedSymbols(
        selections=[
            SelectedSymbol(
                company_name='Reliance Industries Limited',
                symbol='RELIANCE',
                reason="Strong earnings growth driven by diversified business segments including digital, retail, and energy; competitive edge in India's largest private sector company; reasonable valuation relative to growth prospects.",
                conviction_score=9.0,
                time_horizon='long-term',
                risk_factors='Volatility in global oil prices and regulatory changes in telecom sector.'
            ),
            SelectedSymbol(
                company_name='HDFC Bank Limited',
                symbol='HDFCBANK',
                reason='Consistent earnings growth with strong retail and corporate loan portfolio; superior asset quality and strong management; trading at reasonable valuations compared to its growth history.',
                conviction_score=8.0,
                time_horizon='long-term',
                risk_factors='Slower credit growth or asset quality deterioration in a stressed economic environment.'
            ),
            SelectedSymbol(
                company_name='Tata Consultancy Services Limited',
                symbol='TCS', 
                reason='Leading IT services company with strong digital transformation capabilities; benefiting from global tech outsourcing trends; stable cash flows and strong management team; attractive valuation for a growth IT stock.', 
                conviction_score=8.0, 
                time_horizon='long-term', 
                risk_factors='Global tech spending slowdowns could impact revenue growth.'
            ), 
            SelectedSymbol(
                company_name='Infosys Limited', 
                symbol='INFY', 
                reason='Strong quarterly results with robust deal wins and digital revenue growth; leading position in IT services with strong management and consistent cash flows; undervalued compared to peers.', 
                conviction_score=7.0, 
                time_horizon='long-term', 
                risk_factors='Competition in IT sector and possible client budget cuts.'
            ), 
            SelectedSymbol(
                company_name='Bajaj Finance Limited', 
                symbol='BAJFINANCE', 
                reason='High growth NBFC with leadership in consumer finance; strong loan growth and improving asset quality; expanding presence into new segments; reasonable valuation considering growth potential.', 
                conviction_score=8.0, 
                time_horizon='medium-term', 
                risk_factors='Rising interest rates and regulatory changes impacting NBFCs.'
            )
        ]
    )


#* SAMPLE DATA AFTER PHASE 2 - ANALYSIS
def analysis_result_data():
    return {
            "DLF": {
                "fundamental_report": "DLF Ltd shows a high P/E ratio of 44.9, indicating premium valuation. The company has strong sales growth (24.38% YoY) and profit growth (60.32% YoY), supported by a robust ROE of 11.4%. However, ROCE is low at 6.51%, and debt levels are moderate (borrowings at \u20b94,103 cr). Promoter holding is high at 74.08%, with stable FII and DII participation. The balance sheet is healthy, but cash flow from operations is volatile. Given the high valuation and mixed efficiency metrics, caution is advised despite growth potential.",
                "technical_report": "DLF shows a bullish trend with and EMA slopes indicating upward momentum. The MACD is positive, and RSI is neutral (around 60), suggesting room for further upside. Bollinger Bands show price near the upper band, indicating potential overbought conditions. Volume patterns are inconclusive due to insufficient data. Key support is at 838, and resistance at 880. Relative strength is strong compared to peers. Entry: 850, Stop-loss: 838 (1.4%), Target: 880 (3.5%).",
                "fundamental_conviction": 8.0,
                "technical_conviction": 9.0
            },
            "NELCAST": {
                "fundamental_report": "Nelcast Ltd has a P/E ratio of 41.9, above the peer median, reflecting growth expectations. Sales growth is modest (10.54% 3-year CAGR), but profit growth is strong (38.26% 3-year CAGR). ROE is low at 6.48%, and ROCE is 9.55%. Debt levels are manageable (borrowings at \u20b9294 cr), and promoter holding is high at 74.87%. Cash flow from operations is positive but inconsistent. The stock's valuation is stretched relative to peers, and recent quarterly profit declined (-31.48% YoY).",
                "technical_report": "NELCAST lacks sufficient data for a comprehensive technical analysis. Limited indicators suggest a breakout potential, but confirmation is needed. Entry: 152.7, Stop-loss: 147 (3.7%), Target: 163 (6.7%).",
                "fundamental_conviction": 7.0,
                "technical_conviction": 6.0
            },
            "HINDALCO": {
                "fundamental_report": "Hindalco Industries Ltd has a P/E ratio of 18.5, below the peer median, indicating potential undervaluation. The company has shown strong sales growth (15.12% YoY) and profit growth (22.34% YoY), supported by a healthy ROE of 12.5%. ROCE is also robust at 10.8%, and debt levels are manageable (borrowings at \u20b94,500 cr). Promoter holding is stable at 51.0%, with good FII participation. The balance sheet is strong, and cash flow from operations is consistent. Overall, the fundamentals are solid, and the stock appears attractively valued.",
                "technical_report": "HINDALCO exhibits a bullish trend with EMA slopes indicating upward momentum. The MACD is positive, and RSI is around 65, suggesting further upside potential. Bollinger Bands show price near the upper band, indicating potential overbought conditions. Volume patterns are supportive of the trend. Key support is at 450, and resistance at 480. Relative strength is strong compared to peers. Entry: 460, Stop-loss: 450 (2.2%), Target: 480 (4.3%).",
                "fundamental_conviction": 8.5,
                "technical_conviction": 8.0
            }
        }


#* SAMPLE DATA AFTER PHASE 3 - DECISION
def decision_result_data():
    return {
        "trade_candidate": TradeCandidate(
            symbol='HINDALCO',
            entry_price=460.0,
            quantity=10,
            stop_loss=450.0,
            target_price=480.0,
            reason='Strong fundamentals and bullish technical indicators support high conviction.'
        ),
        "watchlist": {
            "symbol": "DLF",
            "reason": "Potential breakout with strong fundamentals."
        },
        "decision": "TRADE"  
    }



