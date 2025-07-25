from dotenv import load_dotenv
from agents.agent import Agent
from trade_agents.models import get_model
from trade_agents.templates import decision_instructions
from data.schemas import DecisionAgentOutput, AgentConfig, TradeCandidate, WatchlistCandidate
from agents import Runner
from all_agents.base_trade_agent import BaseTradeAgent
import json
import traceback    
import logging
from utils.json_response_extractor import extract_json_from_response

load_dotenv(override=True)


class DecisionAgent(BaseTradeAgent):
    def __init__(self, config: AgentConfig, mcp_setup: list):
        super().__init__(config)
        self.name = config.name
        self.model = get_model(config.model_name)
        self.strategy = config.strategy
        self.mcp_servers = mcp_setup
        self.session = config.session
        self.trace_id = config.trace_id
        self.logger = logging.getLogger(__name__)


        # Configure output_type based on model capabilities
        if self.model in ["gpt-4.1-mini", "gpt-4o", "gpt-4o-mini"]:
            self._agent = Agent(
            name=f"{self.name}_decision_agent",
            instructions=decision_instructions(self.name, self.strategy),
            model=self.model,
            mcp_servers=self.mcp_servers,
            output_type=DecisionAgentOutput
        )
        else:                
            self._agent = Agent(
            name=f"{self.name}_decision_agent",
            instructions=decision_instructions(self.name, self.strategy),
            model=self.model,
            mcp_servers=self.mcp_servers            
            )



    async def run(self, fun_tech_analysis: dict) -> DecisionAgentOutput:
        """Run technical analysis on selected symbols"""
        try:
            input_text = (
                "Based on the provided analysis reports, decide which symbols to trade, add to watchlist, or not trade. "
                "Consider the composite scores, entry prices, stop losses, and target prices. "
                "Do not trade if composite score is below 7 or you don't have a clear edge."
                "Return results in DecisionAgentOutput format."
            )

            analysis_data = (
                fun_tech_analysis.dict() 
                if hasattr(fun_tech_analysis, 'dict') 
                else fun_tech_analysis
            )
            
            # Run agent with prepared context
            result = await Runner.run(
                self._agent,
                input=f"{input_text}, \nDecide which are to trade, or add to watchlist or no trade based on these analysis reports: {json.dumps(analysis_data)}",
                session=self.session                
            )            
            
            # When No Trade is returned, we need to return None
            if "NO_TRADE" in result.final_output:
                return DecisionAgentOutput(
                    trade_candidate=None,
                    watchlist=None,
                    decision="NO_TRADE"
                )

            if isinstance(result.final_output, DecisionAgentOutput):
                return result.final_output
            
            elif isinstance(result.final_output, str):
                try:
                    # Use the utility function to extract JSON
                    output_dict = extract_json_from_response(result.final_output)
                    
                    if output_dict:
                        # Transform the dict to match expected schema
                        transformed_dict = {
                            "trade_candidate": output_dict.get("trade_candidate", {}),
                            "watchlist": output_dict.get("watchlist", {}),
                            "decision": output_dict.get("decision", "NO_TRADE")
                        }
                        
                        output = DecisionAgentOutput(**transformed_dict)
                        return output
                        
                except Exception as e:
                    self.logger.error(f"Failed to convert output to DecisionAgentOutput: {e}")
                    return None
            
            return None

            
        except Exception as e:
            traceback.print_exc()
            self.logger.error(f"Decision Agent failed: {str(e)}")
            return None