from dotenv import load_dotenv
from agents.agent import Agent
from typing import List
from trade_agents.models import get_model
from trade_agents.templates import execution_instructions
# from agents.agent import Agent, AgentOutputSchemaBase
from data.schemas import ExecutionAgentOutput, AgentConfig, TradeCandidate
from all_agents.base_trade_agent import BaseTradeAgent
from agents import Runner
import json
from utils.json_response_extractor import extract_json_from_response
import traceback    
import logging

load_dotenv(override=True)

class ExecutionAgent(BaseTradeAgent):
    def __init__(self, config: AgentConfig, mcp_setup: list):
        super().__init__(config)
        self.name = config.name
        self.model = get_model(config.model_name)
        self.mcp_servers = mcp_setup
        self.session = config.session
        self.logger = logging.getLogger(__name__)

        # Configure output_type based on model capabilities
        if self.model in ["gpt-4.1-mini", "gpt-4o", "gpt-4o-mini"]:
            self._agent = Agent(
            name=f"{self.name}_execution_agent",
            instructions=execution_instructions(self.name),
            model=self.model,
            mcp_servers=self.mcp_servers,
            output_type=ExecutionAgentOutput
        )
        else:                
            self._agent = Agent(
            name=f"{self.name}_execution_agent",
            instructions=execution_instructions(self.name),
            model=self.model,
            mcp_servers=self.mcp_servers,                
            )

    async def run(self, trade_candidate: TradeCandidate) -> ExecutionAgentOutput:
        """Run technical analysis on selected symbols"""
        try:
            input_text = (
                "Execute the following trade based on the provided trade candidate. "
                "Ensure to validate the entry price, stop loss, and target price. "
                "Once trade execution is completed, send a push notification to the user. "
                "Provide execution status, trade details and push status."
            )

            # Run agent with prepared context
            trade_candidate_str = json.dumps(trade_candidate, indent=2, default=str)

            result = await Runner.run(
                self._agent,
                input=f"{input_text}\nExecute the given trade:\n{trade_candidate_str}",
                # session=self.session,
                max_turns=5
            )
            
            # OpenAI models return typed output directly
            if isinstance(result.final_output, ExecutionAgentOutput):
                return result.final_output                        
            
            
            # Handle empty watchlist case
            elif isinstance(result.final_output, str):
                try:
                    output_dict = extract_json_from_response(result.final_output)
                    
                    if output_dict:
                        # Handle empty watchlist by setting it to None
                        if not output_dict.get("watchlist") or output_dict.get("watchlist") == {}:
                            output_dict["watchlist"] = None
                        
                        # If there's a trade candidate but no watchlist, that's fine
                        if output_dict.get("trade_candidate") and output_dict.get("decision") == "TRADE":
                            return ExecutionAgentOutput(
                                execution_status="PENDING",
                                trade_details=TradeCandidate(**output_dict["trade_candidate"]),
                                push_sent=False
                            )
                        
                        return ExecutionAgentOutput(**output_dict)
                except Exception as e:
                    self.logger.error(f"Failed to convert output to ExecutionAgentOutput: {e}")
                    return None
            
            return None
                
        except Exception as e:
            traceback.print_exc()
            self.logger.error(f"Technical analysis failed: {str(e)}")
            return None
