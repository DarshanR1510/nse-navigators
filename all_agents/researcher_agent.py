from dotenv import load_dotenv
from agents.agent import Agent
from trade_agents.models import get_model
from trade_agents.templates import researcher_instructions
from data.schemas import SelectedSymbols, AgentConfig
from mcp_servers.mcp_servers_manager import setup_researcher_mcp_servers
import logging
from all_agents.base_trade_agent import BaseTradeAgent
from utils.json_response_extractor import extract_json_from_response
from agents import Runner
import json
import traceback    


load_dotenv(override=True)


class ResearcherAgent(BaseTradeAgent):
    """Agent responsible for market research and symbol selection"""

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.name = config.name
        self.model = get_model(config.model_name)
        self.strategy = config.strategy
        self.session = config.session
        self.logger = logging.getLogger(__name__)
        self._agent = None
        self._servers = None


    async def initialize(self):
        """Initialize agent with MCP servers"""
        if not self._servers:
            researcher_servers, memory_servers = await setup_researcher_mcp_servers()
            
            self._servers = [
                memory_servers[0],      # get_market_context
                researcher_servers[0],  # fetch
                researcher_servers[1],  # serper
                researcher_servers[2],  # brave

            ]
        
        if not self._agent:
            
            # Configure output_type based on model capabilities
            if self.model in ["gpt-4.1-mini", "gpt-4o", "gpt-4o-mini"]:
                self._agent = Agent(
                name=f"{self.name}_researcher_agent",
                instructions=researcher_instructions(self.strategy),
                model=self.model,
                mcp_servers=self._servers,
                output_type=SelectedSymbols
            )
            else:                
                self._agent = Agent(
                name=f"{self.name}_researcher_agent",
                instructions=researcher_instructions(self.strategy),
                model=self.model,
                mcp_servers=self._servers
            )


    async def run(self, context: dict) -> SelectedSymbols:
        """Run the researcher agent with context"""
        try:
            # Ensure initialization
            await self.initialize()
            
            # Prepare input
            input_text = (
                "Get the best trading opportunities based on current market context, latest trending news, events and data. "
                "Any opportunities you think are worth researching or trading, analyse it and if so, return as a SelectedSymbols object."
            )
            
            # Run agent
            result = await Runner.run(
                self._agent,
                input=f"{input_text} {json.dumps(context, default=str)}",
                # session=self.session,
                max_turns=30
            )


            # OpenAI models return typed output directly
            if isinstance(result.final_output, SelectedSymbols):                
                return result.final_output
            
            elif isinstance(result.final_output, str):
                try:
                    output_dict = extract_json_from_response(result.final_output)
                    if output_dict:                        
                        return SelectedSymbols(**output_dict)
                except Exception as e:
                    self.logger.error(f"Failed to convert output to SelectedSymbols: {e}")
                        

        except Exception as e:
            traceback.print_exc()
            self.logger.error(f"Research failed: {str(e)}")
            return None
    