from dotenv import load_dotenv
from agents.agent import Agent
from trade_agents.models import get_model
from trade_agents.templates import fundamental_instructions
from data.schemas import SymbolsAnalysis, AgentConfig
from agents import Runner
from agents.agent_output import AgentOutputSchema
import json
import traceback    
from utils.json_response_extractor import extract_json_from_response
from all_agents.base_trade_agent import BaseTradeAgent
import logging

load_dotenv(override=True)


class FundamentalAgent(BaseTradeAgent):
    def __init__(self, config: AgentConfig, mcp_setup: list):
        super().__init__(config)
        self.model = get_model(config.model_name)
        self.strategy = config.strategy
        self.mcp_servers = mcp_setup
        self.session = config.session
        self.logger = logging.getLogger(__name__)
        
        

        # Configure output_type based on model capabilities
        if self.model in ["gpt-4.1-mini", "gpt-4o", "gpt-4o-mini"]:
            self._agent =  Agent(
            name=f"{self.name}_fundamental_agent",
            instructions=fundamental_instructions(self.strategy),
            model=self.model,
            mcp_servers=self.mcp_servers,
            output_type=AgentOutputSchema(SymbolsAnalysis, strict_json_schema=False)
        )
        else:                
            self._agent =  Agent(
            name=f"{self.name}_fundamental_agent",
            instructions=fundamental_instructions(self.strategy),
            model=self.model,
            mcp_servers=self.mcp_servers,            
        )
 
    
    async def run(self, researched_symbols: dict) -> SymbolsAnalysis:
        """Run fundamental analysis on selected symbols"""
        try:
            # Prepare input for fundamental analysis
            input_text = (
                "Analyze these companies using financial metrics and industry data. Focus on valuation, "
                "growth, profitability, and balance sheet health. Be thorough but concise in your report. "
                "For each company, evaluate based on:\n"
                "- Financial health: P/E, P/B, debt levels, cash flow quality\n"
                "- Growth trajectory: Sales, profit trends, margins\n"
                "- Industry position: Market share, competitive advantages\n"
                "Rate conviction 1-10 and justify. Align analysis with current strategy. "
                "Return analysis in SymbolsAnalysis format."
            )

            symbols_data = (
                researched_symbols.dict() 
                if hasattr(researched_symbols, 'dict') 
                else researched_symbols
            )
            
            # Run agent with prepared context
            result = await Runner.run(
                self._agent,
                input=f"{input_text}, \nAnalyze these symbols: {json.dumps(symbols_data)}",
                # session=self.session,
                max_turns=15
            )

            # OpenAI models return typed output directly
            if isinstance(result.final_output, SymbolsAnalysis):
                self.logger.info(f"Received Fundamental analysis {result.final_output}")
                return result.final_output
            
            elif isinstance(result.final_output, str):
                try:
                    output_dict = extract_json_from_response(result.final_output)
                    if output_dict:
                        return SymbolsAnalysis(**output_dict)                        
                    
                except Exception as e:
                    self.logger.error(f"Failed to convert output to SymbolsAnalysis: {e}")
                    return None
                    
            return None         

                    

        except Exception as e:
            traceback.print_exc()
            self.logger.error(f"Fundamental analysis failed: {str(e)}")
            return None
        
