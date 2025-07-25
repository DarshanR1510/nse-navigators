from dotenv import load_dotenv
from agents.agent import Agent
from trade_agents.models import get_model
from trade_agents.templates import technical_instructions
from data.schemas import SymbolsAnalysis, AgentConfig
from agents import Runner
from agents.agent_output import AgentOutputSchema
from all_agents.base_trade_agent import BaseTradeAgent
import json
from utils.json_response_extractor import extract_json_from_response
import traceback    
import logging

load_dotenv(override=True)


class TechnicalAgent(BaseTradeAgent):
    def __init__(self, config: AgentConfig, mcp_setup: list):
        super().__init__(config)
        self.model = get_model(config.model_name)
        self.strategy = config.strategy
        self.mcp_servers = mcp_setup
        self.session = config.session
        self.logger = logging.getLogger(__name__)
        
        


        # Configure output_type based on model capabilities
        if self.model in ["gpt-4.1-mini", "gpt-4o", "gpt-4o-mini"]:
            self._agent = Agent(
            name=f"{self.name}_technical_agent",
            instructions=technical_instructions(self.strategy),
            model=self.model,
            mcp_servers=self.mcp_servers,
            output_type=AgentOutputSchema(SymbolsAnalysis, strict_json_schema=False)
        )
        else:                
            self._agent = Agent(
            name=f"{self.name}_technical_agent",
            instructions=technical_instructions(self.strategy),
            model=self.model,
            mcp_servers=self.mcp_servers,            
        )
            

    
    async def run(self, researched_symbols: dict) -> SymbolsAnalysis:
        """Run technical analysis on selected symbols"""
        try:
            input_text = (
                "Analyze these companies using technical indicators and price action. Focus on trend strength, "
                "momentum, volume patterns, and key support/resistance levels. For each company, evaluate:\n"
                "- Primary Trend: Moving averages (EMA) direction and slopes\n"
                "- Momentum: MACD signals and RSI conditions\n"
                "- Volatility: Bollinger Band patterns and breakouts\n"
                "- Volume: Recent volume patterns and breakout confirmations\n"
                "- Support/Resistance: Key price levels and potential reversals\n"
                "- Relative Strength: Performance vs peers and market\n\n"
                "Rate conviction 1-10 and provide clear entry, stop-loss (<15% from entry), and target price. "
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
                session=self.session,
                max_turns=30
            )

            # OpenAI models return typed output directly
            if isinstance(result.final_output, SymbolsAnalysis):
                self.logger.info(f"Received Technical analysis {result.final_output}")
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
            self.logger.error(f"Technical analysis failed: {str(e)}")
            return None