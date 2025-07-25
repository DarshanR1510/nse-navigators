import logging
from data.schemas import AgentConfig
from trade_agents.session_manager import SessionManager
from agents import trace



class BaseTradeAgent:
    """Base class for all trading agents"""
    
    def __init__(self, config: AgentConfig):
        self.name = config.name
        self.model_name = config.model_name
        self.strategy = config.strategy
        self.session = config.session
        self.trace_id = config.trace_id
        self.logger = logging.getLogger(__name__)

    # def __post_init__(self):
    #     if not self.session:
    #         self.session = SessionManager.get_session(self.name)

    async def initialize(self):
        """Initialize agent resources"""
        pass

    async def _run_with_trace(self, *args, **kwargs):
        """Run agent operations with trace context"""
        if self.trace_id:
            with trace(trace_id=self.trace_id):
                return await self._run(*args, **kwargs)
        return await self._run(*args, **kwargs)