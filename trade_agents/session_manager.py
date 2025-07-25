from typing import Dict, Tuple
from agents.memory import SQLiteSession
import logging
from trade_agents.tracers import TraceManager

from contextlib import asynccontextmanager

class SessionManager:
    """Centralized session management with integrated tracing"""
    
    _traces: Dict[str, str] = {}
    _sessions: Dict[str, Tuple[str, SQLiteSession]] = {}
    _logger = logging.getLogger(__name__)

    @classmethod
    def _create_session(cls, trader_name: str) -> Tuple[str, SQLiteSession]:
        """Create new session and trace"""
        trace_id, session = TraceManager.start_trace(trader_name)
        cls._sessions[trader_name] = (trace_id, session)
        return cls._sessions[trader_name]

    @classmethod
    def get_session(cls, trader_name: str) -> SQLiteSession:
        """Get or create shared session with trace"""
        if trader_name not in cls._sessions:
            cls._create_shared_trace(trader_name)
        return cls._sessions[trader_name]

    @classmethod
    async def get_trace_id(cls, trader_name: str) -> str:
        """Get shared trace ID for trader"""
        if trader_name not in cls._traces:
            cls._create_shared_trace(trader_name)
        return cls._traces[trader_name]
    
    @classmethod
    def _create_shared_trace(cls, trader_name: str) -> Tuple[str, SQLiteSession]:
        """Create shared trace and session for trader"""
        trace_id = TraceManager.make_trace_id(trader_name)
        session = TraceManager.get_shared_session(trace_id)
        cls._traces[trader_name] = trace_id
        cls._sessions[trader_name] = session
        cls._logger.info(f"Started shared trace for {trader_name}")
        return trace_id, session

    @classmethod
    async def get_session_and_trace(cls, trader_name: str) -> Tuple[str, SQLiteSession]:
        """Get both shared trace_id and session"""
        if trader_name not in cls._traces:
            cls._create_shared_trace(trader_name)
        return cls._traces[trader_name], cls._sessions[trader_name]    

    @classmethod
    @asynccontextmanager
    async def session_context(cls, trader_name: str):
        """Session context manager with proper async handling"""
        try:
            session = cls.get_session(trader_name)          
            yield session
        except Exception as e:
            cls._logger.error(f"Session error: {e}")
            raise