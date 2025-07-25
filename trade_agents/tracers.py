from agents.memory import SQLiteSession
from typing import Tuple
import secrets
import string
import os
import logging

class TraceManager:
    """Centralized trace management"""
    
    ALPHANUM = string.ascii_lowercase + string.digits
    TRACE_DB_PATH = os.getenv("TRACE_DB_PATH", "accounts.db")
    _logger = logging.getLogger(__name__)
    
    @classmethod
    def make_trace_id(cls, tag: str) -> str:
        """Generate unique trace ID"""
        tag += "0"
        pad_len = 32 - len(tag)
        random_suffix = ''.join(secrets.choice(cls.ALPHANUM) for _ in range(pad_len))
        return f"trace_{tag}{random_suffix}"

    @classmethod
    def get_shared_session(cls, trace_id: str) -> SQLiteSession:
        """Get shared SQLite session"""
        return SQLiteSession(db_path=cls.TRACE_DB_PATH, session_id=trace_id)

    @classmethod
    def start_trace(cls, tag: str = "manager") -> Tuple[str, SQLiteSession]:
        """Start new trace with session"""
        trace_id = cls.make_trace_id(tag)
        session = cls.get_shared_session(trace_id)
        cls._logger.info(f"Started new trace for {tag}")
        return (trace_id, session)