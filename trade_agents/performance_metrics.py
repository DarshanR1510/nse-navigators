"""
Performance Metrics Module for NSE Navigators Trading System

This module provides comprehensive performance tracking and analysis
for the trading system including agent performance, portfolio metrics,
and risk-adjusted returns.
"""

import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import pandas as pd
from utils.redis_client import main_redis_client as r
from data.database import DatabaseQueries
from data.schemas import IST


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    trader_name: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    total_return_percent: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    active_positions: int = 0
    avg_holding_period: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win_streak: int = 0
    largest_loss_streak: int = 0
    risk_adjusted_return: float = 0.0
    calmar_ratio: float = 0.0
    sortino_ratio: float = 0.0
    updated_at: str = ""


class PerformanceTracker:
    """
    Comprehensive performance tracking and analysis system
    """
    
    def __init__(self, trader_name: str, benchmark_symbol: str = "NIFTY"):
        """
        Initialize performance tracker
        
        Args:
            trader_name: Name of the agent
            benchmark_symbol: Benchmark symbol for comparison
        """
        self.trader_name = trader_name
        self.benchmark_symbol = benchmark_symbol
        self.redis_client = r
        self.logger = logging.getLogger(__name__)
        
    def calculate_agent_performance(self, 
                                  positions: Dict, 
                                  closed_positions: List,
                                  portfolio_value: float,
                                  initial_portfolio_value: float = 500000.0) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics for an agent
        
        Args:
            positions: Active positions dictionary
            closed_positions: List of closed positions
            portfolio_value: Current portfolio value
            initial_portfolio_value: Initial portfolio value
            
        Returns:
            PerformanceMetrics object
        """
        try:
            # Initialize metrics
            metrics = PerformanceMetrics(trader_name=self.trader_name)
            
            # Combine all positions for analysis
            all_positions = list(positions.values()) + closed_positions
            
            if not all_positions:
                metrics.updated_at = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
                return metrics
            
            # Basic counts
            metrics.active_positions = len(positions)
            metrics.total_trades = len(all_positions)
            
            # Calculate P&L metrics
            total_pnl = 0.0
            winning_trades = 0
            losing_trades = 0
            wins = []
            losses = []
            holding_periods = []
            
            for position in all_positions:
                # Calculate P&L
                if position.status == "ACTIVE":
                    pnl = position.unrealized_pnl
                else:
                    pnl = position.realized_pnl
                
                total_pnl += pnl
                
                # Categorize trades
                if pnl > 0:
                    winning_trades += 1
                    wins.append(pnl)
                elif pnl < 0:
                    losing_trades += 1
                    losses.append(abs(pnl))
                
                # Calculate holding period
                if position.entry_date:
                    entry_date = datetime.strptime(position.entry_date, "%Y-%m-%d %H:%M:%S")
                    if position.status == "ACTIVE":
                        holding_period = (datetime.now(IST) - entry_date).days
                    else:
                        # Assume exit date is today for closed positions
                        holding_period = (datetime.now(IST) - entry_date).days
                    holding_periods.append(holding_period)
            
            # Calculate basic metrics
            metrics.total_pnl = total_pnl
            metrics.winning_trades = winning_trades
            metrics.losing_trades = losing_trades
            metrics.total_return_percent = (total_pnl / initial_portfolio_value) * 100
            
            # Calculate win rate
            closed_trades = winning_trades + losing_trades
            if closed_trades > 0:
                metrics.win_rate = (winning_trades / closed_trades) * 100
            
            # Calculate profit factor
            total_wins = sum(wins) if wins else 0
            total_losses = sum(losses) if losses else 0
            if total_losses > 0:
                metrics.profit_factor = total_wins / total_losses
            else:
                metrics.profit_factor = float('inf') if total_wins > 0 else 0
            
            # Calculate averages
            if wins:
                metrics.avg_win = sum(wins) / len(wins)
                metrics.best_trade = max(wins)
            
            if losses:
                metrics.avg_loss = sum(losses) / len(losses)
                metrics.worst_trade = -max(losses)
            
            if holding_periods:
                metrics.avg_holding_period = sum(holding_periods) / len(holding_periods)
            
            # Calculate streaks
            metrics.largest_win_streak, metrics.largest_loss_streak = self._calculate_streaks(all_positions)
            
            # Calculate risk-adjusted metrics
            metrics.max_drawdown = self._calculate_max_drawdown(all_positions)
            metrics.sharpe_ratio = self._calculate_sharpe_ratio(all_positions, portfolio_value)
            metrics.sortino_ratio = self._calculate_sortino_ratio(all_positions, portfolio_value)
            
            # Calculate risk-adjusted return
            if metrics.max_drawdown > 0:
                metrics.risk_adjusted_return = metrics.total_return_percent / metrics.max_drawdown
                metrics.calmar_ratio = metrics.total_return_percent / metrics.max_drawdown
            
            # Calculate alpha and beta (simplified)
            metrics.alpha, metrics.beta = self._calculate_alpha_beta(all_positions)
            
            metrics.updated_at = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
            
            # Cache metrics in Redis
            self._cache_metrics(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return PerformanceMetrics(trader_name=self.trader_name)
    

    def get_portfolio_performance(self, all_agents_positions: Dict[str, Dict]) -> Dict[str, any]:
        """
        Calculate overall portfolio performance across all agents
        
        Args:
            all_agents_positions: Dictionary of {trader_name: {positions, closed_positions}}
            
        Returns:
            Dictionary with portfolio performance metrics
        """
        try:
            portfolio_metrics = {
                "total_agents": len(all_agents_positions),
                "total_active_positions": 0,
                "total_trades": 0,
                "total_pnl": 0.0,
                "total_winning_trades": 0,
                "total_losing_trades": 0,
                "portfolio_win_rate": 0.0,
                "portfolio_profit_factor": 0.0,
                "best_performing_agent": "",
                "worst_performing_agent": "",
                "agent_performances": {}
            }
            
            agent_pnls = {}
            
            for trader_name, data in all_agents_positions.items():
                positions = data.get("positions", {})
                closed_positions = data.get("closed_positions", [])
                
                # Calculate agent performance
                agent_metrics = self.calculate_agent_performance(
                    positions, closed_positions, data.get("portfolio_value", 500000.0)
                )
                
                # Update portfolio totals
                portfolio_metrics["total_active_positions"] += len(positions)
                portfolio_metrics["total_trades"] += agent_metrics.total_trades
                portfolio_metrics["total_pnl"] += agent_metrics.total_pnl
                portfolio_metrics["total_winning_trades"] += agent_metrics.winning_trades
                portfolio_metrics["total_losing_trades"] += agent_metrics.losing_trades
                
                # Store agent performance
                portfolio_metrics["agent_performances"][trader_name] = asdict(agent_metrics)
                agent_pnls[trader_name] = agent_metrics.total_pnl
            
            # Calculate portfolio-level metrics
            total_closed_trades = portfolio_metrics["total_winning_trades"] + portfolio_metrics["total_losing_trades"]
            if total_closed_trades > 0:
                portfolio_metrics["portfolio_win_rate"] = (
                    portfolio_metrics["total_winning_trades"] / total_closed_trades
                ) * 100
            
            # Find best and worst performing agents
            if agent_pnls:
                portfolio_metrics["best_performing_agent"] = max(agent_pnls, key=agent_pnls.get)
                portfolio_metrics["worst_performing_agent"] = min(agent_pnls, key=agent_pnls.get)
            
            return portfolio_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio performance: {e}")
            return {"error": str(e)}
    

    def generate_performance_report(self, metrics: PerformanceMetrics) -> str:
        """
        Generate a formatted performance report
        
        Args:
            metrics: PerformanceMetrics object
            
        Returns:
            Formatted report string
        """
        report = f"""
            ═══════════════════════════════════════════════════════════════
            🏆 PERFORMANCE REPORT - {metrics.trader_name}
            ═══════════════════════════════════════════════════════════════

            📊 TRADING SUMMARY
            ──────────────────────────────────────────────────────────────
            • Total Trades: {metrics.total_trades}
            • Active Positions: {metrics.active_positions}
            • Winning Trades: {metrics.winning_trades}
            • Losing Trades: {metrics.losing_trades}
            • Win Rate: {metrics.win_rate:.2f}%

            💰 PROFIT & LOSS
            ──────────────────────────────────────────────────────────────
            • Total P&L: ₹{metrics.total_pnl:,.2f}
            • Total Return: {metrics.total_return_percent:.2f}%
            • Best Trade: ₹{metrics.best_trade:,.2f}
            • Worst Trade: ₹{metrics.worst_trade:,.2f}
            • Average Win: ₹{metrics.avg_win:,.2f}
            • Average Loss: ₹{metrics.avg_loss:,.2f}

            📈 RISK METRICS
            ──────────────────────────────────────────────────────────────
            • Profit Factor: {metrics.profit_factor:.2f}
            • Max Drawdown: {metrics.max_drawdown:.2f}%
            • Sharpe Ratio: {metrics.sharpe_ratio:.2f}
            • Sortino Ratio: {metrics.sortino_ratio:.2f}
            • Risk-Adjusted Return: {metrics.risk_adjusted_return:.2f}
            • Calmar Ratio: {metrics.calmar_ratio:.2f}

            🎯 PERFORMANCE INDICATORS
            ──────────────────────────────────────────────────────────────
            • Alpha: {metrics.alpha:.2f}%
            • Beta: {metrics.beta:.2f}
            • Avg Holding Period: {metrics.avg_holding_period:.1f} days
            • Largest Win Streak: {metrics.largest_win_streak}
            • Largest Loss Streak: {metrics.largest_loss_streak}

            🏅 PERFORMANCE RATING
            ──────────────────────────────────────────────────────────────
            """
        
        # Add performance rating
        rating = self._calculate_performance_rating(metrics)
        report += f"• Overall Rating: {rating['rating']} ({rating['score']:.1f}/100)\n"
        report += f"• Status: {rating['status']}\n"
        
        report += f"""
            ──────────────────────────────────────────────────────────────
            📅 Report Generated: {metrics.updated_at}
            ═══════════════════════════════════════════════════════════════
            """
        
        return report


    def _calculate_streaks(self, positions: List) -> Tuple[int, int]:
        """Calculate win/loss streaks"""
        if not positions:
            return 0, 0
        
        # Sort positions by entry date
        sorted_positions = sorted(positions, key=lambda x: x.entry_date)
        
        current_win_streak = 0
        current_loss_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        
        for position in sorted_positions:
            if position.status == "ACTIVE":
                pnl = position.unrealized_pnl
            else:
                pnl = position.realized_pnl
            
            if pnl > 0:
                current_win_streak += 1
                current_loss_streak = 0
                max_win_streak = max(max_win_streak, current_win_streak)
            elif pnl < 0:
                current_loss_streak += 1
                current_win_streak = 0
                max_loss_streak = max(max_loss_streak, current_loss_streak)
        
        return max_win_streak, max_loss_streak
    
    
    def _calculate_max_drawdown(self, positions: List) -> float:
        """Calculate maximum drawdown"""
        if not positions:
            return 0.0
        
        # This is a simplified calculation
        # In practice, you'd need daily portfolio values
        cumulative_returns = []
        cumulative_pnl = 0
        
        for position in positions:
            if position.status == "ACTIVE":
                pnl = position.unrealized_pnl
            else:
                pnl = position.realized_pnl
            
            cumulative_pnl += pnl
            cumulative_returns.append(cumulative_pnl)
        
        if not cumulative_returns:
            return 0.0
        
        # Calculate max drawdown
        peak = cumulative_returns[0]
        max_drawdown = 0
        
        for value in cumulative_returns:
            if value > peak:
                peak = value
            drawdown = (peak - value) / abs(peak) * 100 if peak != 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    

    def _calculate_sharpe_ratio(self, positions: List, portfolio_value: float) -> float:
        """Calculate Sharpe ratio (simplified)"""
        if not positions:
            return 0.0
        
        returns = []
        for position in positions:
            if position.status == "ACTIVE":
                pnl = position.unrealized_pnl
            else:
                pnl = position.realized_pnl
            
            if portfolio_value > 0:
                return_pct = (pnl / portfolio_value) * 100
                returns.append(return_pct)
        
        if len(returns) < 2:
            return 0.0
        
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
        std_dev = variance ** 0.5
        
        # Assume risk-free rate of 6% (simplified)
        risk_free_rate = 6.0
        
        if std_dev > 0:
            return (mean_return - risk_free_rate) / std_dev
        return 0.0


    def _calculate_sortino_ratio(self, positions: List, portfolio_value: float) -> float:
        """Calculate Sortino ratio (simplified)"""
        if not positions:
            return 0.0
        
        returns = []
        negative_returns = []
        
        for position in positions:
            if position.status == "ACTIVE":
                pnl = position.unrealized_pnl
            else:
                pnl = position.realized_pnl
            
            if portfolio_value > 0:
                return_pct = (pnl / portfolio_value) * 100
                returns.append(return_pct)
                if return_pct < 0:
                    negative_returns.append(return_pct)
        
        if len(returns) < 2 or len(negative_returns) == 0:
            return 0.0
        
        mean_return = sum(returns) / len(returns)
        downside_variance = sum(r ** 2 for r in negative_returns) / len(negative_returns)
        downside_deviation = downside_variance ** 0.5
        
        risk_free_rate = 6.0
        
        if downside_deviation > 0:
            return (mean_return - risk_free_rate) / downside_deviation
        return 0.0
    

    def _calculate_alpha_beta(self, positions: List) -> Tuple[float, float]:
        """Calculate alpha and beta (simplified)"""
        # This is a simplified calculation
        # In practice, you'd need benchmark returns correlation
        if not positions:
            return 0.0, 1.0
        
        # Simplified: assume beta of 1.0 and calculate alpha based on excess return
        total_return = sum(pos.unrealized_pnl if pos.status == "ACTIVE" else pos.realized_pnl for pos in positions)
        
        # Assume benchmark return of 12% annually
        benchmark_return = 12.0
        
        # Calculate alpha as excess return over benchmark
        alpha = total_return - benchmark_return
        beta = 1.0  # Simplified assumption
        
        return alpha, beta


    def _calculate_performance_rating(self, metrics: PerformanceMetrics) -> Dict[str, any]:
        """Calculate overall performance rating"""
        score = 0
        
        # Win rate (30 points)
        if metrics.win_rate >= 70:
            score += 30
        elif metrics.win_rate >= 60:
            score += 25
        elif metrics.win_rate >= 50:
            score += 20
        elif metrics.win_rate >= 40:
            score += 15
        else:
            score += 10
        
        # Profit factor (25 points)
        if metrics.profit_factor >= 3.0:
            score += 25
        elif metrics.profit_factor >= 2.0:
            score += 20
        elif metrics.profit_factor >= 1.5:
            score += 15
        elif metrics.profit_factor >= 1.0:
            score += 10
        else:
            score += 5
        
        # Return percentage (25 points)
        if metrics.total_return_percent >= 25:
            score += 25
        elif metrics.total_return_percent >= 20:
            score += 20
        elif metrics.total_return_percent >= 15:
            score += 15
        elif metrics.total_return_percent >= 10:
            score += 10
        elif metrics.total_return_percent >= 5:
            score += 5
        
        # Max drawdown (20 points)
        if metrics.max_drawdown <= 5:
            score += 20
        elif metrics.max_drawdown <= 10:
            score += 15
        elif metrics.max_drawdown <= 15:
            score += 10
        elif metrics.max_drawdown <= 20:
            score += 5
        
        # Determine rating
        if score >= 90:
            rating = "🏆 EXCELLENT"
            status = "OUTSTANDING PERFORMANCE"
        elif score >= 80:
            rating = "🥇 VERY GOOD"
            status = "STRONG PERFORMANCE"
        elif score >= 70:
            rating = "🥈 GOOD"
            status = "SOLID PERFORMANCE"
        elif score >= 60:
            rating = "🥉 AVERAGE"
            status = "ACCEPTABLE PERFORMANCE"
        else:
            rating = "❌ POOR"
            status = "NEEDS IMPROVEMENT"
        
        return {
            "score": score,
            "rating": rating,
            "status": status
        }
    

    def _cache_metrics(self, metrics: PerformanceMetrics):
        """Cache metrics in Redis"""
        try:
            self.redis_client.setex(
                f"performance:{self.trader_name}",
                3600,  # 1 hour TTL
                json.dumps(asdict(metrics))
            )
        except Exception as e:
            self.logger.error(f"Error caching performance metrics: {e}")
    
    
    def get_cached_metrics(self) -> Optional[PerformanceMetrics]:
        """Get cached metrics from Redis"""
        try:
            cached_data = self.redis_client.get(f"performance:{self.trader_name}")
            if cached_data:
                data = json.loads(cached_data)
                return PerformanceMetrics(**data)
        except Exception as e:
            self.logger.error(f"Error getting cached metrics: {e}")
        return None


# Utility functions for performance analysis
def compare_agents_performance(agent_metrics: Dict[str, PerformanceMetrics]) -> Dict[str, any]:
    """
    Compare performance across multiple agents
    
    Args:
        agent_metrics: Dictionary of {trader_name: PerformanceMetrics}
        
    Returns:
        Comparison analysis
    """
    if not agent_metrics:
        return {"error": "No agent metrics provided"}
    
    # Calculate rankings
    rankings = {
        "by_total_return": sorted(agent_metrics.items(), key=lambda x: x[1].total_return_percent, reverse=True),
        "by_win_rate": sorted(agent_metrics.items(), key=lambda x: x[1].win_rate, reverse=True),
        "by_profit_factor": sorted(agent_metrics.items(), key=lambda x: x[1].profit_factor, reverse=True),
        "by_sharpe_ratio": sorted(agent_metrics.items(), key=lambda x: x[1].sharpe_ratio, reverse=True),
        "by_total_pnl": sorted(agent_metrics.items(), key=lambda x: x[1].total_pnl, reverse=True)
    }
    
    # Calculate averages
    total_agents = len(agent_metrics)
    averages = {
        "avg_return": sum(m.total_return_percent for m in agent_metrics.values()) / total_agents,
        "avg_win_rate": sum(m.win_rate for m in agent_metrics.values()) / total_agents,
        "avg_profit_factor": sum(m.profit_factor for m in agent_metrics.values()) / total_agents,
        "avg_sharpe_ratio": sum(m.sharpe_ratio for m in agent_metrics.values()) / total_agents,
        "avg_max_drawdown": sum(m.max_drawdown for m in agent_metrics.values()) / total_agents
    }
    
    return {
        "rankings": rankings,
        "averages": averages,
        "total_agents": total_agents,
        "analysis_timestamp": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
    }