from utils.util import css, js, Color
from trading_floor import trader_names, lastnames, TradingFloor
from data.accounts import Account
from data.database import DatabaseQueries
import gradio as gr
import pandas as pd
import plotly.express as px
import numpy as np
import os
import threading
from dotenv import load_dotenv
from memory.agent_memory import AgentMemory
from market_tools.market import get_security_id
from market_tools.live_prices import live_prices, data_lock, run_websocket_listener, update_instruments

load_dotenv()

mapper = {
    "trace": Color.WHITE,
    "agent": Color.CYAN,
    "function": Color.GREEN,
    "generation": Color.YELLOW,
    "response": Color.MAGENTA,
    "account": Color.BLUE,
    "error": Color.RED,
}

all_symbols = set()

for trader_name in trader_names:
    account = Account.get(trader_name)
    all_symbols.update(account.get_holdings().keys())

new_instruments = {symbol: get_security_id(symbol) for symbol in all_symbols}
update_instruments(new_instruments)

listener_thread = threading.Thread(target=run_websocket_listener, daemon=True)
listener_thread.start()


def convert_to_native(obj):
    """
    Recursively convert numpy types and other non-native types to native Python types.
    """
    if hasattr(obj, 'item'):  # numpy scalar types
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_native(item) for item in obj)
    elif isinstance(obj, pd.Series):
        return obj.apply(convert_to_native).tolist()
    elif pd.isna(obj):
        return None
    else:
        return obj


def df_to_native(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all numpy types in DataFrame to native Python types."""
    if df.empty:
        return df
    
    # Create a copy to avoid modifying original
    df_copy = df.copy()
    
    # Convert each column
    for col in df_copy.columns:
        df_copy[col] = df_copy[col].apply(convert_to_native)
    
    # Also ensure index is native
    if hasattr(df_copy.index, 'to_list'):
        df_copy.index = df_copy.index.to_list()
    
    return df_copy


class Trader:
    def __init__(self, name: str, lastname: str, model_name: str):
        self.name = name
        self.lastname = lastname
        self.model_name = model_name
        self.account = Account.get(name)

    def reload(self):
        self.account = Account.get(self.name)

    def get_title(self) -> str:
        return f"<div style='text-align: center;font-size:34px;'>{(self.name).capitalize()}<span style='color:#ccc;font-size:24px;'> ({self.model_name})  -  {self.lastname}</span></div>"

    def get_strategy(self) -> str:
        return self.account.get_strategy()

    def get_portfolio_value_df(self) -> pd.DataFrame:
        try:
            portfolio_data = self.account.portfolio_value_time_series
            if not portfolio_data:
                return pd.DataFrame(columns=["datetime", "value"])
            
            # Convert to native types before creating DataFrame
            native_data = [convert_to_native(item) for item in portfolio_data]
            df = pd.DataFrame(native_data, columns=["datetime", "value"])
            df["datetime"] = pd.to_datetime(df["datetime"])
            return df_to_native(df)
        except Exception as e:
            print(f"Error in get_portfolio_value_df: {e}")
            return pd.DataFrame(columns=["datetime", "value"])


    def get_portfolio_value_chart(self):
        try:
            df = self.get_portfolio_value_df()
            if df.empty:
                # Return empty chart
                fig = px.line(title="No data available")
            else:
                fig = px.line(df, x="datetime", y="value")
            
            margin = dict(l=40, r=20, t=20, b=40) 
            fig.update_layout(
                height=300,
                margin=margin,
                xaxis_title=None,
                yaxis_title=None,
                paper_bgcolor="rgba(24,24,24,1)",
                plot_bgcolor="rgba(30,30,30,1)",
                font_color="#ccc",
                xaxis=dict(
                    showgrid=True,
                    gridcolor="rgba(180,180,180,0.15)",  # subtle grey grid
                    gridwidth=1
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor="rgba(180,180,180,0.15)",  # subtle grey grid
                    gridwidth=1
                ),
            )
            fig.update_xaxes(tickformat="%m/%d", tickangle=45, tickfont=dict(size=8))
            fig.update_yaxes(tickfont=dict(size=8), tickformat=",.0f")
            return fig
        except Exception as e:
            print(f"Error in get_portfolio_value_chart: {e}")
            return px.line(title="Error loading chart")



    def get_holdings_html(self) -> str:
        try:
            # Reload account to get fresh data
            self.reload()
            holdings = self.account.get_holdings()
            if not holdings:
                return "<div>No holdings</div>"
            
            agent_memory = AgentMemory(self.name)
            active_positions = agent_memory.get_active_positions() or {}    
            rows = []

            with data_lock:
                for symbol, quantity in holdings.items():                    
                    ltp = live_prices.get(symbol, {}).get("LTP", 0.0)
                    prev_ltp = live_prices.get(symbol, {}).get("prev_LTP", ltp)                    
                    entry_price = active_positions.get(symbol, {}).get('entry_price', 0.0)

                    pnl = ((ltp - entry_price) / entry_price) * 100
                    pnl_color = mapper.get("function").value if pnl >= 0 else mapper.get("error").value
                    pnl_html = f"<span style='color:{pnl_color};'>{pnl:.2f}%</span>"
                    
                    color = mapper.get("function").value if ltp > prev_ltp else (
                        mapper.get("error").value 
                    )

                    price_html = f"<span style='color:{color};'>{float(ltp):.2f}</span>"
                    rows.append(
                        f"<tr>"
                        f"<td>{symbol}</td>"
                        f"<td>{int(quantity)}</td>"
                        f"<td>{float(entry_price):.2f}</td>"
                        f"<td>{price_html}</td>"
                        f"<td>{pnl_html}</td>"
                        f"</tr>"
                    )

            # Pad to 5 rows if needed
            while len(rows) < 5:
                rows.append("<tr><td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td></tr>")

            table_html = (
                "<div style='max-height:250px;overflow-y:auto;width:100%;'>"
                "<table style='width:100%;border-collapse:collapse;'>"
                "<thead><tr style='position:sticky;top:0;background:#222;'>"
                "<th>Symbol</th>"
                "<th>Quantity</th>"
                "<th>Entry Price</th>"
                "<th>Live Price</th>"
                "<th>P&L %</th>"
                "</tr></thead>"
                "<tbody>"
                + "".join(rows) +
                "</tbody></table></div>"
            )
            return table_html
        except Exception as e:
            print(f"Error in get_holdings_html: {e}")
            return "<div>Error loading holdings</div>"


    def get_transactions_df(self) -> pd.DataFrame:
        """Convert transactions to DataFrame for display"""
        try:
            # Reload account to get fresh data
            self.reload()
            transactions = self.account.list_transactions()
            if not transactions:
                return pd.DataFrame(columns=["Timestamp", "Symbol", "Quantity", "Price", "Rationale"])

            # Convert all transaction data to native types
            native_transactions = []
            for transaction in transactions:
                native_transaction = convert_to_native(transaction)
                native_transactions.append(native_transaction)

            df = pd.DataFrame(native_transactions)
            gr.update(df, inplace=True)
            return df_to_native(df)
        except Exception as e:
            print(f"Error in get_transactions_df: {e}")
            return pd.DataFrame(columns=["Timestamp", "Symbol", "Quantity", "Price", "Rationale"])


    def get_portfolio_value(self) -> str:
        """Calculate total portfolio value based on current prices"""
        try:
            # Reload account to get fresh data
            self.reload()
            portfolio_value = convert_to_native(self.account.calculate_portfolio_value()) or 0.0
            pnl = convert_to_native(self.account.calculate_profit_loss(portfolio_value)) or 0.0
            
            portfolio_value = float(portfolio_value)
            pnl = float(pnl)
            
            color = "green" if pnl >= 0 else "red"
            emoji = "⬆" if pnl >= 0 else "⬇"
            return (
                f"<div style='text-align: center; background-color:{color}; width: 100%; box-sizing: border-box; overflow-wrap: break-word; padding: 6px 0;'>"
                f"<span style='font-size:26px;'>&nbsp;₹{portfolio_value:,.0f}&nbsp;</span>"
                f"<span style='font-size:20px;'>&nbsp;&nbsp;{emoji}&nbsp;₹{pnl:,.0f}&nbsp;</span>"
                "</div>"
            )
        except Exception as e:
            print(f"Error in get_portfolio_value: {e}")
            return "<div style='text-align: center;'>Error loading portfolio value</div>"


    def get_logs(self, previous=None) -> str:
        try:
            # Get only last 10 logs
            logs = DatabaseQueries.read_log(self.name, last_n=10)
            
            # Build log entries with proper styling
            log_entries = []
            for log in logs:
                timestamp, log_type, message = convert_to_native(log)
                color = mapper.get(log_type, Color.WHITE).value
                log_entries.append(
                    f"<div style='margin-bottom:4px;'>"
                    f"<span style='color:{color}'>{timestamp} : {message}</span>"
                    f"</div>"
                )
            
            # Create fixed-size container with latest logs
            log_html = (
                f"<div style='padding:12px;background:rgba(0,0,0,0.2);border-radius:8px;'>"
                f"{''.join(log_entries)}"
                f"</div>"
            )
            
            if log_html != previous:
                return log_html
            return gr.update()
            
        except Exception as e:
            print(f"Error in get_logs: {e}")
            return "<div>Error loading logs</div>"
    

    def get_portfolio_metrics(self) -> str:
        try:
            self.reload()
            holdings = self.account.get_holdings()
            total_positions = len(holdings)
            cash_available = int(self.account.balance)

            # Calculate PnL percentage
            portfolio_value = self.account.calculate_portfolio_value()
            pnl = self.account.calculate_profit_loss(portfolio_value)
            initial_value = int(os.getenv("INITIAL_BALANCE", "500000"))
            pnl_percentage = (pnl / initial_value * 100) if initial_value > 0 else 0
            
            # Determine color based on PnL
            color = "green" if pnl >= 0 else "red"
            pnl_symbol = "⬆" if pnl_percentage >= 0 else "⬇"            
            
            return f"""
            <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;padding:8px;'>
                <div style='background:rgba(0,0,0,0.2);padding:12px;border-radius:8px;text-align:center;'>
                    <div style='color:#888;font-size:14px;margin-bottom:4px;'>Available Balance</div>
                    <div style='font-size:20px;color:#4ecdc4;'>₹{cash_available}</div>
                </div>
                <div style='background:rgba(0,0,0,0.2);padding:12px;border-radius:8px;text-align:center;'>
                    <div style='color:#888;font-size:14px;margin-bottom:4px;'>Active Positions</div>
                    <div style='font-size:20px;color:#45b7d1;'>{total_positions}</div>
                </div>
                <div style='background:rgba(0,0,0,0.2);padding:12px;border-radius:8px;text-align:center;'>
                    <div style='color:#888;font-size:14px;margin-bottom:4px;'>Portfolio P&L</div>
                    <div style='font-size:20px;color:{color};'>
                        {pnl_symbol} {abs(pnl_percentage):.2f}%
                    </div>
                </div>
            </div>
            """
        
        except Exception as e:
            print(f"Error in get_portfolio_metrics: {e}")
            return "<div>Error loading portfolio metrics</div>"

        
    def get_agent_status(self) -> str:
        try:
            # Get last log entry
            last_logs = DatabaseQueries.read_log(self.name, last_n=1)
            if not last_logs:
                return self._create_status_html("Monitoring")
            
            # Convert to list and get first item
            last_log = list(last_logs)[0]
            timestamp, log_type, message = last_log
            
            if "error" in message.lower():
                return self._create_status_html("error")
            elif "research" in message.lower():
                return self._create_status_html("researching")
            elif "analysis" in message.lower():
                return self._create_status_html("analyzing")
            elif "decision" in message.lower():
                return self._create_status_html("deciding")
            elif "execution" in message.lower():
                return self._create_status_html("executing")
            else:
                return self._create_status_html("Monitoring")
        
        except Exception as e:
            print(f"Error in get_agent_status: {e}")
            return self._create_status_html("error")


    def _create_status_html(self, status: str) -> str:
        status_colors = {
            "Monitoring": "#888",
            "researching": "#ff6b6b",
            "analyzing": "#4ecdc4",
            "deciding": "#45b7d1",
            "executing": "#96f7d2",
            "error": "#ff0000"
        }
        
        
        if status == "Monitoring": 
            status_text = "Trader is idle and keeping watch at the current market" 
        elif status == "researching":
            status_text = "Trader is researching the market for new opportunities"
        elif status == "analyzing":
            status_text = "Trader is performing analysis on selected stocks"
        elif status == "deciding":
            status_text = "Trader is selecting best stock based on analysis"
        elif status == "executing":
            status_text = "Trader is executing trades based on the strategy"
        elif status == "error":
            status_text = "Agent encountered an error and going back to monitoring"
        else: 
            status_text = status.capitalize()
        
        color = status_colors.get(status, "#888")

        if status != "Monitoring":
            spinner = f"""
            <div class="spinner" style="display:inline-block;margin-right:12px;">
                <div style="width:12px;height:12px;background:{color};border-radius:50%;
                        animation:pulse 1s infinite;"></div>
            </div>
            """
        else:
            spinner = ""
            
        return f"""
        <div style="display:flex;align-items:center;justify-content:center;padding:6px 12px;
                    background:rgba(0,0,0,0.2);border-radius:12px;margin:8px 0;">
            {spinner}
            <span style="color:{color}">{status_text}</span>
        </div>
        """



class TraderView:
    def __init__(self, trader: Trader):
        self.trader = trader
        self.portfolio_value = None
        self.chart = None
        self.holdings_table = None
        self.transactions_table = None
        self.status_indicator = None 

    def make_ui(self):
        with gr.Column(elem_classes=["trader-card"]):
            # Enhanced title with gradient
            gr.HTML(f"""
                <div style='text-align: center;'>
                    <h1 style='font-size: 32px; margin-bottom: 4px; background: linear-gradient(90deg, #00dbde, #fc00ff);
                        -webkit-background-clip: text;
                        -webkit-text-fill-color: transparent;'>
                        {(self.trader.name).capitalize()}
                    </h1>
                    <div style='color: #aaa; font-size: 16px;'>
                        {self.trader.lastname} • {self.trader.model_name}
                    </div>
                </div>
            """)

            with gr.Row():
                self.status_indicator = gr.HTML(
                    self.trader.get_agent_status,
                    elem_classes=["agent-status"]
                )
                
            # Portfolio value card
            with gr.Column(scale=1, min_width=300):
                with gr.Group(elem_classes=["value-card"]):
                    gr.HTML("<div class='card-header'>PORTFOLIO VALUE</div>")
                    self.portfolio_value = gr.HTML(self.trader.get_portfolio_value)

            # Logs section
            with gr.Group():
                gr.HTML("<div class='card-header'>ACTIVITY LOG</div>")
                self.log = gr.HTML(self.trader.get_logs)

            
            # Chart column
            with gr.Column(scale=2):
                with gr.Group():
                    gr.HTML("<div class='card-header'>PERFORMANCE CHART</div>")
                    self.chart = gr.Plot(
                        self.trader.get_portfolio_value_chart, 
                        container=True, 
                        show_label=False
                    )

            # Holdings section
            with gr.Group():
                gr.HTML("<div class='card-header'>CURRENT HOLDINGS</div>")
                self.holdings_table = gr.HTML(self.trader.get_holdings_html) 

            
            with gr.Group():
                gr.HTML("<div class='card-header'>PORTFOLIO METRICS</div>")
                self.portfolio_metrics = gr.HTML(self.trader.get_portfolio_metrics)

                               
            
            with gr.Group():
                gr.HTML("<div class='card-header'>TRADING STRATEGY</div>")
                gr.HTML(f"<div style='padding:12px;font-size:14px;color:#ddd;'>{self.trader.get_strategy()}</div>")


            with gr.Group():
                gr.HTML("<div class='card-header'>TRANSACTIONS HISTORY</div>")
                self.transactions_table = gr.Dataframe(
                    value=self.trader.get_transactions_df,
                    headers=["Timestamp", "Symbol", "Quantity", "Price", "Rationale"],
                    row_count=(5, "dynamic"),
                    col_count=5,
                    max_height=300,
                    elem_classes=["dataframe-fix"],
                    datatype=["str", "str", "number", "number", "str"]
                )                        

            
        # Timers for updates
        timer = gr.Timer(value=10)
        timer.tick(
            fn=self.refresh_all, 
            inputs=[], 
            outputs=[self.portfolio_value, self.chart, self.transactions_table], 
            show_progress="hidden", 
            queue=False
        )

        holdings_timer = gr.Timer(value=2) 
        holdings_timer.tick(
            fn=lambda: (
                self.trader.get_holdings_html(),
                self.trader.get_portfolio_metrics()
            ),
            inputs=[],
            outputs=[self.holdings_table, self.portfolio_metrics],
            show_progress="hidden",
            queue=False
        )
        
        log_timer = gr.Timer(value=2)
        log_timer.tick(
            fn=self.trader.get_logs, 
            inputs=[self.log], 
            outputs=[self.log], 
            show_progress="hidden", 
            queue=False
        )

        status_timer = gr.Timer(value=2)
        status_timer.tick(
            fn=self.trader.get_agent_status,
            inputs=[],
            outputs=[self.status_indicator],
            show_progress="hidden",
            queue=False
        )

    def refresh_all(self):
        """Refresh all components that need live updates"""
        try:
            return (
                self.trader.get_portfolio_value(),
                self.trader.get_portfolio_value_chart(),
                self.trader.get_transactions_df()
            )
        except Exception as e:
            print(f"Error in refresh_all: {e}")
            return [
                "<div style='text-align: center;'>Error loading portfolio value</div>",
                px.line(title="Error loading chart"),
                "<div>Error loading holdings</div>",
                pd.DataFrame(columns=["Timestamp", "Symbol", "Quantity", "Price", "Rationale"])
            ]

trading_floor = TradingFloor()
model_display_names = trading_floor.short_model_names

def create_ui():
    """Create the enhanced Gradio UI for the trading simulation"""
    traders = [Trader(trader_name, lastname, model_name) for trader_name, lastname, model_name in zip(trader_names, lastnames, model_display_names)]
    trader_views = [TraderView(trader) for trader in traders]

    with gr.Blocks(
        title="NSE Navigators",
        css=css,
        js=js,
        theme=gr.themes.Default(
            primary_hue="violet",
            secondary_hue="blue",
            neutral_hue="slate"
        ),
        fill_width=True
    ) as ui:
        
        # Header section
        gr.HTML("""
            <div style='text-align:center;margin-bottom:5px;'>
                <h1 style='font-size:42px;margin-bottom:10px;background: linear-gradient(90deg, #00dbde, #fc00ff);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;'>
                    NSE NAVIGATORS
                </h1>
                <p style='color:#aaa;font-size:18px;'>
                    Real-time monitoring of AI trading agents
                </p>
            </div>
        """)
        
        # Main content grid
        with gr.Row():
            for trader_view in trader_views:
                trader_view.make_ui()
        
        # Footer
        gr.HTML("""
            <div style='text-align:center;padding:20px;margin-top:20px;color:#888;'>
                <div style='font-size:16px;margin-bottom:8px;'>
                    Crafted with <span style='color:#fc00ff;'>❤</span> by <b style='background: linear-gradient(90deg, #00dbde, #fc00ff);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;'>Darshan Ramani</b>
                </div>
                <div style='font-size:16px;color:#aaa;'>
                    Live market data • AI-powered trading • Real-time analytics
                </div>
            </div>
        """)
        
    return ui


if __name__ == "__main__":
    fonts = os.path.abspath("static/fonts")
    ui = create_ui()
    ui.launch(
        inbrowser=True,
        allowed_paths=[fonts] 
    )