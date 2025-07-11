from utils.util import css, js, Color
from trading_floor import agent_names, lastnames, short_model_names
from data.accounts import Account
from data.database import DatabaseQueries
import gradio as gr
import pandas as pd
import plotly.express as px
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
    "account": Color.RED,
}

all_symbols = set()

for agent_name in agent_names:
    account = Account.get(agent_name)
    all_symbols.update(account.get_holdings().keys())

new_instruments = {symbol: get_security_id(symbol) for symbol in all_symbols}
update_instruments(new_instruments)

listener_thread = threading.Thread(target=run_websocket_listener, daemon=True)
listener_thread.start()

class Trader:
    def __init__(self, name: str, lastname: str, model_name: str):
        self.name = name
        self.lastname = lastname
        self.model_name = model_name
        self.account = Account.get(name)

    def reload(self):
        self.account = Account.get(self.name)

    def get_title(self) -> str:
        return f"<div style='text-align: center;font-size:34px;'>{self.name}<span style='color:#ccc;font-size:24px;'> ({self.model_name}) - {self.lastname}</span></div>"

    def get_strategy(self) -> str:
        return self.account.get_strategy()

    def get_portfolio_value_df(self) -> pd.DataFrame:
        df = pd.DataFrame(self.account.portfolio_value_time_series, columns=["datetime", "value"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df
    
    def get_portfolio_value_chart(self):
        df = self.get_portfolio_value_df()
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
        )
        fig.update_xaxes(tickformat="%m/%d", tickangle=45, tickfont=dict(size=8))
        fig.update_yaxes(tickfont=dict(size=8), tickformat=",.0f")
        return fig
        
    def get_holdings_df(self) -> pd.DataFrame:
        """Convert holdings to DataFrame for display"""
        holdings = self.account.get_holdings()
        if not holdings:
            return pd.DataFrame(columns=["Symbol", "Quantity", "Live Price"])
        
        rows = []
        with data_lock:
            for symbol, quantity in holdings.items():
                ltp = live_prices.get(symbol, {}).get("LTP", 0.0)
                prev_ltp = live_prices.get(symbol, {}).get("prev_LTP", ltp)
                color = mapper.get("function").value if ltp > prev_ltp else (mapper.get("account").value if ltp < prev_ltp else mapper.get("trace").value)                
                # Use HTML for colored cell
                price_html = f"<span style='color:{color};font-weight:bold'>{ltp:.2f}</span>"
                rows.append({
                    "Symbol": symbol,
                    "Quantity": quantity,
                    "Live Price": price_html
                })
        df = pd.DataFrame(rows)
        return df
    
    def get_holdings_html(self) -> str:
        holdings = self.account.get_holdings()
        rows = []
        with data_lock:
            for symbol, quantity in holdings.items():
                ltp = live_prices.get(symbol, {}).get("LTP", 0.0)
                prev_ltp = live_prices.get(symbol, {}).get("prev_LTP", ltp)
                color = mapper.get("function").value if ltp > prev_ltp else (mapper.get("account").value if ltp < prev_ltp else mapper.get("trace").value)
                price_html = f"<span style='color:{color};font-weight:bold'>{ltp:.2f}</span>"
                rows.append(f"<tr><td>{symbol}</td><td>{quantity}</td><td>{price_html}</td></tr>")
        # Pad to 5 rows if needed
        while len(rows) < 5:
            rows.append("<tr><td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td></tr>")
        table_html = (
            "<div style='max-height:180px;overflow-y:auto;width:100%;'>"
            "<table style='width:100%;border-collapse:collapse;'>"
            "<thead><tr style='position:sticky;top:0;background:#222;'><th>Symbol</th><th>Quantity</th><th>Live Price</th></tr></thead>"
            "<tbody>"
            + "".join(rows) +
            "</tbody></table></div>"
        )
        return table_html
    
    def get_transactions_df(self) -> pd.DataFrame:
        """Convert transactions to DataFrame for display"""
        transactions = self.account.list_transactions()
        if not transactions:
            return pd.DataFrame(columns=["Timestamp", "Symbol", "Quantity", "Price", "Rationale"])
        
        return pd.DataFrame(transactions)
    
    def get_memory_status_df(self) -> pd.DataFrame:
        """Return a DataFrame summarizing agent memory (positions, watchlist, context) for display."""
        mem = AgentMemory(self.name)
        positions = mem.get_active_positions() or {}
        watchlist = mem.get_watchlist() or []
        context = mem.get_daily_context() or {}

        # Positions Table
        pos_rows = []
        for symbol, pos in positions.items():
            row = {"Symbol": symbol}
            if isinstance(pos, dict):
                row.update({
                    "Quantity": pos.get("quantity", "-"),
                    "Entry Price": pos.get("entry_price", "-"),
                    "Stop Loss": pos.get("stop_loss", "-"),
                    "Target": pos.get("target", "-"),
                    "Reason": pos.get("reason", "-"),
                    "Entry Date": pos.get("entry_date", "-")
                })
            pos_rows.append(row)
        pos_df = pd.DataFrame(pos_rows) if pos_rows else pd.DataFrame(columns=["Symbol", "Quantity", "Entry Price", "Stop Loss", "Target", "Reason", "Entry Date"])

        # Watchlist Table
        watchlist_df = pd.DataFrame(watchlist, columns=["Watchlist"]) if watchlist else pd.DataFrame(columns=["Watchlist"])

        # Context Table (flatten dict if needed)
        if isinstance(context, dict):
            context_items = list(context.items())
        else:
            context_items = [("Context", str(context))]
        context_df = pd.DataFrame(context_items, columns=["Key", "Value"]) if context_items else pd.DataFrame(columns=["Key", "Value"])

        # Combine all as a dict of DataFrames for display
        return {"Positions": pos_df, "Watchlist": watchlist_df, "Context": context_df}
    
    def get_portfolio_value(self) -> str:
        """Calculate total portfolio value based on current prices"""
        portfolio_value = self.account.calculate_portfolio_value() or 0.0
        pnl = self.account.calculate_profit_loss(portfolio_value) or 0.0
        color = "green" if pnl >= 0 else "red"
        emoji = "⬆" if pnl >= 0 else "⬇"
        return f"<div style='text-align: center;background-color:{color};'><span style='font-size:32px'>₹{portfolio_value:,.0f}</span><span style='font-size:24px'>&nbsp;&nbsp;&nbsp;{emoji}&nbsp;₹{pnl:,.0f}</span></div>"

    def get_logs(self, previous=None) -> str:
        logs = DatabaseQueries.read_log(self.name, last_n=13)        
        response = ""
        for log in logs:
            timestamp, type, message = log
            color = mapper.get(type, Color.WHITE).value
            response += f"<span style='color:{color}'>{timestamp} : [{type}] {message}</span><br/>"
        response = f"<div style='height:250px; overflow-y:auto;'>{response}</div>"        
        if response != previous:
            return response
        return gr.update()

    
    
class TraderView:

    def __init__(self, trader: Trader):
        self.trader = trader
        self.portfolio_value = None
        self.chart = None
        self.holdings_table = None
        self.transactions_table = None

    def make_ui(self):
        with gr.Column():
            gr.HTML(self.trader.get_title())
            with gr.Row():
                self.portfolio_value = gr.HTML(self.trader.get_portfolio_value)
            with gr.Row():
                self.chart = gr.Plot(self.trader.get_portfolio_value_chart, container=True, show_label=False)
            with gr.Row(variant="panel"):
                self.log = gr.HTML(self.trader.get_logs)
                
            # with gr.Row():
            #     self.holdings_table = gr.Dataframe(
            #         value=self.trader.get_holdings_df,
            #         label="Holdings",
            #         headers=["Symbol", "Quantity", "Live Price"],
            #         row_count=(5, "dynamic"),
            #         col_count=3,
            #         max_height=300,
            #         elem_classes=["dataframe-fix-small"],
            #         every=1,
            #         render=True
            #     )

            with gr.Row():
                self.holdings_table = gr.HTML(self.trader.get_holdings_html())

            # Memory Status as DataFrames
            mem_dfs = self.trader.get_memory_status_df()
            with gr.Row():
                self.memory_positions = gr.Dataframe(
                    value=lambda: mem_dfs["Positions"],
                    label="Memory: Positions",
                    headers=list(mem_dfs["Positions"].columns) if not mem_dfs["Positions"].empty else [],
                    row_count=(5, "dynamic"),
                    col_count=len(mem_dfs["Positions"].columns) if not mem_dfs["Positions"].empty else 0,
                    max_height=200,
                    elem_classes=["dataframe-fix-small"]
                )
            with gr.Row():
                self.memory_watchlist = gr.Dataframe(
                    value=lambda: mem_dfs["Watchlist"],
                    label="Memory: Watchlist",
                    headers=list(mem_dfs["Watchlist"].columns) if not mem_dfs["Watchlist"].empty else [],
                    row_count=(5, "dynamic"),
                    col_count=len(mem_dfs["Watchlist"].columns) if not mem_dfs["Watchlist"].empty else 0,
                    max_height=100,
                    elem_classes=["dataframe-fix-small"]
                )
            with gr.Row():
                self.memory_context = gr.Dataframe(
                    value=lambda: mem_dfs["Context"],
                    label="Memory: Context",
                    headers=list(mem_dfs["Context"].columns) if not mem_dfs["Context"].empty else [],
                    row_count=(3, "dynamic"),
                    col_count=len(mem_dfs["Context"].columns) if not mem_dfs["Context"].empty else 0,
                    max_height=100,
                    elem_classes=["dataframe-fix-small"]
                )
            with gr.Row():
                self.transactions_table = gr.Dataframe(
                    value=self.trader.get_transactions_df,
                    label="Recent Transactions",
                    headers=["Timestamp", "Symbol", "Quantity", "Price", "Rationale"],
                    row_count=(5, "dynamic"),
                    col_count=5,
                    max_height=300,
                    elem_classes=["dataframe-fix"]
                )            

        timer = gr.Timer(value=120)
        timer.tick(fn=self.refresh, inputs=[], outputs=[self.portfolio_value, self.chart, self.holdings_table, self.memory_positions, self.memory_watchlist, self.memory_context, self.transactions_table], show_progress="hidden", queue=False)
        log_timer = gr.Timer(value=0.5)
        log_timer.tick(fn=self.trader.get_logs, inputs=[self.log], outputs=[self.log], show_progress="hidden", queue=False)

    def refresh(self):
        self.trader.reload()
        mem_dfs = self.trader.get_memory_status_df()
        return (
            self.trader.get_portfolio_value(),
            self.trader.get_portfolio_value_chart(),
            self.trader.get_holdings_df(),
            mem_dfs["Positions"],
            mem_dfs["Watchlist"],
            mem_dfs["Context"],
            self.trader.get_transactions_df()
        )

# Main UI construction
def create_ui():
    """Create the main Gradio UI for the trading simulation"""
    
    traders = [Trader(trader_name, lastname, model_name) for trader_name, lastname, model_name in zip(agent_names, lastnames, short_model_names)]    
    trader_views = [TraderView(trader) for trader in traders]
  
    with gr.Blocks(title="Traders", css=css, js=js, theme=gr.themes.Default(primary_hue="sky"), fill_width=True) as ui:                
        with gr.Row():
            for trader_view in trader_views:
                trader_view.make_ui()
        with gr.Row():
                    gr.HTML(
                        "<div style='text-align:center;font-size:16px;color:#888;'>"
                        "Crafted with expertise and passion by <b>Darshan Ramani</b>."
                        "</div>"
                    )
        
    return ui

if __name__ == "__main__":
    ui = create_ui()
    share_ui = os.getenv("UI_SHARE", "False").lower() == "true"
    ui.launch(inbrowser=True, share=share_ui)