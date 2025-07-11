PROJECT DETAILS:

This project is a robust, multi-agent trading system designed for Indian stock markets. It uses advanced AI agents, each with their own trading strategies (like Warren, George, Ray, and Cathie), to analyze stocks and make buy/sell decisions. All trading actions are managed centrally to ensure safety, consistency, and compliance with risk rules.

The system uses two main types of storage:

Redis for fast, real-time data access (like agent memory and quick lookups)
SQLite for reliable, long-term storage (like trade history and account records)
Technical and fundamental analysis tools are built in, including indicators like SMA, EMA, MACD, Bollinger Bands, and RSI. These tools are optimized to give clear, recent, and easy-to-use results for both the AI agents and any large language models (LLMs) that help with decision-making.

Trade execution is fully centralized in a dedicated module, so all agents must go through the same process to place trades. This ensures that risk management, position tracking, and compliance checks are always enforced. There is only one PositionManager per agent, which keeps track of all open trades and positions.

The system is designed for reliability and scalability. It includes strong error handling, detailed logging, and robust data validation to prevent mistakes and make debugging easier. Data is kept in sync between Redis and SQLite, and special care is taken to handle type conversions and edge cases.

Agent instructions and prompts are carefully crafted to get the best results from LLMs, making the system both powerful and cost-effective. Utility functions and batch analysis tools are also made robust and easy to use, so the system can handle large amounts of data and many agents at once.

Overall, this project provides a flexible, safe, and efficient platform for automated stock trading, combining the speed of Redis, the reliability of SQLite, and the intelligence of multiple AI agents—all working together under strict risk controls.















# NSE Navigators: All-in-One Project Plan

---

## 1. Project Overview

**NSE Navigators** is an intelligent, modular, multi-agent trading system for Indian equities. It leverages LLM-powered agents, robust financial data scraping, MCP (Model Context Protocol) tools, and a modern UI to autonomously research, analyze, and trade stocks. The system is designed for extensibility, reliability, and transparency, with a focus on agentic workflows and real-time monitoring.

---

## 2. Architecture & Technology Stack

### **Core Technologies**
- **Python 3.12+**: Main language for agents, tools, and servers
- **Node.js (Playwright)**: For dynamic web scraping via MCP
- **Redis**: Fast, persistent cache for symbol resolution and API results
- **SQLite**: Lightweight database for accounts, scripts, and logs
- **Gradio**: Modern UI/dashboard for real-time monitoring
- **MCP (Model Context Protocol)**: Modular tool/server protocol for agent-tool communication
- **ntfy.sh**: Push notification service for trade alerts

### **Key Libraries**
- `openai`, `anthropic`, `openai-agents`: LLM and agent orchestration
- `requests`, `httpx`, `bs4`, `lxml`: Web scraping and HTTP
- `playwright`: Dynamic scraping fallback
- `pandas`, `plotly`: Data analysis and visualization
- `pydantic`: Data validation and models
- `python-dotenv`: Environment variable management
- `redis`: Redis client for Python

---

## 3. Project Structure

```
project-root/
│
├── src/
│   ├── agents/                # Agent logic, strategies, orchestration
│   ├── mcp_tools/             # MCP tool/server implementations
│   ├── data/                  # Scraping, market data, DB
│   ├── ui/                    # Gradio UI, templates, utils
│   ├── redis_data_import.py   # Script to load symbol data into Redis
│   ├── ...                    # Other modules
│
├── app.py                    # Main UI entry point
├── trading_floor.py           # Agent orchestration script
├── requirements.txt
├── pyproject.toml
├── README.md
└── ...
```

---

## 4. Agents & Workflows

### **Agents**
- **Researcher Agent**: Scans news, trends, and web for opportunities using LLMs and MCP tools.
- **Financial Analyst Agent**: Performs deep financial and fundamental analysis using scraped and API data.
- **Technical Analyst Agent**: Computes technical indicators (SMA, EMA, MACD, RSI) and signals.
- **Trader Agent**: Executes trades, manages portfolio, and applies risk management.

Each agent operates with:
- Distinct strategies (e.g., value, macro, systematic, innovation)
- Dedicated MCP toolset (symbol resolution, price/financials, scraping, notifications)
- Real-time state and trace logging

### **Agentic Workflow**
1. **Symbol Resolution**: Always resolve company name to symbol using Redis-backed MCP tool.
2. **Data Fetching**: Use MCP tools for prices, financials, and technicals (with caching and rate-limiting).
3. **Analysis**: Each agent applies its strategy using LLMs and tool outputs.
4. **Trade Execution**: Trader agent places buy/sell orders, updates portfolio, and triggers push notifications.
5. **Monitoring**: All actions and trades are visible in the Gradio dashboard.

---

## 5. MCP Tools & Servers

### **Key MCP Tools**
- **resolve_symbol**: Fast, Redis-backed symbol lookup (no API call, instant response)
- **get_symbol_price / get_prices_with_cache**: Batch and cached price fetching, rate-limit safe
- **get_closing_sma/ema/macd/rsi**: Technical indicator tools, with caching and batching
- **get_financial_data**: Structured financials from Screener.in (requests + Playwright fallback)
- **push**: Sends trade notifications via ntfy.sh

### **MCP Server Design**
- Each tool is registered as an async MCP tool
- Global Redis cache is used for all symbol and price lookups
- All tools have robust error handling and logging

---

## 6. Data & Caching

- **Symbol/Name/Display Indexes**: Loaded from SQLite `scripts` table into Redis at startup
- **Price/Indicator Cache**: All price and indicator results cached in Redis (5-min TTL)
- **Portfolio/Profit Calculation**: Uses cached prices for instant, rate-limit-safe reporting

---

## 7. UI & Monitoring

- **Gradio Dashboard**: Real-time view of agent actions, trades, and portfolio value
- **Push Notifications**: ntfy.sh integration for trade alerts (with agent/action/quantity in header)
- **Trace Logging**: All agent actions and tool calls are logged for transparency and debugging

---

## 8. Deployment & Operations

- **AWS Lightsail (Ubuntu 22.04, 1GB RAM)**: Recommended for cost-effective, always-on deployment
- **Systemd/tmux**: Used to keep agents, MCP servers, and UI running
- **Firewall**: Only necessary ports (7860, 7861, 80, 443) are open
- **Environment Variables**: Managed via `.env` file
- **Redis**: Installed and running as a service
- **Database**: SQLite for persistent storage, Redis for fast cache

---

## 9. Extensibility & Best Practices

- **Add new agents or strategies** by creating new agent classes and registering them in the workflow
- **Add new MCP tools** by implementing and registering new async tool functions
- **Plug in new data sources** (APIs, scrapers) with minimal code changes
- **All code is modular, testable, and follows async best practices**

---

## 10. Security & Reliability

- **API keys and secrets** are never hardcoded; always loaded from environment
- **Error handling** in all tools and agents
- **Rate limiting and caching** to avoid API bans and ensure fast response
- **Logging** for all errors and important events

---

## 11. Example Agent Notification

**Header:**
> Ray bought 20 shares of TCS
**Data:**
> Bought after technical breakout. Stop loss at 3400.

---

## 12. Future Enhancements

- Add webhooks or email notifications
- Support for more exchanges or asset classes
- Advanced analytics and reporting
- Multi-user support and authentication
- Dockerization for easier deployment

---

## 13. Authors & Credits

- **Darshan Ramani** (Solo Project Developer)
- OpenAI, Anthropic, Playwright, Redis, Gradio, ntfy.sh, and the open-source community

---

## 14. License

MIT License. See [LICENSE](LICENSE) for details.

---

# End of Plan
