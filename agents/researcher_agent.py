from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
from dotenv import load_dotenv
import requests
import os
import time

load_dotenv(override=True)

class ResearcherAgent:
    def __init__(self, name):
        self.logger = logging.getLogger(__name__)
        self.name = name
        # You can add more API keys or configs as needed

    def generate_search_queries(self) -> List[str]:
        """
        Dynamically generate search queries based on research instructions.
        You can make this more advanced by using market context, sector focus, etc.
        """
        # Example: Focus on trending sectors and small caps
        queries = [
            "latest EV sector news India",
            "specialty chemicals companies news India",
            "niche pharma stocks news India",
            "small cap stock news India",
            "block deals NSE news",
            "FII DII flows India"
        ]
        return queries

    def serper_search(self, query: str) -> List[Dict[str, Any]]:
        """
        Search news using Serper API for a given query.
        Returns a list of dicts with company/news info.
        """        
        headers = {"X-API-KEY": self.serper_api_key, "Content-Type": "application/json"}
        payload = {"q": query}
        try:
            response = requests.post(self.serper_url, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            results = []
            for item in data.get("news", []):
                # You may want to parse company name/symbol from title or snippet using NLP or regex
                results.append({
                    "headline": item.get("title"),
                    "snippet": item.get("snippet"),
                    "source": item.get("source"),
                    "link": item.get("link"),
                    "published_date": item.get("date")
                })
            return results
        except Exception as e:
            self.logger.error(f"Serper API error: {e}")
            return []
        
    def discover_candidates(self) -> List[Dict[str, Any]]:
        """
        Main entry: generate queries, run serper_search, and extract companies.
        """
        queries = self.generate_search_queries()
        all_results = []
        for q in queries:
            news_items = self.serper_search(q)
            # You can add logic here to extract company names/symbols from news_items
            all_results.extend(news_items)
        # Further process all_results to extract candidate companies if needed
        return all_results

    def fetch_financial_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch financial data for a symbol (mock implementation)."""
        self.logger.info(f"Fetching financial data for {symbol}")
        # Example: requests.get("https://fetch-api-url", params={"symbol": symbol, "api_key": self.fetch_api_key})
        return {
            "ROE": 18.5,
            "Debt/Equity": 0.2,
            "Growth_CAGR": 22.0,
            "Market_Cap": 3200,
            "Sector": "EV"
        }

    def brave_search(self, query: str) -> List[Dict[str, Any]]:
        """Search using Brave API (mock implementation, with rate limit)."""
        self.logger.info(f"Brave-search: {query}")
        time.sleep(1)  # Enforce 1s gap
        return [
            {"company_name": "XYZ Pharma", "company_symbol": "XYZ", "headline": "Niche pharma expansion", "source": "ET"}
        ]

    def filter_candidates(self, raw_candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply filters based on market cap, sector, and quality metrics."""
        filtered = []
        for c in raw_candidates:
            financials = self.fetch_financial_data(c["company_symbol"])
            if (
                financials["Market_Cap"] < 5000 and
                financials["ROE"] > 15 and
                financials["Debt/Equity"] < 0.5 and
                financials["Growth_CAGR"] > 20
            ):
                filtered.append({
                    "company_name": c["company_name"],
                    "company_symbol": c["company_symbol"],
                    "reason": f"{c['headline']} | Sector: {financials['Sector']}",
                    "conviction_score": self.score_conviction(financials),
                    "time_horizon": "medium-term",
                    "risk_factors": "Regulatory risk, market volatility"
                })
        return filtered

    def score_conviction(self, financials: Dict[str, Any]) -> int:
        """Score conviction based on financial metrics."""
        score = 5
        if financials["ROE"] > 20:
            score += 2
        if financials["Growth_CAGR"] > 25:
            score += 2
        if financials["Debt/Equity"] < 0.2:
            score += 1
        return min(score, 10)

    def get_candidates(self) -> List[Dict[str, Any]]:
        """
        Main workflow: scan news, apply filters, and return candidate companies.
        """
        self.logger.info("Starting candidate discovery workflow...")
        # 1. Initial Screening (Serper)
        raw_candidates = self.serper_search("latest trending stocks India")
        # 2. Fallback: Fetch if Serper fails
        if not raw_candidates:
            raw_candidates = [self.fetch_financial_data("ABC")]
        # 3. Validation: Brave search (max 4 times, 1s gap)
        brave_results = []
        for i, c in enumerate(raw_candidates[:4]):
            brave_results += self.brave_search(f"{c['company_name']} news")
        # 4. Combine and filter
        all_candidates = raw_candidates + brave_results
        filtered = self.filter_candidates(all_candidates)
        # 5. Limit to 4-6 companies
        final_candidates = filtered[:6]
        self.logger.info(f"Researcher found {len(final_candidates)} candidates.")
        return final_candidates

    def get_research_summary(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format output as per template."""
        return {
            "RESEARCH_SUMMARY": f"Market overview: {len(candidates)} actionable opportunities found. Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "SYMBOL_LIST": candidates
        }

    def run(self) -> Dict[str, Any]:
        """Run full research workflow and return structured output."""
        candidates = self.get_candidates()
        return self.get_research_summary(candidates)

    # Optionally, add async versions for integration with async workflows


# Example usage:
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agent = ResearcherAgent()
    result = agent.run()
    print(result)