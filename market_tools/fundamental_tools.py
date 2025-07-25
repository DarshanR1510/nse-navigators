from bs4 import BeautifulSoup
import requests
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import re
import subprocess
import json
import time
import warnings
import pytz

IST = pytz.timezone('Asia/Kolkata')

warnings.filterwarnings("ignore", category=FutureWarning, module="soupsieve.css_parser")
class ScreenerScraper:
    """
    Enhanced Screener.in scraper using requests + Playwright MCP fallback.
    Returns structured data optimized for LLM analysis and company comparison.
    """

    def __init__(self, use_playwright=True, playwright_mcp_cmd=None):
        self.base_url = "https://www.screener.in/company/{}/consolidated/"
        self.use_playwright = use_playwright
        self.session = requests.Session()
        self.playwright_mcp_cmd = playwright_mcp_cmd or ["node", "utils/playwright_mcp_server.js"]
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        self.request_timeout = 15
        self.playwright_timeout = 30


    def scrape_company(self, symbol: str, method: str = "auto") -> Dict[str, Any]:
        """Scrape company data with optimized structure for comparison"""
        start_time = time.time()
        
        try:
            if method == "auto":
                try:
                    data = self._scrape_with_requests(symbol)
                    if self._is_valid_data(data):
                        data['scrape_method'] = 'requests'
                        data['scrape_time_sec'] = round(time.time() - start_time, 2)
                        return data
                    print(f"Requests method failed for {symbol}, trying Playwright MCP...")
                except Exception as e:
                    print(f"Requests method error: {e}")
                
                data = self._scrape_with_playwright(symbol)
                data['scrape_method'] = 'playwright'
                data['scrape_time_sec'] = round(time.time() - start_time, 2)
                return data
            
            elif method == "requests":
                data = self._scrape_with_requests(symbol)
                data['scrape_method'] = 'requests'
                data['scrape_time_sec'] = round(time.time() - start_time, 2)
                return data
            
            elif method == "playwright":
                data = self._scrape_with_playwright(symbol)
                data['scrape_method'] = 'playwright'
                data['scrape_time_sec'] = round(time.time() - start_time, 2)
                return data
                
        except Exception as e:
            return {
                "symbol": symbol.upper(),
                "error": str(e),
                "scrape_time_sec": round(time.time() - start_time, 2)
            }


    def _scrape_with_requests(self, symbol: str) -> Dict[str, Any]:
        """Fast scraping using requests + BeautifulSoup"""
        url = self.base_url.format(symbol.upper())
        
        try:
            response = self.session.get(url, headers=self.headers, timeout=self.request_timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            return self._parse_all_data(soup, symbol)
            
        except Exception as e:
            raise Exception(f"Requests scraping failed: {str(e)}")


    def _scrape_with_playwright(self, symbol: str) -> Dict[str, Any]:
        """Robust scraping using Playwright MCP for dynamic content"""
        url = self.base_url.format(symbol.upper())
        
        try:
            html = self._fetch_html_with_playwright(url, wait_selector="#peers table.data-table")
            if not html:
                raise Exception("Playwright MCP did not return HTML.")
                
            soup = BeautifulSoup(html, 'html.parser')
            return self._parse_all_data(soup, symbol)
            
        except Exception as e:
            raise Exception(f"Playwright scraping failed: {str(e)}")


    def _fetch_html_with_playwright(self, url: str, wait_selector: str = None) -> Optional[str]:
        """Fetch HTML using Playwright MCP server"""
        try:
            cmd = self.playwright_mcp_cmd + [url]
            if wait_selector:
                cmd += ["--wait-for", wait_selector]
            proc = subprocess.run(
                cmd,
                capture_output=True,
                timeout=self.playwright_timeout,
                text=True
            )
            
            if proc.returncode == 0:
                return proc.stdout
            else:
                error_msg = proc.stderr if proc.stderr else "Unknown error"
                print(f"Playwright MCP error: {error_msg}")
                return None
                
        except subprocess.TimeoutExpired:
            print("Playwright MCP timeout expired")
            return None
        except Exception as e:
            print(f"Error calling Playwright MCP: {e}")
            return None


    def _parse_all_data(self, soup: BeautifulSoup, symbol: str) -> Dict[str, Any]:
        """Parse all relevant data from BeautifulSoup object"""
        # Extract basic company info
        company_name = self._extract_company_name(soup)
        current_price = self._extract_current_price(soup)
        market_cap = self._extract_market_cap(soup)
        
        # Extract financial statements
        balance_sheet = self._extract_balance_sheet(soup)
        profit_loss = self._extract_profit_loss(soup, quarterly=False)
        cash_flow = self._extract_cash_flow(soup)
        shareholding = self._extract_shareholding(soup, quarterly=False)
        
        # Extract additional metrics
        ratios = self._extract_ratios(soup)
        growth_metrics = self._extract_growth_metrics(soup)
        efficiency_metrics = self._extract_efficiency_metrics(soup)
        risk_metrics = self._extract_risk_metrics(balance_sheet, profit_loss)        
        growth_tables = self._extract_growth_tables(soup)
        # Always use Playwright for peer comparison
        
        try:
            html_peers = self._fetch_html_with_playwright(self.base_url.format(symbol.upper()), wait_selector="#peers table.data-table")
            if html_peers:
                soup_peers = BeautifulSoup(html_peers, 'html.parser')
                peers = self._extract_peer_comparison(soup_peers)
            else:
                peers = self._extract_peer_comparison(soup)  # fallback to requests soup if Playwright fails
        except Exception as e:
            print(f"Error extracting peers with Playwright: {e}")
            peers = self._extract_peer_comparison(soup)
        
        # Get latest year with most data
        latest_year = self._get_latest_year([
            balance_sheet.keys(),
            profit_loss.keys(),
            cash_flow.keys(),
            shareholding.keys()
        ])
        
        return {
            "symbol": symbol.upper(),
            "company_name": company_name,
            "current_price": current_price,
            "market_cap": market_cap,
            "latest_year": latest_year,
            "valuation_metrics": {
                **ratios,
                "market_cap_cr": market_cap,
                "price_to_book": round(ratios.get("book_value") and current_price / ratios["book_value"], 2),
                "enterprise_value": self._extract_enterprise_value(soup),
                "dividend_payout_ratio": self._extract_dividend_payout(soup)
            },
            "growth_metrics": {
                **growth_metrics,
                **growth_tables
            },
            "efficiency_metrics": efficiency_metrics,
            "risk_metrics": risk_metrics,        
            "financial_statements": {
                "balance_sheet": self._get_latest_data(balance_sheet, latest_year),
                "income_statement": self._get_latest_data(profit_loss, latest_year),
                "cash_flow": self._get_latest_data(cash_flow, latest_year),
                "shareholding_pattern": self._get_latest_data(shareholding, latest_year)
            },
            "historical_trends": {
                "balance_sheet": self._last_n_years_dict(balance_sheet, 5),
                "profit_loss": self._last_n_years_dict(profit_loss, 5),
                "cash_flow": self._last_n_years_dict(cash_flow, 5),
                "shareholding": self._last_n_years_dict(shareholding, 5)
            },
            "peer_comparison": peers,
            "scrape_timestamp": datetime.now(IST).isoformat()
        }


    def _extract_company_name(self, soup: BeautifulSoup) -> str:
        """Extract company name with multiple fallback selectors"""
        selectors = [
            'h1', '.company-name', '[data-test="company-name"]',
            '.company-title h1', '.company-header h1'
        ]
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                return element.get_text().strip()
        return "Unknown"


    def _extract_current_price(self, soup: BeautifulSoup) -> float:
        """Extract current stock price with multiple fallback selectors"""
        price_selectors = [
            '#top-ratios > li:nth-child(2) > span.nowrap span.number',
            '[data-test="current-price"]',
            '.company-info .number',
            '.current-price',
            '.price .number'
        ]
        
        for selector in price_selectors:
            element = soup.select_one(selector)
            if element:
                text = element.get_text().strip()
                match = re.search(r'₹?\s*([\d,]+\.?\d*)', text)
                if match:
                    return float(match.group(1).replace(',', ''))
        return 0.0


    def _extract_market_cap(self, soup: BeautifulSoup) -> float:
        """Extract market capitalization in Cr"""
        try:
            # Try specific selector first
            cap_element = soup.select_one('.company-ratios li:nth-child(1) span.number')
            if cap_element:
                text = cap_element.get_text().strip()
                match = re.search(r'([\d,]+\.?\d*)\s*Cr', text)
                if match:
                    return float(match.group(1).replace(',', ''))
            
            # Fallback to text search
            text = soup.get_text()
            match = re.search(r'Market Cap[:\s]+₹?\s*([\d,]+\.?\d*)\s*Cr', text, re.IGNORECASE)
            if match:
                return float(match.group(1).replace(',', ''))
            return 0.0
        except:
            return 0.0


    def _extract_enterprise_value(self, soup: BeautifulSoup) -> float:
        """Extract enterprise value in Cr"""
        try:
            # Primary selector
            ev_element = soup.select_one('#top-ratios li:contains("Enterprise Value") span.number')
            if ev_element:
                text = ev_element.get_text().strip()
                return self._clean_financial_value(text)
            
            # Secondary selector
            ev_element = soup.select_one('.company-ratios li:nth-child(3) span.number')
            if ev_element:
                return self._clean_financial_value(ev_element.get_text())
            
            # Text-based fallback
            text = soup.get_text()
            match = re.search(r'Enterprise Value:\s*₹?\s*([\d,]+\.?\d*)\s*Cr', text, re.IGNORECASE)

            if match:
                return float(match.group(1).replace(',', ''))
            return 0.0
        except:
            return 0.0


    def _extract_ratios(self, soup: BeautifulSoup) -> Dict[str, float]:
        """Enhanced ratio extraction with multiple fallback strategies"""
        ratios = {
            "pe_ratio": None,            
            "roe": None,
            "roce": None,
            "face_value": None,
            "book_value": None,
            "dividend_yield": None,                        
        }
        
        try:
            # Primary extraction from top ratios
            ul = soup.find("ul", id="top-ratios")
            if ul:
                for li in ul.find_all("li"):
                    label = li.find("span", class_="name")
                    value = li.find("span", class_="number")
                    if not label or not value:
                        continue
                        
                    label_text = label.get_text(strip=True).lower()
                    value_text = value.get_text(strip=True)
                    cleaned_value = self._clean_financial_value(value_text)

                    if "p/e" in label_text:
                        ratios["pe_ratio"] = cleaned_value                    
                    elif "roe" in label_text:
                        ratios["roe"] = cleaned_value
                    elif "roce" in label_text:
                        ratios["roce"] = cleaned_value
                    elif "face value" in label_text:
                        ratios["face_value"] = cleaned_value
                    elif "book value" in label_text:
                        ratios["book_value"] = round(cleaned_value, 2)
                    elif "dividend yield" in label_text:
                        ratios["dividend_yield"] = cleaned_value                    

            # Fallback 1: Check ratios section
            ratios_section = soup.find("section", id="ratios")
            if ratios_section:
                rows = ratios_section.find_all("tr")
                for row in rows:
                    cells = row.find_all("td")
                    if len(cells) >= 2:
                        metric = cells[0].get_text(strip=True).lower()
                        value = self._clean_financial_value(cells[1].get_text())
                        
                        if "price to earning" in metric and not ratios["pe_ratio"]:
                            ratios["pe_ratio"] = value
                        elif "return on equity" in metric and not ratios["roe"]:
                            ratios["roe"] = value
                        elif "return on capital employed" in metric and not ratios["roce"]:
                            ratios["roce"] = value

            # Fallback 2: Text search for critical ratios
            if not ratios["pe_ratio"]:
                pe_element = soup.find("td", string=re.compile("P/E Ratio", re.IGNORECASE))
                if pe_element and pe_element.find_next_sibling("td"):
                    ratios["pe_ratio"] = self._clean_financial_value(
                        pe_element.find_next_sibling("td").get_text()
                    )
                    
            if not ratios["book_value"]:
                bv_element = soup.find("td", string=re.compile("Book Value", re.IGNORECASE))
                if bv_element and bv_element.find_next_sibling("td"):
                    ratios["book_value"] = self._clean_financial_value(
                        bv_element.find_next_sibling("td").get_text()
                    )

            return ratios
            
        except Exception as e:
            print(f"Error extracting ratios: {e}")
            return ratios


    def _extract_growth_metrics(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Enhanced growth metrics extraction from profit/loss table"""
        
        growth = {
            "sales_growth_3yr": None,
            "profit_growth_3yr": None,
            "quarterly_growth": {
                "sales_qoq": None,
                "sales_yoy": None,
                "profit_qoq": None,
                "profit_yoy": None
            },
            "annual_trends": {
                "sales": {},
                "net_profit": {},
                "operating_profit": {},
                "eps": {}
            }
        }
        
        try:
            # Extract from profit/loss table
            pl_table = soup.select_one("#profit-loss table.data-table")
            if pl_table:
                # Get all years from header
                header_row = pl_table.select_one("thead tr")
                years = [th.get_text(strip=True) for th in header_row.find_all("th")[1:]]
                
                # Process each data row
                for row in pl_table.select("tbody tr"):
                    row_label = row.get_text(separator=" ", strip=True).lower()
                    cells = row.find_all("td")
                    
                    # Skip rows without data
                    if len(cells) < 2:
                        continue
                    
                    # Extract sales data
                    if "sales" in row_label and not "opm" in row_label:
                        for i, year in enumerate(years):
                            growth["annual_trends"]["sales"][year] = self._clean_financial_value(cells[i+1].get_text())
                        
                        # Calculate YoY growth if we have at least 2 years
                        if len(years) >= 2:
                            current = growth["annual_trends"]["sales"].get(years[-1], 0)
                            previous = growth["annual_trends"]["sales"].get(years[-2], 0)
                            if previous != 0:
                                growth["quarterly_growth"]["sales_yoy"] = round(((current - previous) / previous) * 100, 2)

                    # Extract net profit data
                    elif "net profit" in row_label:
                        for i, year in enumerate(years):
                            growth["annual_trends"]["net_profit"][year] = self._clean_financial_value(cells[i+1].get_text())
                        
                        # Calculate YoY growth if we have at least 2 years
                        if len(years) >= 2:
                            current = growth["annual_trends"]["net_profit"].get(years[-1], 0)
                            previous = growth["annual_trends"]["net_profit"].get(years[-2], 0)
                            if previous != 0:
                                growth["quarterly_growth"]["profit_yoy"] = round(((current - previous) / previous) * 100, 2)
                    
                    # Extract operating profit
                    elif "operating profit" in row_label:
                        for i, year in enumerate(years):
                            growth["annual_trends"]["operating_profit"][year] = self._clean_financial_value(cells[i+1].get_text())
                    
                    # Extract EPS data
                    elif "eps in rs" in row_label:
                        for i, year in enumerate(years):
                            growth["annual_trends"]["eps"][year] = self._clean_financial_value(cells[i+1].get_text())
                
                # Calculate 3-year growth rates if we have enough data
                if len(years) >= 4:
                    # Sales growth
                    current_sales = growth["annual_trends"]["sales"].get(years[-1], 0)
                    past_sales = growth["annual_trends"]["sales"].get(years[-4], 0)
                    if past_sales != 0:
                        growth["sales_growth_3yr"] = round(((current_sales / past_sales) ** (1/3) - 1) * 100, 2)

                    # Profit growth
                    current_profit = growth["annual_trends"]["net_profit"].get(years[-1], 0)
                    past_profit = growth["annual_trends"]["net_profit"].get(years[-4], 0)
                    if past_profit != 0:
                        growth["profit_growth_3yr"] = round(((current_profit / past_profit) ** (1/3) - 1) * 100, 2)


            # 2. Extract from quarterly table for QoQ calculations
            qtr_table = soup.select_one("#quarters table.data-table")
            if qtr_table:
                # Get all quarters from header
                header_row = qtr_table.select_one("thead tr")
                quarters = [th.get_text(strip=True) for th in header_row.find_all("th")[1:]]
                
                # We need at least 2 quarters for QoQ comparison
                if len(quarters) >= 2:
                    # Process each data row
                    for row in qtr_table.select("tbody tr"):
                        row_label = row.get_text(separator=" ", strip=True).lower()
                        cells = row.find_all("td")
                        
                        if len(cells) < 2:
                            continue
                        
                        # Extract sales data
                        if "sales" in row_label and not "opm" in row_label:
                            current_qtr = self._clean_financial_value(cells[-1].get_text())
                            prev_qtr = self._clean_financial_value(cells[-2].get_text())                            
                            
                            if prev_qtr != 0:
                                growth["quarterly_growth"]["sales_qoq"] = round(((current_qtr - prev_qtr) / prev_qtr) * 100, 2)
                        
                        # Extract net profit data
                        elif "net profit" in row_label:
                            current_qtr = self._clean_financial_value(cells[-1].get_text())
                            prev_qtr = self._clean_financial_value(cells[-2].get_text())
                            
                            if prev_qtr != 0:
                                growth["quarterly_growth"]["profit_qoq"] = round(((current_qtr - prev_qtr) / prev_qtr) * 100, 2)
            
            return growth
            
        except Exception as e:
            print(f"Error extracting growth metrics: {e}")
            return growth
        

    def _extract_growth_tables(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract growth metrics from the ranges tables"""
        growth_tables = {
            "compounded_sales_growth": {},
            "compounded_profit_growth": {},
            "stock_price_cagr": {},
            "return_on_equity": {}
        }
        
        try:
            # Find all range tables
            tables = soup.select("table.ranges-table")
            
            for table in tables:
                # Get the table header to identify which metrics we're extracting
                header = table.select_one("th").get_text(strip=True).lower()
                
                # Extract all rows
                rows = table.select("tr:has(td)")
                for row in rows:
                    cells = row.find_all("td")
                    if len(cells) == 2:
                        period = cells[0].get_text(strip=True).lower().replace(":", "")
                        value = self._clean_financial_value(cells[1].get_text())
                        
                        # Map to the appropriate dictionary based on header
                        if "sales" in header:
                            growth_tables["compounded_sales_growth"][period] = value
                        elif "profit" in header:
                            growth_tables["compounded_profit_growth"][period] = value
                        elif "stock price" in header:
                            growth_tables["stock_price_cagr"][period] = value
                        elif "return on equity" in header:
                            growth_tables["return_on_equity"][period] = value
            
            return growth_tables
            
        except Exception as e:
            print(f"Error extracting growth tables: {e}")
            return growth_tables


    def _extract_efficiency_metrics(self, soup: BeautifulSoup) -> Dict[str, Any]:        
        efficiency = {
            "debtor_days": None,
            "inventory_days": None,
            "payable_days": None,
            "cash_conversation_cycle": None,
            "working_capital_days": None,
            "ROCE%" : None
        }
        
        try:
            
            # 1: Check ratios section
            ratios_section = soup.find("section", id="ratios")
            if ratios_section:
                rows = ratios_section.find_all("tr")
                for row in rows:
                    cells = row.find_all("td")
                    if len(cells) >= 2:
                        metric = cells[0].get_text().strip().lower()
                        value = self._clean_financial_value(cells[1].get_text())

                        if "debtor days" in metric:
                            efficiency["debtor_days"] = value
                        elif "inventory days" in metric:
                            efficiency["inventory_days"] = value
                        elif "days payable" in metric:
                            efficiency["payable_days"] = value
                        elif "cash conversion cycle" in metric:
                            efficiency["cash_conversation_cycle"] = value
                        elif "working capital days" in metric:
                            efficiency["working_capital_days"] = value
                        elif "roce" in metric:
                            efficiency["ROCE%"] = value
                
            return efficiency
            
        except Exception as e:
            print(f"Error extracting efficiency metrics: {e}")
            return efficiency


    def _extract_risk_metrics(self, balance_sheet: Dict[str, Any], profit_loss: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced risk metrics extraction with comprehensive fallbacks"""
        risk = {
            "debt_to_equity": None,
            "interest_coverage": None
        }
        
        try:
            # Calculate debt_to_equity from balance_sheet
            if balance_sheet:
                latest_year = next(iter(balance_sheet))
                latest_year_data = balance_sheet[latest_year]
                total_debt = latest_year_data.get("equity_borrowings", 0.0)
                total_equity = latest_year_data.get("equity_equity_capital", 0.0) + latest_year_data.get("equity_reserves", 0.0)
                
                risk["debt_to_equity"] = round(total_debt / total_equity, 2) if total_equity else 0.0

            
            # Calculate interest_coverage from profit_loss
            if profit_loss:
                latest_year = next(iter(profit_loss))
                latest_year_data = profit_loss[latest_year]
                operating_profit = latest_year_data.get("operating_profit", 0.0)
                interest_expense = latest_year_data.get("interest_expense", 0.0)
                
                if interest_expense > 0:
                    risk["interest_coverage"] = round(operating_profit / interest_expense, 2)
                else:
                    risk["interest_coverage"] = None

        except Exception as e:
            print(f"Error extracting risk metrics: {e}")
            return risk


    def _extract_dividend_payout(self, soup: BeautifulSoup) -> float:
        """Extract dividend payout ratio"""
        try:
            payout_element = soup.find("td", string=re.compile("Dividend Payout", re.IGNORECASE))
            if payout_element and payout_element.find_next_sibling("td"):
                return self._clean_financial_value(
                    payout_element.find_next_sibling("td").get_text()
                )
            return 0.0
        except:
            return 0.0


    def _extract_balance_sheet(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract balance sheet data by year"""
        balance_sheet = {}
        
        try:
            table = soup.select_one("#balance-sheet table.data-table")
            if not table:
                return balance_sheet
                
            rows = table.find_all('tr')
            if len(rows) < 2:
                return balance_sheet

            # Get years from header
            years = self._extract_years_from_header(rows[0])
            for year in years:
                balance_sheet[year] = {}

            current_section = None

            for row in rows[1:]:
                cells = row.find_all(['td', 'th'])
                if len(cells) < 2:
                    continue

                row_label = cells[0].get_text().strip().lower()
                
                # Identify sections
                if any(keyword in row_label for keyword in ['assets', 'current assets', 'fixed assets']):
                    current_section = 'assets'
                elif any(keyword in row_label for keyword in ['liabilities', 'current liabilities']):
                    current_section = 'liabilities'
                elif any(keyword in row_label for keyword in ['equity', 'shareholders']):
                    current_section = 'equity'

                # Extract values
                if current_section:
                    field_name = self._normalize_field_name(row_label)
                    for i, cell in enumerate(cells[1:]):
                        if i < len(years):
                            value = self._clean_financial_value(cell.get_text().strip())
                            balance_sheet[years[i]][f"{current_section}_{field_name}"] = value

            # Sort years descending
            return dict(sorted(balance_sheet.items(), key=lambda x: self._extract_year_from_string(x[0]), reverse=True))            
            
        except Exception as e:
            print(f"Error extracting balance sheet: {e}")
            return balance_sheet


    def _extract_profit_loss(self, soup: BeautifulSoup, quarterly: bool) -> Dict[str, Any]:
        """Extract profit & loss statement data by year"""
        profit_loss = {}
        
        try:
            if quarterly:
                table = soup.select_one("#quarters table.data-table")
            else:
                table = soup.select_one("#profit-loss table.data-table")

            if not table:
                return profit_loss
                
            rows = table.find_all('tr')
            if len(rows) < 2:
                return profit_loss

            # Get years from header
            years = self._extract_years_from_header(rows[0])
            
            # Initialize result dict for each year
            for year in years:
                profit_loss[year] = {}

            # Map row labels to our keys
            row_map = {
                'operating profit': 'operating_profit',
                'profit before tax': 'profit_before_tax',
                'net profit': 'net_profit',
                'total revenue': 'total_revenue',
                'total expenses': 'total_expenses',
                'eps': 'earnings_per_share',
                'sales': 'sales',
                'other income': 'other_income'
            }

            for row in rows[1:]:
                cells = row.find_all(['td', 'th'])
                if len(cells) < 2:
                    continue
                    
                first_cell = cells[0]
                button = first_cell.find("button")
                if button:
                    row_label = button.get_text(separator=" ").strip().lower()
                else:
                    row_label = first_cell.get_text().strip().lower()
                    
                key = None
                for pattern in row_map:
                    if pattern in row_label:
                        key = row_map[pattern]
                        break
                        
                if not key:
                    continue
                    
                values = [self._clean_financial_value(cell.get_text().strip()) for cell in cells[1:]]
                for i, year in enumerate(years):
                    if i < len(values):
                        profit_loss[year][key] = values[i]

            # Sort years descending
            return dict(sorted(profit_loss.items(), key=lambda x: self._extract_year_from_string(x[0]), reverse=True))
            
        except Exception as e:
            print(f"Error extracting profit & loss: {e}")
            return profit_loss


    def _extract_cash_flow(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract cash flow statement data by year"""
        cash_flow = {}
        
        try:
            table = soup.select_one("#cash-flow table.data-table")
            if not table:
                return cash_flow
                
            rows = table.find_all('tr')
            if len(rows) < 2:
                return cash_flow

            # Get years from header
            years = self._extract_years_from_header(rows[0])
            for year in years:
                cash_flow[year] = {}

            current_section = None

            for row in rows[1:]:
                cells = row.find_all(['td', 'th'])
                if len(cells) < 2:
                    continue

                row_label = cells[0].get_text().strip().lower()
                
                # Identify cash flow sections
                if 'operating' in row_label:
                    current_section = 'operating'
                elif 'investing' in row_label:
                    current_section = 'investing'
                elif 'financing' in row_label:
                    current_section = 'financing'
                elif 'net cash' in row_label:
                    current_section = 'net'

                # Extract values
                if current_section:
                    field_name = self._normalize_field_name(row_label)
                    for i, cell in enumerate(cells[1:]):
                        if i < len(years):
                            value = self._clean_financial_value(cell.get_text().strip())
                            cash_flow[years[i]][f"{current_section}_{field_name}"] = value

            # Sort years descending
            return dict(sorted(cash_flow.items(), key=lambda x: self._extract_year_from_string(x[0]), reverse=True))
            
        except Exception as e:
            print(f"Error extracting cash flow: {e}")
            return cash_flow


    def _extract_shareholding(self, soup: BeautifulSoup, quarterly: bool) -> Dict[str, Any]:
        """Extract shareholding pattern data by year"""
        shareholding = {}
        
        try:
            if quarterly:
                table = soup.select_one("#shareholding #quarterly-shp table.data-table")
            else:
                table = soup.select_one("#shareholding #yearly-shp table.data-table")

            if not table:
                return shareholding
                
            rows = table.find_all('tr')
            if len(rows) < 2:
                return shareholding

            # Get years from header
            years = self._extract_years_from_header(rows[0])
            
            # Initialize result dict for each year
            for year in years:
                shareholding[year] = {}

            # Map row labels to our keys
            row_map = {
                'promoters': 'promoter_holding',
                'foreign_institutions': 'fii_holding',
                'domestic_institutions': 'dii_holding',
                'government': 'government_holding',
                'public': 'public_holding',
                'others': 'other_holding'
            }

            for row in rows[1:]:
                cells = row.find_all(['td', 'th'])
                if len(cells) < 2:
                    continue
                    
                first_cell = cells[0]
                button = first_cell.find("button")
                if not button:
                    continue

                onclick = button.get("onclick", "")
                match = re.search(r"showShareholders\('([^']+)'", onclick)
                if match:
                    group = match.group(1).lower()
                else:
                    group = button.get_text(separator=" ").strip().lower()
                    
                key = row_map.get(group)
                if not key:
                    continue
                    
                values = [self._clean_financial_value(cell.get_text().strip()) for cell in cells[1:]]
                for i, year in enumerate(years):
                    if i < len(values):
                        shareholding[year][key] = values[i]

            # Sort years descending
            return dict(sorted(shareholding.items(), key=lambda x: self._extract_year_from_string(x[0]), reverse=True))
            
        except Exception as e:
            print(f"Error extracting shareholding: {e}")
            return shareholding


    def _extract_peer_comparison(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract peer comparison data"""
        peers = []
        median = None
        
        try:
            table = soup.select_one("#peers table.data-table")
            if not table:
                return {"peers": peers, "median": median}

            rows = table.find_all('tr')
            if len(rows) < 2:
                return {"peers": peers, "median": median}

            # Get headers and map to our required fields
            header_row = rows[0]
            headers = []
            for th in header_row.find_all(['th', 'td']):
                header_text = th.get_text(separator=" ").strip().lower()
                headers.append(header_text)

            # Map headers to our required keys
            header_map = {
                'name': 'name',
                'cmp': 'current_price',
                'p/e': 'pe_ratio',
                'market cap': 'market_cap',
                'div yield': 'dividend_yield',
                'np qtr': 'net_profit_qtr',
                'qtr profit var': 'qtr_profit_var',
                'sales qtr': 'sales_qtr',
                'qtr sales var': 'qtr_sales_var',
                'roce': 'roce'
            }

            # Find the index of each required column
            col_indices = {}
            for idx, h in enumerate(headers):
                for key, val in header_map.items():
                    if h.startswith(key):
                        col_indices[val] = idx

            for row in rows[1:]:
                cells = row.find_all(['td', 'th'])
                if len(cells) < len(col_indices):
                    continue
                    
                peer = {}
                for key, idx in col_indices.items():
                    value = cells[idx].get_text(separator=" ").strip()
                    # Clean numeric values
                    if key in ['current_price', 'pe_ratio', 'market_cap', 'dividend_yield', 
                             'net_profit_qtr', 'qtr_profit_var', 'sales_qtr', 'qtr_sales_var', 
                             'roce', 'rank']:
                        value = self._clean_financial_value(value)
                    # For name, extract text from <a> if present
                    if key == 'name':
                        a = cells[idx].find('a')
                        value = a.get_text(strip=True) if a else value
                    peer[key] = value
                    
                peers.append(peer)

            # Check if last row is median
            if peers and isinstance(peers[-1].get("name", ""), str) and "median" in peers[-1]["name"].lower():
                median = peers.pop(-1)
                median.pop("rank", None)

            return {"peers": peers, "median": median}
            
        except Exception as e:
            print(f"Error extracting peer comparison: {e}")
            return {"peers": peers, "median": None}


    def _extract_years_from_header(self, header_row) -> List[str]:
        """Extract years from table header row"""
        years = []
        for th in header_row.find_all(['th', 'td'])[1:]:
            year_text = th.get_text().strip()
            if year_text and re.search(r'\d{4}', year_text):
                years.append(year_text)
        return years


    def _get_latest_year(self, year_lists: List) -> Optional[str]:
        """Get the latest year from multiple lists of years"""
        all_years = set()
        for year_list in year_lists:
            all_years.update(year_list)
            
        if not all_years:
            return None
            
        return max(all_years, key=lambda y: self._extract_year_from_string(y))


    def _get_latest_data(self, data: Dict[str, Any], latest_year: str) -> Dict[str, Any]:
        """Get data for latest year with fallback to most recent available"""
        if not data:
            return {}
            
        if latest_year in data:
            return data[latest_year]
            
        # Fallback to most recent year if exact match not found
        sorted_years = sorted(data.keys(), key=lambda y: self._extract_year_from_string(y), reverse=True)
        return data[sorted_years[0]] if sorted_years else {}


    def _clean_financial_value(self, value_text: str) -> float:
        """Clean and convert financial value text to float"""
        if not value_text or value_text.strip() == '-':
            return 0.0
            
        try:
            # Remove currency symbols, commas, and percentage signs
            cleaned = re.sub(r'[₹,\s%()\-]', '', value_text)

            # Handle negative values
            is_negative = False
            if '(' in value_text and ')' in value_text or '-' in value_text[0]:
                is_negative = True
            
            # Convert based on suffixes
            multiplier = 1
            if 'cr' in value_text.lower() or 'crore' in value_text.lower() or 'crs' in value_text.lower():
                multiplier = 10000000  # 1 crore = 10 million
            elif 'l' in value_text.lower() or 'lakh' in value_text.lower():
                multiplier = 100000    # 1 lakh = 100 thousand
            elif 'k' in value_text.lower():
                multiplier = 1000            
            
            # Extract numeric value
            numeric_match = re.search(r'([\d.]+)', cleaned)
            if numeric_match:
                value = float(numeric_match.group(1)) * multiplier
                return -value if is_negative else value
                
            return 0.0
            
        except (ValueError, AttributeError):
            return 0.0


    def _normalize_field_name(self, text: str) -> str:
        """Normalize field names for consistency"""
        if not text:
            return "unknown_field"
            
        # Remove special characters and normalize
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        text = re.sub(r'\s+', '_', text.strip())
        return text


    def _extract_year_from_string(self, year_str: str) -> int:
        """Extract year from string for sorting"""
        match = re.search(r'\d{4}', year_str)
        return int(match.group()) if match else 0


    def _is_valid_data(self, data: Dict[str, Any]) -> bool:
        """Check if scraped data is valid and complete"""
        return (
            data.get('company_name', 'Unknown') != 'Unknown' and
            data.get('current_price', 0) > 0 and
            data.get('market_cap', 0) > 0
        )


    def _last_n_years_dict(self, data: dict, n: int = 5) -> dict:
        """Return a dict with only the last n years (sorted descending)."""
        if not data:
            return {}
        years = sorted(data.keys(), key=lambda y: int(re.search(r'\d{4}', y).group()), reverse=True)
        selected_years = years[:n]
        return {year: data[year] for year in selected_years}


    def close(self):
        """Clean up resources"""
        self.session.close()



def get_company_financials(symbol: str) -> Dict[str, Any]:
    """
    Get structured financial data for a company optimized for LLM analysis.
    Returns data in a format that enables easy comparison between companies.
    """
    scraper = ScreenerScraper(use_playwright=True)
    try:
        return scraper.scrape_company(symbol, method="auto")
    finally:
        scraper.close()
        

        
def save_financials_to_json(symbol: str):
    """
    Scrape financials for the given symbol and save to a JSON file named '{symbol}_fin.json'.
    """
    data = get_company_financials(symbol)
    filename = f"{symbol}_fin.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved financials for {symbol} to {filename}")

# Example usage:
# save_financials_to_json("FEDERALBNK")