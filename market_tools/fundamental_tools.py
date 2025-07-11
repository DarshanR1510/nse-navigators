from bs4 import BeautifulSoup
import requests
from datetime import datetime
from typing import Dict, Any, Optional, List
import re
import subprocess
import json


class ScreenerScraper:
    """
    Screener.in scraper using requests and Playwright MCP fallback
    """

    def __init__(self, use_playwright=True, playwright_mcp_cmd=None):
        self.base_url = "https://www.screener.in/company/{}/consolidated/"
        self.use_playwright = use_playwright
        self.session = requests.Session()
        self.playwright_mcp_cmd = playwright_mcp_cmd or ["node", "playwright_mcp_server.js"]
        # Set up headers to mimic real browser
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }

    def scrape_company(self, symbol: str, method: str = "auto") -> Dict[str, Any]:
        """
        Scrape company data using requests, fallback to Playwright MCP if needed.
        
        Args:
            symbol: Company symbol (e.g., 'RELIANCE')
            method: 'requests', 'playwright', or 'auto' (tries requests first, falls back to playwright)
        """
        
        if method == "auto":
            try:
                data = self._scrape_with_requests(symbol)
                if self._is_valid_data(data):
                    return data
                print(f"Requests method failed for {symbol}, trying Playwright MCP...")
            except Exception as e:
                print(f"Requests method error: {e}")
            # Fallback to Playwright MCP
            return self._scrape_with_playwright(symbol)
        
        elif method == "requests":
            return self._scrape_with_requests(symbol)
        
        elif method == "playwright":
            return self._scrape_with_playwright(symbol)

    def _scrape_with_requests(self, symbol: str) -> Dict[str, Any]:
        """Fast scraping using requests + BeautifulSoup"""
        url = self.base_url.format(symbol.upper())
        
        try:
            response = self.session.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            return self._parse_soup_data(soup, symbol)
            
        except Exception as e:
            raise Exception(f"Requests scraping failed: {str(e)}")
    
    def _scrape_with_playwright(self, symbol: str) -> Dict[str, Any]:
        """Robust scraping using Playwright MCP for dynamic content"""
        url = self.base_url.format(symbol.upper())
        print(f"Using Playwright MCP to scrape for symbol {symbol}")
        try:
            html = self._fetch_html_with_playwright(url)
            if not html:
                raise Exception("Playwright MCP did not return HTML.")
            soup = BeautifulSoup(html, 'html.parser')
            return self._parse_soup_data(soup, symbol)
        except Exception as e:
            raise Exception(f"Playwright scraping failed: {str(e)}")

    def _fetch_html_with_playwright(self, url: str) -> Optional[str]:
        """
        Calls Playwright MCP server (or a node script) to fetch HTML for the given URL.
        Expects the script to print the HTML to stdout.
        """
        try:
            proc = subprocess.run(["node", "playwright_mcp_server.js", url], capture_output=True, timeout=40)
            if proc.returncode == 0:
                return proc.stdout.decode(errors="ignore")
            else:
                print(f"Playwright MCP error: {proc.stderr.decode(errors='ignore')}")
                return None
        except Exception as e:
            print(f"Error calling Playwright MCP: {e}")
            return None

    def _parse_soup_data(self, soup: BeautifulSoup, symbol: str) -> Dict[str, Any]:
        """Parse BeautifulSoup object to extract financial data"""
        
        data = {
            "symbol": symbol.upper(),
            "company_name": self._extract_company_name(soup),
            "current_price": self._extract_current_price(soup),
            "market_cap": self._extract_market_cap(soup),
            "financial_ratios": self._extract_ratios(soup),
            "balance_sheet": self._extract_balance_sheet(soup),
            # "quarterly_profit_loss": self._extract_profit_loss(soup, quarterly=True),
            "yearly_profit_loss": self._extract_profit_loss(soup, quarterly=False),
            "cash_flow": self._extract_cash_flow(soup),
            # "quarterly_shareholding": self._extract_shareholding(soup, quarterly=True),
            "yearly_shareholding": self._extract_shareholding(soup, quarterly=False),
            # "peers": self._extract_peer_comparison(soup),
            "scrape_timestamp": datetime.now().isoformat()            
        }
        
        return data
    
    def _extract_company_name(self, soup: BeautifulSoup) -> str:
        """Extract company name"""
        try:
            # Multiple selectors to try
            selectors = ['h1', '.company-name', '[data-test="company-name"]']
            for selector in selectors:
                element = soup.select_one(selector)
                if element:
                    return element.get_text().strip()
            return "Unknown"
        except:
            return "Unknown"
    
    def _extract_current_price(self, soup: BeautifulSoup) -> float:
        """Extract current stock price"""
        try:
            element = soup.select_one("#top-ratios > li:nth-child(2) > span.nowrap span.number")
            if element:
                text = element.get_text().strip()
                match = re.search(r'([\d,]+\.?\d*)', text)
                if match:
                    return float(match.group(1).replace(',', ''))
            # Fallback to previous selectors if needed
            price_selectors = [
                    '.number',
                    '[data-test="current-price"]',
                    '.company-info .number'
                ]
            
            for selector in price_selectors:
                elements = soup.select(selector)
                for element in elements:
                    text = element.get_text().strip()
                    # Match price pattern (₹ followed by numbers)
                    match = re.search(r'₹?\s*([\d,]+\.?\d*)', text)
                    if match:
                        return float(match.group(1).replace(',', ''))
            return 0.0
        except:
            return 0.0
    
    def _extract_market_cap(self, soup: BeautifulSoup) -> float:
        """Extract market capitalization"""
        try:
            # Look for market cap in company info section
            text = soup.get_text()
            match = re.search(r'Market Cap[:\s]+₹?\s*([\d,]+\.?\d*)\s*Cr', text, re.IGNORECASE)
            if match:
                return float(match.group(1).replace(',', ''))
            return 0.0
        except:
            return 0.0

    def _extract_ratios(self, soup: BeautifulSoup) -> dict:
        """
        Extracts pe, roe, roce, face_value, book_value, and dividend_yield for given stock.
        """
        ratios = {
            "stock p/e": None,
            "roe": None,
            "roce": None,
            "face_value": None,
            "book_value": None,
            "dividend_yield": None
        }
        ul = soup.find("ul", id="top-ratios")        
        if not ul:
            return ratios

        for li in ul.find_all("li"):
            label = li.find("span", class_="name")
            value = li.find("span", class_="number")
            if not label or not value:
                continue
            label_text = label.get_text(strip=True).lower()
            value_text = value.get_text(strip=True).replace(',', '')
            cleaned_value = self._clean_financial_value(value_text)

            if "p/e" in label_text or "stock p/e" in label_text:
                ratios["stock p/e"] = cleaned_value
            elif "roe" in label_text:
                ratios["roe"] = cleaned_value
            elif "roce" in label_text:
                ratios["roce"] = cleaned_value
            elif "face value" in label_text:
                ratios["face_value"] = cleaned_value
            elif "book value" in label_text:
                ratios["book_value"] = cleaned_value
            elif "dividend yield" in label_text:
                ratios["dividend_yield"] = cleaned_value

        return ratios
    
    def _extract_balance_sheet(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Get balance sheet data for given stock"""
        balance_sheet = {}
        try:
            
            table = soup.select_one("#balance-sheet table.data-table")
            if not table:
                return balance_sheet
            rows = table.find_all('tr')
            
            if len(rows) < 2:
                return balance_sheet

            # Get years from header
            header_row = rows[0]
            years = []
            for th in header_row.find_all(['th', 'td'])[1:]:
                year_text = th.get_text().strip()
                if year_text and re.search(r'\d{4}', year_text):
                    years.append(year_text)

            # Initialize dict for each year
            for year in years:
                if year not in balance_sheet:
                    balance_sheet[year] = {}

            current_section = None

            for row in rows[1:]:
                cells = row.find_all(['td', 'th'])
                if len(cells) < 2:
                    continue

                row_label = cells[0].get_text().strip().lower()

                # Identify sections (optional: you can add section info as prefix if you want)
                if any(keyword in row_label for keyword in ['assets', 'current assets', 'fixed assets']):
                    current_section = 'assets'
                elif any(keyword in row_label for keyword in ['liabilities', 'current liabilities']):
                    current_section = 'liabilities'
                elif any(keyword in row_label for keyword in ['equity', 'shareholders']):
                    current_section = 'equity'

                # Extract values for each year
                if current_section:
                    field_name = self._normalize_field_name(row_label)
                    for i, cell in enumerate(cells[1:]):
                        if i < len(years):
                            value_text = cell.get_text().strip()
                            value = self._clean_financial_value(value_text)
                            # Optionally, prefix field name with section: f"{current_section}_{field_name}"
                            balance_sheet[years[i]][field_name] = value
            

            # Optionally, sort years descending
            balance_sheet = dict(sorted(balance_sheet.items(), key=lambda x: int(re.search(r'\d{4}', x[0]).group()), reverse=True))
            return balance_sheet

        except Exception as e:
            print(f"Error extracting balance sheet: {e}")
            return balance_sheet

    def _extract_profit_loss(self, soup: BeautifulSoup, quarterly: bool) -> Dict[str, Any]:
        """Extract profit & loss statement data as a dict: {year: {operating_profit, profit_before_tax, net_profit}}"""
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
            header_row = rows[0]
            years = [th.get_text().strip() for th in header_row.find_all(['th', 'td'])[1:]]

            # Map row labels to our keys
            row_map = {
                'operating profit': 'operating_profit',
                'profit before tax': 'profit_before_tax',
                'net profit': 'net_profit',
                'eps in rs': 'earnings_per_share',
            }
            # Initialize result dict for each year
            for year in years:
                profit_loss[year] = {}

            for row in rows[1:]:
                cells = row.find_all(['td', 'th'])
                
                if len(cells) < 2:
                    continue
                first_td = cells[0]
                button = first_td.find("button")
                if button:
                    button_text = button.get_text(separator=" ").strip().lower()                    
                    button_text = re.sub(r'[\xa0\s]+\+₹', '', button_text)  # Remove trailing + and spaces
                    button_text = re.sub(r'\s+', ' ', button_text)  # Normalize spaces
                    row_label = button_text.strip()
                else:
                    row_label = first_td.get_text().strip().lower()
                key = row_map.get(row_label)
                
                if not key:
                    continue
                values = [self._clean_financial_value(cell.get_text().strip()) for cell in cells[1:]]
                for i, year in enumerate(years):
                    if i < len(values):
                        profit_loss[year][key] = values[i]

            # Sort years in descending order and build ordered dict
            ordered_profit_loss = {}
            for year in sorted(profit_loss.keys(), key=lambda y: int(re.search(r'\d{4}', y).group()) if re.search(r'\d{4}', y) else 0, reverse=True):
                ordered_profit_loss[year] = profit_loss[year]
            return ordered_profit_loss
        
        except Exception as e:
            print(f"Error extracting profit & loss: {e}")
            return profit_loss
    
    def _extract_cash_flow(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract cash flow statement data grouped by year"""
        
        cash_flow = {}
        try:
            table = soup.select_one("#cash-flow table.data-table")
            if not table:
                return cash_flow
            rows = table.find_all('tr')
            if len(rows) < 2:
                return cash_flow

            # Get years from header
            header_row = rows[0]
            years = []
            for th in header_row.find_all(['th', 'td'])[1:]:
                year_text = th.get_text().strip()
                if year_text and re.search(r'\d{4}', year_text):
                    years.append(year_text)

            # Initialize dict for each year
            for year in years:
                if year not in cash_flow:
                    cash_flow[year] = {}

            current_section = None

            for row in rows[1:]:
                cells = row.find_all(['td', 'th'])
                if len(cells) < 2:
                    continue

                row_label = cells[0].get_text().strip().lower()

                # Identify cash flow sections (optional: add as prefix if you want)
                if 'operating' in row_label:
                    current_section = 'operating_activities'
                elif 'investing' in row_label:
                    current_section = 'investing_activities'
                elif 'financing' in row_label:
                    current_section = 'financing_activities'
                elif 'net cash' in row_label:
                    current_section = 'net_cash_flow'

                # Extract values for each year
                if current_section:
                    field_name = self._normalize_field_name(row_label)
                    for i, cell in enumerate(cells[1:]):
                        if i < len(years):
                            value_text = cell.get_text().strip()
                            value = self._clean_financial_value(value_text)
                            # Optionally, prefix with section: f"{current_section}_{field_name}"
                            cash_flow[years[i]][field_name] = value

            # Optionally, sort years descending
            cash_flow = dict(sorted(cash_flow.items(), key=lambda x: int(re.search(r'\d{4}', x[0]).group()), reverse=True))            
            return cash_flow
        

        except Exception as e:
            print(f"Error extracting cash flow: {e}")
            return cash_flow

    def _extract_shareholding(self, soup: BeautifulSoup, quarterly: bool) -> Dict[str, Any]:
        """Extract shareholding pattern data, all data are in percentage"""
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
            header_row = rows[0]
            years = [th.get_text().strip() for th in header_row.find_all(['th', 'td'])[1:]]

            # Map row labels to our keys
            row_map = {
                'promoters': 'promoter_holding',
                'foreign_institutions': 'fii_holding',
                'domestic_institutions': 'dii_holding',
                'government': 'government_holding',
                'public': 'public_holding',
            }
            # Initialize result dict for each year
            for year in years:
                shareholding[year] = {}

            for row in rows[1:]:
                cells = row.find_all(['td', 'th'])
                
                if len(cells) < 2:
                    continue
                first_td = cells[0]
                button = first_td.find("button")
                if not button:
                    continue

                onclick = button.get("onclick", "")
                match = re.search(r"showShareholders\('([^']+)'", onclick)
                if match:
                    group = match.group(1).lower()
                else:
                    # fallback to button text if onclick is missing
                    button_text = button.get_text(separator=" ").strip().lower()
                    button_text = re.sub(r'[\xa0\s]+\+₹', '', button_text)  # Remove trailing + and spaces
                    button_text = re.sub(r'\s+', ' ', button_text)  # Normalize spaces
                    group = button_text.strip()
                
                row_map = {
                    'promoters': 'promoter_holding',
                    'foreign_institutions': 'fii_holding',
                    'domestic_institutions': 'dii_holding',
                    'government': 'government_holding',
                    'public': 'public_holding',
                }

                key = row_map.get(group)
                
                if not key:
                    continue
                values = [self._clean_financial_value(cell.get_text().strip()) for cell in cells[1:]]
                for i, year in enumerate(years):
                    if i < len(values):
                        shareholding[year][key] = values[i]

            # Sort years in descending order and build ordered dict
            ordered_shareholding = {}
            for year in sorted(shareholding.keys(), key=lambda y: int(re.search(r'\d{4}', y).group()) if re.search(r'\d{4}', y) else 0, reverse=True):
                ordered_shareholding[year] = shareholding[year]
            return ordered_shareholding

        except Exception as e:
            print(f"Error extracting shareholding: {e}")
            return shareholding
    
    def _extract_peer_comparison(self, soup: BeautifulSoup) -> dict:
        """Extract peer comparison data from the peers table using BeautifulSoup only."""
        peers = []
        median = None
        try:
            table = soup.select_one("#peers #peers-table-placeholder table.data-table")
            if not table:
                print("No peer comparison table found.")
                return {"peers": peers, "median": median}

            rows = table.find_all('tr')
            if len(rows) < 2:
                print("Peer comparison table has insufficient data.")
                return {"peers": peers, "median": median}

            # Get headers and map to our required fields
            header_row = rows[0]
            headers = []
            for th in header_row.find_all(['th', 'td']):
                header_text = th.get_text(separator=" ").strip().lower()
                headers.append(header_text)

            # Map headers to our required keys (robust to variations)
            header_map = {
                's.no.': 'rank',
                'name': 'name',
                'cmp': 'cmp',
                'p/e': 'pe',
                'mar cap': 'market_cap',
                'div yld': 'div_yld',
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
                    # Clean numeric values where appropriate
                    if key in ['cmp', 'pe', 'market_cap', 'div_yld', 'net_profit_qtr', 'qtr_profit_var', 'sales_qtr', 'qtr_sales_var', 'roce', 'rank']:
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

    # Helper methods
    def _clean_financial_value(self, value_text: str) -> float:
        """Clean and convert financial value text to float"""
        if not value_text or value_text.strip() == '-':
            return 0.0
        
        try:
            # Remove currency symbols and commas
            cleaned = re.sub(r'[₹,\s%]', '', value_text)
            
            # Handle negative values
            is_negative = '(' in value_text or '-' in cleaned
            cleaned = cleaned.replace('(', '').replace(')', '').replace('-', '')
            
            # Convert based on suffixes
            multiplier = 1
            if 'cr' in value_text.lower() or 'crore' in value_text.lower():
                multiplier = 10000000  # 1 crore = 10 million
            elif 'l' in value_text.lower() or 'lakh' in value_text.lower():
                multiplier = 100000    # 1 lakh = 100 thousand
            elif 'k' in value_text.lower():
                multiplier = 1000
            
            # Extract numeric value
            numeric_match = re.search(r'([\d.]+)', cleaned)
            if numeric_match:
                value = float((numeric_match.group(1)) * multiplier)
                return -value if is_negative else value
            
            return 0.0
            
        except (ValueError, AttributeError):
            return 0.0
    
    def _normalize_field_name(self, text: str) -> str:
        """Normalize field names for consistency and LLM readability"""
        if not text:
            return "unknown_field"
        
        # Remove common prefixes/suffixes and normalize
        text = re.sub(r'^\s*[-•]\s*', '', text)  # Remove bullet points
        text = re.sub(r'\s*\([^)]*\)\s*', '', text)  # Remove parenthetical info
        text = re.sub(r'[^\w\s]', ' ', text)  # Replace special chars with spaces
        text = re.sub(r'\s+', '_', text.strip())  # Replace spaces with underscores
        
        return text.lower()
    
    def _extract_year_from_string(self, year_str: str) -> int:
        """Extract year from string for sorting"""
        match = re.search(r'\d{4}', year_str)
        return int(match.group()) if match else 0
    
    def _categorize_pl_field(self, field_name: str) -> str:
        """Categorize P&L field into revenue, expenses, or profit"""
        field_lower = field_name.lower()
        
        if any(keyword in field_lower for keyword in ['revenue', 'sales', 'income', 'turnover']):
            return 'revenue'
        elif any(keyword in field_lower for keyword in ['profit', 'net', 'ebit', 'pbt', 'pat']):
            return 'profit'
        else:
            return 'expenses'
    
    def _is_valid_data(self, data: Dict[str, Any]) -> bool:
        """Check if scraped data is valid and complete"""
        return (
            data.get('company_name', 'Unknown') != 'Unknown' and
            data.get('current_price', 0) > 0
        )
    
    def close(self):
        """Clean up resources"""
        if hasattr(self, 'driver'):
            self.driver.quit()


def scraping_data(symbol: str) -> Dict[str, Any]:
    scraper = ScreenerScraper(use_playwright=True)
    try:
        return scraper.scrape_company(symbol, method="auto")
    finally:
        scraper.close()


def get_latest_structured_financial(symbol: str) -> dict:
    """
    Returns the latest structured financial data from the scraped data.
    This function extracts the data such as symbol, company name, current price,
    market cap, financial ratios, and structured financial statements like balance sheet,
    income statement, cash flow statement, and shareholding pattern of the latest year.
    It also determines the latest fiscal year based on the available data.
    If no data is available, it returns an empty dictionary.
    """
    scraped_data = scraping_data(symbol)

    def latest_year(data):
        if not data:
            return None, {}
        years = sorted(data.keys(), key=lambda y: int(re.search(r'\d{4}', y).group()), reverse=True)
        return years[0], data[years[0]]

    # Map balance sheet
    bs_year, bs_data = latest_year(scraped_data.get("balance_sheet", {}))
    pl_year, pl_data = latest_year(scraped_data.get("yearly_profit_loss", {}))
    cf_year, cf_data = latest_year(scraped_data.get("cash_flow", {}))
    sh_year, sh_data = latest_year(scraped_data.get("yearly_shareholding", {}))

    def wrap_section(section_dict):
        # Detect if this is shareholding pattern section by key names
        is_shareholding = any("holding" in k for k in section_dict.keys())
        unit = "percentage" if is_shareholding else "INR Crore"
        return {
            k: {
            "label": k.replace("_", " ").title(),
            "unit": unit,
            "value": v
            }
            for k, v in section_dict.items()
        }

    result = {        
        "results": [
            {
                "symbol": scraped_data.get("symbol"),
                "company_name": scraped_data.get("company_name"),
                "end_date": bs_year or pl_year or cf_year,
                "current_price": scraped_data.get("current_price", 0.0),
                "market_cap": f"{scraped_data.get("market_cap", 0.0)} Cr",
                "financial_ratios": scraped_data.get("financial_ratios", {}),
                "financial": {
                    "balance_sheet": wrap_section(bs_data),
                    "income_statement": wrap_section(pl_data),
                    "cash_flow_statement": wrap_section(cf_data),
                    "shareholding_pattern": wrap_section(sh_data),
                },
                "fiscal_year": bs_year or pl_year or cf_year or sh_year,
                "scrape_timestamp": scraped_data.get("scrape_timestamp"),
            }
        ],
    }
    return result     
        

print(json.dumps(get_latest_structured_financial("IDEA"), indent=2, default=str))




# print(f"Company: {json.dumps(company_data['company_name'], indent=2, default=str)}")
# print(f"Current Price: ₹{json.dumps(company_data['current_price'], indent=2, default=str)}")
# print(f"Market Cap: ₹{json.dumps(company_data['market_cap'], indent=2, default=str)} Cr")
# print(f"Financial Ratios: {json.dumps(company_data['financial_ratios'], indent=2, default=str)}")
# print(f"Quarterly Results: {len(company_data['quarterly_results'])} quarters")
# print(f"Balance Sheet: {json.dumps(company_data['balance_sheet'], indent=2, default=str)}")
# print(f"Profit & Loss: {json.dumps(company_data['profit_loss'], indent=2, default=str)}")
# print(f"Cash Flow: {json.dumps(company_data['cash_flow'], indent=2, default=str)}")
# print(f"Shareholding Pattern: {json.dumps(company_data['shareholding'], indent=2, default=str)}")
# print(f"Peers: {len(company_data['peers'])} peers found")


# This is how you can print the entire data structure in JSON format
# print(json.dumps(company_data['peers'], indent=2, default=str))


#  def example_usage():
#     """Example usage of the scraper"""
    
#     # Initialize scraper
#     scraper = ScreenerScraper(headless=True, use_selenium=True)
    
#     try:
#         # Scrape single company
#         company = "BBOX"
#         start_time = time.time()
#         print(f"Scraping {company}...")
#         company_data = scraper.scrape_company(company, method="auto")
        
#         # Save all data to a JSON file
#         # filename = f"{company}_screener_data.json"
#         # with open(filename, "w", encoding="utf-8") as f:
#         #     json.dump(company_data, f, indent=2, default=str)
#         # print(f"Data saved to {filename}")
       
        
#         print(f"Balance Sheet: {json.dumps(company_data['cash_flow'], indent=2, default=str)}")
#         end_time = time.time()
#         print(f"Cash flow extraction took {end_time - start_time:.2f} seconds")
        
#     finally:
#         scraper.close()


