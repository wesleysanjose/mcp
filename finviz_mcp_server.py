import requests
from bs4 import BeautifulSoup
import json
import time
import logging
import asyncio
from typing import Dict, List, Any, Optional
from mcp.server.fastmcp import FastMCP
import os
import sys

# Set up logging
log_file = os.path.join(os.path.expanduser("~"), "finviz_mcp.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("finviz_mcp")

# Initialize FastMCP server
mcp = FastMCP("finviz")

# Constants
FINVIZ_URL = "https://finviz.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# Cache for storing scraped data to reduce requests to Finviz
cache = {}
CACHE_TIMEOUT = 300  # 5 minutes


async def make_request(url: str) -> Optional[str]:
    """Make an async HTTP request"""
    try:
        # Use asyncio to run the blocking request in a separate thread
        return await asyncio.to_thread(
            lambda: requests.get(url, headers=HEADERS).text
        )
    except Exception as e:
        logger.error(f"Request error: {str(e)}")
        return None


@mcp.tool()
async def get_stock_data(ticker: str) -> Dict[str, Any]:
    """Get detailed stock data for a specific ticker.
    
    Args:
        ticker: The stock ticker symbol (e.g., AAPL, MSFT)
    
    Returns:
        A dictionary containing stock information including price, fundamentals, 
        technical indicators, news, and insider trading.
    """
    cache_key = f"stock_{ticker}"
    current_time = time.time()
    
    # Check if data is in cache and still fresh
    if cache_key in cache and current_time - cache[cache_key]["timestamp"] < CACHE_TIMEOUT:
        logger.info(f"Using cached data for {ticker}")
        return cache[cache_key]["data"]
    
    logger.info(f"Fetching data for {ticker} from Finviz")
    try:
        url = f"{FINVIZ_URL}/quote.ashx?t={ticker}"
        html_content = await make_request(url)
        
        if not html_content:
            return {"error": f"Could not fetch data for ticker {ticker}"}
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract snapshot table data
        snapshot_table = soup.find('table', {'class': 'snapshot-table2'})
        
        if not snapshot_table:
            return {"error": f"Could not find data for ticker {ticker}"}
            
        # Parse the snapshot table
        data = {}
        rows = snapshot_table.find_all('tr')
        for row in rows:
            cells = row.find_all('td')
            for i in range(0, len(cells), 2):
                if i + 1 < len(cells):
                    key = cells[i].text.strip()
                    value = cells[i+1].text.strip()
                    data[key] = value
        
        # Get news headlines
        news_table = soup.find('table', {'class': 'fullview-news-outer'})
        news = []
        if news_table:
            for row in news_table.find_all('tr'):
                date_cell = row.find('td', {'align': 'right'})
                news_cell = row.find('td', {'align': 'left'})
                
                if date_cell and news_cell and news_cell.a:
                    date = date_cell.text.strip()
                    headline = news_cell.a.text.strip()
                    link = news_cell.a.get('href', '')
                    news.append({
                        "date": date,
                        "headline": headline,
                        "link": link
                    })
        
        data["news"] = news
        
        # Get insider trading if available
        insider_table = soup.find('table', {'class': 'body-table'})
        insider_trades = []
        if insider_table:
            headers = [th.text for th in insider_table.find_all('th')]
            for row in insider_table.find_all('tr')[1:]:  # Skip header row
                cells = row.find_all('td')
                if len(cells) >= len(headers):
                    trade = {}
                    for i, header in enumerate(headers):
                        trade[header.strip()] = cells[i].text.strip()
                    insider_trades.append(trade)
        
        data["insider_trades"] = insider_trades
        
        # Store in cache
        cache[cache_key] = {
            "timestamp": current_time,
            "data": data
        }
        
        return data
        
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {str(e)}")
        return {"error": str(e)}


@mcp.tool()
async def get_market_news() -> List[Dict[str, str]]:
    """Get the latest market news from Finviz.
    
    Returns:
        A list of news items with date, source, headline, and link.
    """
    cache_key = "market_news"
    current_time = time.time()
    
    # Check if data is in cache and still fresh
    if cache_key in cache and current_time - cache[cache_key]["timestamp"] < CACHE_TIMEOUT:
        logger.info(f"Using cached market news")
        return cache[cache_key]["data"]
    
    logger.info("Fetching market news from Finviz")
    try:
        url = f"{FINVIZ_URL}/news.ashx"
        html_content = await make_request(url)
        
        if not html_content:
            return {"error": "Could not fetch market news"}
        
        soup = BeautifulSoup(html_content, 'html.parser')
        news_table = soup.find('table', {'class': 'newstable'})
        
        news = []
        if news_table:
            for row in news_table.find_all('tr'):
                cells = row.find_all('td')
                if len(cells) >= 3 and cells[2].a:
                    date = cells[0].text.strip()
                    source = cells[1].text.strip()
                    headline = cells[2].a.text.strip()
                    link = cells[2].a.get('href', '')
                    news.append({
                        "date": date,
                        "source": source,
                        "headline": headline,
                        "link": link
                    })
        
        # Store in cache
        cache[cache_key] = {
            "timestamp": current_time,
            "data": news
        }
        
        return news
        
    except Exception as e:
        logger.error(f"Error fetching market news: {str(e)}")
        return {"error": str(e)}


@mcp.tool()
async def get_screener_results(filters: Optional[Dict[str, str]] = None) -> List[Dict[str, str]]:
    """Get stocks from the Finviz screener with optional filters.
    
    Args:
        filters: Optional dictionary of Finviz screening parameters
        
    Returns:
        A list of stocks matching the specified filters.
    """
    cache_key = f"screener_{json.dumps(filters) if filters else 'default'}"
    current_time = time.time()
    
    # Check if data is in cache and still fresh
    if cache_key in cache and current_time - cache[cache_key]["timestamp"] < CACHE_TIMEOUT:
        logger.info(f"Using cached screener results")
        return cache[cache_key]["data"]
    
    logger.info("Fetching screener results from Finviz")
    try:
        url = f"{FINVIZ_URL}/screener.ashx?v=111"
        
        # Add filters if provided
        if filters:
            filter_params = "&".join([f"f={k}_{v}" for k, v in filters.items()])
            url = f"{url}&{filter_params}"
        
        html_content = await make_request(url)
        
        if not html_content:
            return {"error": "Could not fetch screener results"}
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Find the screener table
        table = soup.find('table', {'class': 'table-light'})
        if not table:
            return {"error": "Could not find screener results"}
        
        # Get headers
        headers = []
        header_row = table.find_all('tr')[0]
        for th in header_row.find_all('td'):
            headers.append(th.text.strip())
        
        # Get stock data
        stocks = []
        rows = table.find_all('tr')[1:]  # Skip header row
        for row in rows:
            cells = row.find_all('td')
            if len(cells) >= len(headers):
                stock = {}
                for i, header in enumerate(headers):
                    if header:  # Skip empty headers
                        stock[header] = cells[i].text.strip()
                stocks.append(stock)
        
        # Store in cache
        cache[cache_key] = {
            "timestamp": current_time,
            "data": stocks
        }
        
        return stocks
        
    except Exception as e:
        logger.error(f"Error fetching screener results: {str(e)}")
        return {"error": str(e)}


@mcp.tool()
async def get_stock_analysis(ticker: str) -> Dict[str, Any]:
    """Get analysis for a stock based on technical and fundamental metrics.
    
    Args:
        ticker: The stock ticker symbol (e.g., AAPL, MSFT)
        
    Returns:
        A dictionary containing analysis results including recommendation,
        reasoning, risk level, and key data points.
    """
    stock_data = await get_stock_data(ticker)
    
    if "error" in stock_data:
        return stock_data
    
    # Basic analysis
    analysis = {
        "ticker": ticker,
        "price": stock_data.get("Price", "N/A"),
        "recommendation": "Unknown",
        "reasoning": [],
        "data_points": {},
        "risk_level": "Medium"
    }
    
    # Extract and normalize key metrics for analysis
    try:
        # Price metrics
        if "Price" in stock_data:
            price = float(stock_data["Price"].replace("$", ""))
            analysis["data_points"]["price"] = price
        
        if "Target Price" in stock_data:
            target_price = float(stock_data["Target Price"].replace("$", ""))
            analysis["data_points"]["target_price"] = target_price
            
            # Calculate upside potential
            if "price" in analysis["data_points"]:
                upside = ((target_price / price) - 1) * 100
                analysis["data_points"]["upside_potential"] = upside
                
                if upside > 20:
                    analysis["reasoning"].append(f"Target price indicates {upside:.1f}% upside potential")
                elif upside < -10:
                    analysis["reasoning"].append(f"Target price indicates {abs(upside):.1f}% downside risk")
        
        # Valuation metrics
        if "P/E" in stock_data and stock_data["P/E"] != "-":
            pe = float(stock_data["P/E"])
            analysis["data_points"]["pe_ratio"] = pe
            
            if pe < 15:
                analysis["reasoning"].append(f"P/E ratio of {pe} is relatively low")
            elif pe > 30:
                analysis["reasoning"].append(f"P/E ratio of {pe} is relatively high")
        
        # Growth metrics
        if "EPS growth this year" in stock_data and stock_data["EPS growth this year"] != "-":
            eps_growth = float(stock_data["EPS growth this year"].replace("%", ""))
            analysis["data_points"]["eps_growth"] = eps_growth
            
            if eps_growth > 20:
                analysis["reasoning"].append(f"Strong EPS growth of {eps_growth}%")
            elif eps_growth < 0:
                analysis["reasoning"].append(f"Negative EPS growth of {eps_growth}%")
        
        # Technical indicators
        if "RSI (14)" in stock_data and stock_data["RSI (14)"] != "-":
            rsi = float(stock_data["RSI (14)"])
            analysis["data_points"]["rsi"] = rsi
            
            if rsi < 30:
                analysis["reasoning"].append(f"RSI of {rsi} indicates potential oversold condition")
            elif rsi > 70:
                analysis["reasoning"].append(f"RSI of {rsi} indicates potential overbought condition")
        
        # Insider activity
        if stock_data.get("insider_trades") and len(stock_data["insider_trades"]) > 0:
            buys = sum(1 for trade in stock_data["insider_trades"] if "Buy" in trade.get("Transaction", ""))
            sells = sum(1 for trade in stock_data["insider_trades"] if "Sell" in trade.get("Transaction", ""))
            
            analysis["data_points"]["insider_buys"] = buys
            analysis["data_points"]["insider_sells"] = sells
            
            if buys > sells * 2:
                analysis["reasoning"].append(f"Strong insider buying ({buys} buys vs {sells} sells)")
            elif sells > buys * 2:
                analysis["reasoning"].append(f"Significant insider selling ({sells} sells vs {buys} buys)")
        
        # Make a simple recommendation based on the analysis
        positive_points = sum(1 for reason in analysis["reasoning"] if "high" not in reason.lower() and "overbought" not in reason.lower() and "downside" not in reason.lower() and "negative" not in reason.lower())
        negative_points = len(analysis["reasoning"]) - positive_points
        
        if positive_points > negative_points + 1:
            analysis["recommendation"] = "Buy"
            analysis["risk_level"] = "Low" if negative_points == 0 else "Medium"
        elif negative_points > positive_points + 1:
            analysis["recommendation"] = "Sell"
            analysis["risk_level"] = "Low" if positive_points == 0 else "Medium"
        else:
            analysis["recommendation"] = "Hold"
            analysis["risk_level"] = "Medium"
        
    except Exception as e:
        logger.error(f"Error in analysis for {ticker}: {str(e)}")
        analysis["error"] = str(e)
    
    return analysis


@mcp.tool()
async def batch_analyze_stocks(tickers: List[str]) -> Dict[str, Dict[str, Any]]:
    """Analyze multiple stocks at once.
    
    Args:
        tickers: List of stock ticker symbols to analyze
        
    Returns:
        A dictionary mapping each ticker to its analysis result.
    """
    results = {}
    
    for ticker in tickers:
        results[ticker] = await get_stock_analysis(ticker)
    
    return results


@mcp.tool()
async def find_stocks_by_criteria(
    min_pe: Optional[float] = None,
    max_pe: Optional[float] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    min_market_cap: Optional[str] = None,
    sector: Optional[str] = None,
    min_dividend: Optional[float] = None
) -> List[Dict[str, Any]]:
    """Find stocks that match specific criteria.
    
    Args:
        min_pe: Minimum P/E ratio
        max_pe: Maximum P/E ratio
        min_price: Minimum stock price
        max_price: Maximum stock price
        min_market_cap: Minimum market cap (e.g., "Large", "Mid", "Small")
        sector: Specific sector (e.g., "Technology", "Healthcare")
        min_dividend: Minimum dividend yield percentage
        
    Returns:
        A list of stocks that match the specified criteria.
    """
    # Build Finviz filter parameters
    filters = {}
    
    if min_pe is not None:
        filters["fa_pe_o"] = str(min_pe)
    if max_pe is not None:
        filters["fa_pe_u"] = str(max_pe)
    
    if min_price is not None:
        filters["sh_price_o"] = str(min_price)
    if max_price is not None:
        filters["sh_price_u"] = str(max_price)
    
    if min_market_cap is not None:
        if min_market_cap.lower() == "large":
            filters["fa_marketcap"] = "largeover"
        elif min_market_cap.lower() == "mid":
            filters["fa_marketcap"] = "midover"
        elif min_market_cap.lower() == "small":
            filters["fa_marketcap"] = "smallover"
    
    if sector is not None:
        filters["sec_sector"] = sector
    
    if min_dividend is not None:
        filters["fa_div_o"] = str(min_dividend)
    
    # Get screener results with these filters
    return await get_screener_results(filters)


if __name__ == "__main__":
    # Initialize and run the server with stdio transport for MCP
    mcp.run(transport='stdio')
    # After initializing the server
    print("Starting Finviz MCP server", file=sys.stderr)
    print(f"Python executable: {sys.executable}", file=sys.stderr)