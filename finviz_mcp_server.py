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
async def get_stock_info(ticker: str) -> Dict[str, Any]:
    """Get basic information about a stock.
    
    Args:
        ticker: The stock ticker symbol (e.g., AAPL, MSFT)
        
    Returns:
        Dictionary with stock information including company details, sector, industry, etc.
    """
    cache_key = f"stock_info_{ticker}"
    current_time = time.time()
    
    # Check if data is in cache and still fresh
    if cache_key in cache and current_time - cache[cache_key]["timestamp"] < CACHE_TIMEOUT:
        logger.info(f"Using cached stock info for {ticker}")
        return cache[cache_key]["data"]
    
    logger.info(f"Fetching stock info for {ticker}")
    try:
        # Get stock info from yfinance
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Filter and organize the information
        result = {
            "ticker": ticker,
            "name": info.get("shortName", "N/A"),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "country": info.get("country", "N/A"),
            "exchange": info.get("exchange", "N/A"),
            "currency": info.get("currency", "N/A"),
            "market_cap": info.get("marketCap", "N/A"),
            "pe_ratio": info.get("trailingPE", "N/A"),
            "dividend_yield": info.get("dividendYield", "N/A"),
            "52wk_high": info.get("fiftyTwoWeekHigh", "N/A"),
            "52wk_low": info.get("fiftyTwoWeekLow", "N/A"),
            "avg_volume": info.get("averageVolume", "N/A"),
            "website": info.get("website", "N/A"),
            "business_summary": info.get("longBusinessSummary", "N/A")
        }
        
        # Store in cache
        cache[cache_key] = {
            "timestamp": current_time,
            "data": result
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error fetching stock info for {ticker}: {str(e)}")
        return {"error": str(e)}

@mcp.tool()
async def get_historical_data(
    ticker: str, 
    period: str = "1y", 
    interval: str = "1d"
) -> Dict[str, Any]:
    """Get historical price data for a stock.
    
    Args:
        ticker: The stock ticker symbol
        period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        
    Returns:
        Dictionary with historical price data and basic statistics
    """
    cache_key = f"historical_{ticker}_{period}_{interval}"
    current_time = time.time()
    
    # Check if data is in cache and still fresh
    if cache_key in cache and current_time - cache[cache_key]["timestamp"] < CACHE_TIMEOUT:
        logger.info(f"Using cached historical data for {ticker}")
        return cache[cache_key]["data"]
    
    logger.info(f"Fetching historical data for {ticker} with period {period} and interval {interval}")
    try:
        # Get historical data from yfinance
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, interval=interval)
        
        if hist.empty:
            return {"error": f"No historical data available for {ticker}"}
        
        # Convert DataFrame to dictionary for JSON serialization
        hist_dict = hist.reset_index().to_dict(orient='records')
        
        # Calculate basic statistics
        stats = {
            "start_date": hist.index[0].strftime("%Y-%m-%d"),
            "end_date": hist.index[-1].strftime("%Y-%m-%d"),
            "days": len(hist),
            "price_start": round(hist['Close'].iloc[0], 2),
            "price_end": round(hist['Close'].iloc[-1], 2),
            "price_change": round(hist['Close'].iloc[-1] - hist['Close'].iloc[0], 2),
            "price_change_pct": round(((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1) * 100, 2),
            "price_high": round(hist['High'].max(), 2),
            "price_low": round(hist['Low'].min(), 2),
            "volume_avg": int(hist['Volume'].mean())
        }
        
        result = {
            "ticker": ticker,
            "period": period,
            "interval": interval,
            "statistics": stats,
            "data": hist_dict[:100]  # Limit to 100 data points to avoid large responses
        }
        
        # Store in cache
        cache[cache_key] = {
            "timestamp": current_time,
            "data": result
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error fetching historical data for {ticker}: {str(e)}")
        return {"error": str(e)}

@mcp.tool()
async def get_financial_data(ticker: str) -> Dict[str, Any]:
    """Get financial data for a stock including income statement, balance sheet, and cash flow.
    
    Args:
        ticker: The stock ticker symbol
        
    Returns:
        Dictionary with financial data
    """
    cache_key = f"financials_{ticker}"
    current_time = time.time()
    
    # Check if data is in cache and still fresh
    if cache_key in cache and current_time - cache[cache_key]["timestamp"] < CACHE_TIMEOUT:
        logger.info(f"Using cached financial data for {ticker}")
        return cache[cache_key]["data"]
    
    logger.info(f"Fetching financial data for {ticker}")
    try:
        # Get financial data from yfinance
        stock = yf.Ticker(ticker)
        
        # Get income statement, balance sheet, and cash flow
        income_stmt = stock.income_stmt
        balance_sheet = stock.balance_sheet
        cash_flow = stock.cashflow
        
        # Convert DataFrames to dictionaries for JSON serialization
        result = {
            "ticker": ticker,
            "income_statement": income_stmt.to_dict() if not income_stmt.empty else {},
            "balance_sheet": balance_sheet.to_dict() if not balance_sheet.empty else {},
            "cash_flow": cash_flow.to_dict() if not cash_flow.empty else {}
        }
        
        # Store in cache
        cache[cache_key] = {
            "timestamp": current_time,
            "data": result
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error fetching financial data for {ticker}: {str(e)}")
        return {"error": str(e)}

@mcp.tool()
async def get_technical_indicators(
    ticker: str, 
    period: str = "1y"
) -> Dict[str, Any]:
    """Calculate technical indicators for a stock.
    
    Args:
        ticker: The stock ticker symbol
        period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        
    Returns:
        Dictionary with technical indicators including moving averages, RSI, MACD, etc.
    """
    cache_key = f"technical_{ticker}_{period}"
    current_time = time.time()
    
    # Check if data is in cache and still fresh
    if cache_key in cache and current_time - cache[cache_key]["timestamp"] < CACHE_TIMEOUT:
        logger.info(f"Using cached technical indicators for {ticker}")
        return cache[cache_key]["data"]
    
    logger.info(f"Calculating technical indicators for {ticker}")
    try:
        # Get historical data
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        
        if hist.empty:
            return {"error": f"No historical data available for {ticker}"}
        
        # Calculate moving averages
        hist['SMA20'] = hist['Close'].rolling(window=20).mean()
        hist['SMA50'] = hist['Close'].rolling(window=50).mean()
        hist['SMA200'] = hist['Close'].rolling(window=200).mean()
        hist['EMA12'] = hist['Close'].ewm(span=12, adjust=False).mean()
        hist['EMA26'] = hist['Close'].ewm(span=26, adjust=False).mean()
        
        # Calculate RSI
        delta = hist['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        hist['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        hist['MACD'] = hist['EMA12'] - hist['EMA26']
        hist['MACD_Signal'] = hist['MACD'].ewm(span=9, adjust=False).mean()
        hist['MACD_Hist'] = hist['MACD'] - hist['MACD_Signal']
        
        # Calculate Bollinger Bands
        hist['BB_Middle'] = hist['Close'].rolling(window=20).mean()
        hist['BB_Std'] = hist['Close'].rolling(window=20).std()
        hist['BB_Upper'] = hist['BB_Middle'] + (hist['BB_Std'] * 2)
        hist['BB_Lower'] = hist['BB_Middle'] - (hist['BB_Std'] * 2)
        
        # Get the latest values
        latest = hist.iloc[-1]
        
        # Prepare the result
        result = {
            "ticker": ticker,
            "date": hist.index[-1].strftime("%Y-%m-%d"),
            "price": round(latest['Close'], 2),
            "indicators": {
                "sma20": round(latest['SMA20'], 2) if not pd.isna(latest['SMA20']) else None,
                "sma50": round(latest['SMA50'], 2) if not pd.isna(latest['SMA50']) else None,
                "sma200": round(latest['SMA200'], 2) if not pd.isna(latest['SMA200']) else None,
                "ema12": round(latest['EMA12'], 2) if not pd.isna(latest['EMA12']) else None,
                "ema26": round(latest['EMA26'], 2) if not pd.isna(latest['EMA26']) else None,
                "rsi": round(latest['RSI'], 2) if not pd.isna(latest['RSI']) else None,
                "macd": round(latest['MACD'], 2) if not pd.isna(latest['MACD']) else None,
                "macd_signal": round(latest['MACD_Signal'], 2) if not pd.isna(latest['MACD_Signal']) else None,
                "macd_hist": round(latest['MACD_Hist'], 2) if not pd.isna(latest['MACD_Hist']) else None,
                "bb_upper": round(latest['BB_Upper'], 2) if not pd.isna(latest['BB_Upper']) else None,
                "bb_middle": round(latest['BB_Middle'], 2) if not pd.isna(latest['BB_Middle']) else None,
                "bb_lower": round(latest['BB_Lower'], 2) if not pd.isna(latest['BB_Lower']) else None
            },
            "signals": []
        }
        
        # Generate signals
        signals = []
        
        # Moving Average signals
        if not pd.isna(latest['SMA20']) and not pd.isna(latest['SMA50']):
            if latest['Close'] > latest['SMA20'] and latest['Close'] > latest['SMA50']:
                signals.append({"indicator": "Moving Averages", "signal": "Bullish", "description": "Price above SMA20 and SMA50"})
            elif latest['Close'] < latest['SMA20'] and latest['Close'] < latest['SMA50']:
                signals.append({"indicator": "Moving Averages", "signal": "Bearish", "description": "Price below SMA20 and SMA50"})
        
        # RSI signals
        if not pd.isna(latest['RSI']):
            if latest['RSI'] > 70:
                signals.append({"indicator": "RSI", "signal": "Overbought", "description": f"RSI at {round(latest['RSI'], 2)} indicates overbought conditions"})
            elif latest['RSI'] < 30:
                signals.append({"indicator": "RSI", "signal": "Oversold", "description": f"RSI at {round(latest['RSI'], 2)} indicates oversold conditions"})
        
        # MACD signals
        if not pd.isna(latest['MACD']) and not pd.isna(latest['MACD_Signal']):
            if latest['MACD'] > latest['MACD_Signal']:
                signals.append({"indicator": "MACD", "signal": "Bullish", "description": "MACD above signal line"})
            else:
                signals.append({"indicator": "MACD", "signal": "Bearish", "description": "MACD below signal line"})
        
        # Bollinger Bands signals
        if not pd.isna(latest['BB_Upper']) and not pd.isna(latest['BB_Lower']):
            if latest['Close'] > latest['BB_Upper']:
                signals.append({"indicator": "Bollinger Bands", "signal": "Overbought", "description": "Price above upper Bollinger Band"})
            elif latest['Close'] < latest['BB_Lower']:
                signals.append({"indicator": "Bollinger Bands", "signal": "Oversold", "description": "Price below lower Bollinger Band"})
        
        result["signals"] = signals
        
        # Store in cache
        cache[cache_key] = {
            "timestamp": current_time,
            "data": result
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error calculating technical indicators for {ticker}: {str(e)}")
        return {"error": str(e)}

@mcp.tool()
async def get_stock_news(ticker: str, days: int = 7) -> Dict[str, Any]:
    """Get recent news articles for a stock.
    
    Args:
        ticker: The stock ticker symbol
        days: Number of days to look back for news
        
    Returns:
        Dictionary with news articles
    """
    cache_key = f"news_{ticker}_{days}"
    current_time = time.time()
    
    # Check if data is in cache and still fresh
    if cache_key in cache and current_time - cache[cache_key]["timestamp"] < CACHE_TIMEOUT:
        logger.info(f"Using cached news for {ticker}")
        return cache[cache_key]["data"]
    
    logger.info(f"Fetching news for {ticker}")
    try:
        # Get news from yfinance
        stock = yf.Ticker(ticker)
        news = stock.news
        
        # Filter and format news
        result = {
            "ticker": ticker,
            "count": len(news),
            "articles": []
        }
        
        for article in news:
            # Convert timestamp to datetime
            article_date = datetime.fromtimestamp(article.get("providerPublishTime", 0))
            
            # Check if article is within the specified days
            if (datetime.now() - article_date).days <= days:
                result["articles"].append({
                    "title": article.get("title", ""),
                    "publisher": article.get("publisher", ""),
                    "link": article.get("link", ""),
                    "published": article_date.strftime("%Y-%m-%d %H:%M:%S"),
                    "summary": article.get("summary", "")
                })
        
        # Store in cache
        cache[cache_key] = {
            "timestamp": current_time,
            "data": result
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error fetching news for {ticker}: {str(e)}")
        return {"error": str(e)}

@mcp.tool()
async def market_scanner(
    tickers: List[str],
    scan_type: str = "technical",
    min_volume_increase: float = 2.0,
    breakout_threshold: float = 3.0,
    max_results: int = 10
) -> Dict[str, Any]:
    """Scan the market for trading opportunities.
    
    Args:
        tickers: List of stock ticker symbols to scan
        scan_type: Type of scan to perform (technical, breakout, volume, momentum, reversal)
        min_volume_increase: Minimum volume increase ratio for volume scans
        breakout_threshold: Percentage threshold for breakout detection
        max_results: Maximum number of results to return
        
    Returns:
        Dictionary with scan results
    """
    logger.info(f"Running market scanner on {len(tickers)} tickers with scan type: {scan_type}")
    
    try:
        # Initialize results
        results = {
            "scan_type": scan_type,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "matches": [],
            "stats": {
                "total_scanned": len(tickers),
                "total_matches": 0,
                "execution_time": 0
            }
        }
        
        start_time = time.time()
        
        # Process tickers in parallel using ThreadPoolExecutor
        matches = []
        
        # This is a placeholder for the actual scanning logic
        # In a real implementation, you would download data for each ticker
        # and apply the appropriate scanning criteria
        
        # For now, we'll just return a message indicating this is a skeleton
        results["matches"] = [
            {
                "ticker": "EXAMPLE",
                "price": 100.0,
                "signals": ["This is a skeleton implementation"],
                "match_reason": "Placeholder for actual scanning logic"
            }
        ]
        
        results["stats"]["total_matches"] = len(results["matches"])
        results["stats"]["execution_time"] = round(time.time() - start_time, 2)
        
        return results
        
    except Exception as e:
        logger.error(f"Error in market scanner: {str(e)}")
        return {"error": str(e)}

@mcp.tool()
async def portfolio_analytics(
    tickers: List[str],
    weights: Optional[List[float]] = None,
    period: str = "1y",
    benchmark: str = "SPY"
) -> Dict[str, Any]:
    """Analyze a portfolio of stocks and calculate performance metrics.
    
    Args:
        tickers: List of stock ticker symbols in the portfolio
        weights: Optional list of portfolio weights (will use equal weights if not provided)
        period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        benchmark: Ticker symbol for benchmark comparison
        
    Returns:
        Dictionary with portfolio analytics
    """
    logger.info(f"Running portfolio analytics on {len(tickers)} tickers")
    
    try:
        # Initialize results
        results = {
            "portfolio": {
                "tickers": tickers,
                "weights": weights if weights else [1/len(tickers)] * len(tickers),
                "period": period
            },
            "performance": {},
            "risk_metrics": {},
            "correlations": {},
            "benchmark_comparison": {}
        }
        
        # This is a placeholder for the actual portfolio analytics logic
        # In a real implementation, you would download data for each ticker,
        # calculate returns, risk metrics, correlations, etc.
        
        # For now, we'll just return a message indicating this is a skeleton
        results["performance"] = {
            "message": "This is a skeleton implementation. In a real implementation, this would contain portfolio performance metrics."
        }
        
        return results
        
    except Exception as e:
        logger.error(f"Error in portfolio analytics: {str(e)}")
        return {"error": str(e)}

@mcp.tool()
async def trading_signal_generator(
    ticker: str,
    strategy: str = "sma_crossover",
    period: str = "1y",
    interval: str = "1d",
    backtest: bool = True
) -> Dict[str, Any]:
    """Generate trading signals based on technical indicators and backtest performance.
    
    Args:
        ticker: Stock ticker symbol
        strategy: Trading strategy to use (sma_crossover, macd, rsi, bollinger, multi)
        period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        backtest: Whether to run a backtest on historical data
        
    Returns:
        Dictionary with trading signals and backtest results if requested
    """
    logger.info(f"Generating trading signals for {ticker} using {strategy} strategy")
    
    try:
        # Initialize results
        results = {
            "ticker": ticker,
            "strategy": strategy,
            "current_price": 0,
            "signals": [],
            "current_position": "Neutral",
            "backtest_results": {} if backtest else None
        }
        
        # This is a placeholder for the actual signal generation logic
        # In a real implementation, you would download data for the ticker,
        # apply the strategy, generate signals, and run a backtest if requested
        
        # For now, we'll just return a message indicating this is a skeleton
        results["signals"] = [
            {
                "type": "placeholder",
                "signal": "Neutral",
                "reason": "This is a skeleton implementation"
            }
        ]
        
        return results
        
    except Exception as e:
        logger.error(f"Error in trading signal generator: {str(e)}")
        return {"error": str(e)}

@mcp.tool()
async def get_options_data(ticker: str) -> Dict[str, Any]:
    """Get options chain data for a stock.
    
    Args:
        ticker: The stock ticker symbol
        
    Returns:
        Dictionary with options data including calls and puts
    """
    cache_key = f"options_{ticker}"
    current_time = time.time()
    
    # Check if data is in cache and still fresh
    if cache_key in cache and current_time - cache[cache_key]["timestamp"] < CACHE_TIMEOUT:
        logger.info(f"Using cached options data for {ticker}")
        return cache[cache_key]["data"]
    
    logger.info(f"Fetching options data for {ticker}")
    try:
        # Get options data from yfinance
        stock = yf.Ticker(ticker)
        
        # Get options expiration dates
        expirations = stock.options
        
        if not expirations:
            return {"error": f"No options data available for {ticker}"}
        
        # Get options for the nearest expiration date
        expiry = expirations[0]
        
        # Get options chain
        options = stock.option_chain(expiry)
        
        # Convert DataFrames to dictionaries for JSON serialization
        calls = options.calls.to_dict(orient='records')
        puts = options.puts.to_dict(orient='records')
        
        result = {
            "ticker": ticker,
            "underlying_price": stock.info.get("regularMarketPrice", 0),
            "expiration_date": expiry,
            "all_expirations": expirations,
            "calls": calls,
            "puts": puts
        }
        
        # Store in cache
        cache[cache_key] = {
            "timestamp": current_time,
            "data": result
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error fetching options data for {ticker}: {str(e)}")
        return {"error": str(e)}

@mcp.tool()
async def get_earnings_calendar(tickers: List[str]) -> Dict[str, Any]:
    """Get upcoming earnings dates for a list of stocks.
    
    Args:
        tickers: List of stock ticker symbols
        
    Returns:
        Dictionary with upcoming earnings dates
    """
    cache_key = f"earnings_{'_'.join(tickers)}"
    current_time = time.time()
    
    # Check if data is in cache and still fresh
    if cache_key in cache and current_time - cache[cache_key]["timestamp"] < CACHE_TIMEOUT:
        logger.info(f"Using cached earnings calendar")
        return cache[cache_key]["data"]
    
    logger.info(f"Fetching earnings calendar for {len(tickers)} tickers")
    try:
        result = {
            "tickers": tickers,
            "earnings": []
        }
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                calendar = stock.calendar
                
                if calendar is not None and not calendar.empty:
                    earnings_date = calendar.loc["Earnings Date", 0] if "Earnings Date" in calendar.index else None
                    
                    if earnings_date:
                        # Convert to datetime if it's a timestamp
                        if isinstance(earnings_date, (int, float)):
                            earnings_date = datetime.fromtimestamp(earnings_date)
                        
                        # Format the date
                        earnings_date_str = earnings_date.strftime("%Y-%m-%d") if isinstance(earnings_date, datetime) else str(earnings_date)
                        
                        result["earnings"].append({
                            "ticker": ticker,
                            "earnings_date": earnings_date_str,
                            "days_until": (earnings_date - datetime.now()).days if isinstance(earnings_date, datetime) else "Unknown"
                        })
            except Exception as e:
                logger.warning(f"Error fetching earnings for {ticker}: {str(e)}")
        
        # Sort by earnings date
        result["earnings"].sort(key=lambda x: x["earnings_date"])
        
        # Store in cache
        cache[cache_key] = {
            "timestamp": current_time,
            "data": result
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error fetching earnings calendar: {str(e)}")
        return {"error": str(e)}

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
        print(f'html url: {url}')
        
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

@mcp.tool()
async def analyze_technical_indicators(ticker: str, timeframe: str = "daily") -> Dict[str, Any]:
    """Analyze technical indicators for a specific stock.
    
    Args:
        ticker: The stock ticker symbol
        timeframe: Time period for analysis (daily, hourly, weekly)
        
    Returns:
        Dictionary with technical indicators and signals
    """
    logger.info(f"Analyzing technical indicators for {ticker} on {timeframe} timeframe")
    
    cache_key = f"tech_indicators_{ticker}_{timeframe}"
    current_time = time.time()
    
    # Check if data is in cache and still fresh
    if cache_key in cache and current_time - cache[cache_key]["timestamp"] < CACHE_TIMEOUT:
        logger.info(f"Using cached technical indicators for {ticker}")
        return cache[cache_key]["data"]
    
    try:
        # First get the basic stock data from Finviz
        stock_data = await get_stock_data(ticker)
        
        if "error" in stock_data:
            return stock_data
        
        # Initialize results dictionary
        results = {
            "ticker": ticker,
            "timeframe": timeframe,
            "price": stock_data.get("Price", "N/A"),
            "indicators": {},
            "signals": [],
            "overall_signal": "Neutral"
        }
        
        # Extract technical indicators from Finviz data
        indicators = {}
        
        # RSI
        if "RSI (14)" in stock_data and stock_data["RSI (14)"] != "-":
            rsi = float(stock_data["RSI (14)"])
            indicators["rsi"] = rsi
            
            # RSI signal
            if rsi < 30:
                results["signals"].append({
                    "indicator": "RSI",
                    "signal": "Buy",
                    "strength": "Strong",
                    "reason": f"RSI at {rsi} indicates oversold conditions"
                })
            elif rsi > 70:
                results["signals"].append({
                    "indicator": "RSI",
                    "signal": "Sell",
                    "strength": "Strong",
                    "reason": f"RSI at {rsi} indicates overbought conditions"
                })
            elif rsi < 40:
                results["signals"].append({
                    "indicator": "RSI",
                    "signal": "Buy",
                    "strength": "Weak",
                    "reason": f"RSI at {rsi} is approaching oversold territory"
                })
            elif rsi > 60:
                results["signals"].append({
                    "indicator": "RSI",
                    "signal": "Sell",
                    "strength": "Weak",
                    "reason": f"RSI at {rsi} is approaching overbought territory"
                })
        
        # Moving Averages
        ma_signals = []
        
        # SMA 20
        if "SMA20" in stock_data and stock_data["SMA20"] != "-":
            sma20 = float(stock_data["SMA20"].replace("%", ""))
            indicators["sma20"] = sma20
            
            if sma20 > 0:
                ma_signals.append({
                    "indicator": "SMA20",
                    "signal": "Buy",
                    "strength": "Medium",
                    "reason": f"Price is {abs(sma20):.2f}% above SMA20"
                })
            else:
                ma_signals.append({
                    "indicator": "SMA20",
                    "signal": "Sell",
                    "strength": "Medium",
                    "reason": f"Price is {abs(sma20):.2f}% below SMA20"
                })
        
        # SMA 50
        if "SMA50" in stock_data and stock_data["SMA50"] != "-":
            sma50 = float(stock_data["SMA50"].replace("%", ""))
            indicators["sma50"] = sma50
            
            if sma50 > 0:
                ma_signals.append({
                    "indicator": "SMA50",
                    "signal": "Buy",
                    "strength": "Medium",
                    "reason": f"Price is {abs(sma50):.2f}% above SMA50"
                })
            else:
                ma_signals.append({
                    "indicator": "SMA50",
                    "signal": "Sell",
                    "strength": "Medium",
                    "reason": f"Price is {abs(sma50):.2f}% below SMA50"
                })
        
        # SMA 200
        if "SMA200" in stock_data and stock_data["SMA200"] != "-":
            sma200 = float(stock_data["SMA200"].replace("%", ""))
            indicators["sma200"] = sma200
            
            if sma200 > 0:
                ma_signals.append({
                    "indicator": "SMA200",
                    "signal": "Buy",
                    "strength": "Strong",
                    "reason": f"Price is {abs(sma200):.2f}% above SMA200 (long-term uptrend)"
                })
            else:
                ma_signals.append({
                    "indicator": "SMA200",
                    "signal": "Sell",
                    "strength": "Strong",
                    "reason": f"Price is {abs(sma200):.2f}% below SMA200 (long-term downtrend)"
                })
        
        # Add MA signals to results
        results["signals"].extend(ma_signals)
        
        # MACD
        if "MACD" in stock_data and stock_data["MACD"] != "-":
            macd = stock_data["MACD"]
            indicators["macd"] = macd
            
            # Since Finviz doesn't provide detailed MACD components,
            # we'll just use the signal from Finviz
            if macd == "Bullish":
                results["signals"].append({
                    "indicator": "MACD",
                    "signal": "Buy",
                    "strength": "Medium",
                    "reason": "MACD shows bullish signal"
                })
            elif macd == "Bearish":
                results["signals"].append({
                    "indicator": "MACD",
                    "signal": "Sell",
                    "strength": "Medium",
                    "reason": "MACD shows bearish signal"
                })
        
        # Bollinger Bands
        if "Volatility" in stock_data and stock_data["Volatility"] != "-":
            volatility = float(stock_data["Volatility"].replace("%", ""))
            indicators["volatility"] = volatility
            
            # High volatility might indicate potential for Bollinger Band signals
            if volatility > 3:
                results["signals"].append({
                    "indicator": "Volatility",
                    "signal": "Neutral",
                    "strength": "Weak",
                    "reason": f"High volatility ({volatility}%) indicates potential for price swings"
                })
        
        # Volume
        if "Rel Volume" in stock_data and stock_data["Rel Volume"] != "-":
            rel_volume = float(stock_data["Rel Volume"])
            indicators["relative_volume"] = rel_volume
            
            if rel_volume > 2:
                results["signals"].append({
                    "indicator": "Volume",
                    "signal": "Strong Movement",
                    "strength": "Strong",
                    "reason": f"Volume is {rel_volume}x average, indicating strong interest"
                })
            elif rel_volume < 0.5:
                results["signals"].append({
                    "indicator": "Volume",
                    "signal": "Weak Movement",
                    "strength": "Weak",
                    "reason": f"Volume is only {rel_volume}x average, indicating low interest"
                })
        
        # Add all indicators to results
        results["indicators"] = indicators
        
        # Calculate overall signal based on the individual signals
        buy_signals = sum(1 for signal in results["signals"] if signal["signal"] == "Buy")
        sell_signals = sum(1 for signal in results["signals"] if signal["signal"] == "Sell")
        
        # Weight by strength
        strong_buy = sum(1 for signal in results["signals"] if signal["signal"] == "Buy" and signal["strength"] == "Strong")
        strong_sell = sum(1 for signal in results["signals"] if signal["signal"] == "Sell" and signal["strength"] == "Strong")
        
        # Calculate weighted score
        buy_score = buy_signals + strong_buy
        sell_score = sell_signals + strong_sell
        
        # Determine overall signal
        if buy_score > sell_score + 2:
            results["overall_signal"] = "Strong Buy"
        elif buy_score > sell_score:
            results["overall_signal"] = "Buy"
        elif sell_score > buy_score + 2:
            results["overall_signal"] = "Strong Sell"
        elif sell_score > buy_score:
            results["overall_signal"] = "Sell"
        else:
            results["overall_signal"] = "Neutral"
        
        # Store in cache
        cache[cache_key] = {
            "timestamp": current_time,
            "data": results
        }
        
        return results
        
    except Exception as e:
        logger.error(f"Error analyzing technical indicators for {ticker}: {str(e)}")
        return {"error": str(e)}

@mcp.tool()
async def scan_for_trading_opportunities(strategy: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """Scan the market for trading opportunities based on a strategy.
    
    Args:
        strategy: Trading strategy (momentum, value, breakout, etc.)
        max_results: Maximum number of results to return
        
    Returns:
        List of potential trading opportunities with analysis
    """
    logger.info(f"Scanning for trading opportunities using {strategy} strategy")
    
    cache_key = f"opportunities_{strategy.lower()}_{max_results}"
    current_time = time.time()
    
    # Check if data is in cache and still fresh
    if cache_key in cache and current_time - cache[cache_key]["timestamp"] < CACHE_TIMEOUT:
        logger.info(f"Using cached trading opportunities for {strategy} strategy")
        return cache[cache_key]["data"]
    
    try:
        opportunities = []
        
        # Define strategy-specific filters and criteria
        if strategy.lower() == "momentum":
            # Look for stocks with strong upward momentum
            filters = {
                "ta_perf_1w": "o10",  # Performance over 10% in the last week
                "ta_rsi_14": "ob40",  # RSI over 40 (showing strength)
                "sh_avgvol": "o100",  # Average volume over 100K
                "sh_price": "o5"      # Price over $5
            }
            
            # Get stocks matching the filters
            stocks = await get_screener_results(filters)
            
            # Analyze each stock for momentum indicators
            for stock in stocks[:max_results * 2]:  # Get more than needed for filtering
                if "Ticker" in stock:
                    ticker = stock["Ticker"]
                    
                    # Get technical indicators
                    tech_analysis = await analyze_technical_indicators(ticker)
                    
                    # Check for momentum criteria
                    if isinstance(tech_analysis, dict) and "overall_signal" in tech_analysis:
                        if tech_analysis["overall_signal"] in ["Buy", "Strong Buy"]:
                            # Add to opportunities with reason
                            opportunities.append({
                                "ticker": ticker,
                                "strategy": "momentum",
                                "price": stock.get("Price", "N/A"),
                                "signal": tech_analysis["overall_signal"],
                                "reasons": [signal["reason"] for signal in tech_analysis.get("signals", []) 
                                           if signal["signal"] == "Buy"],
                                "technical_data": tech_analysis["indicators"]
                            })
        
        elif strategy.lower() == "value":
            # Look for undervalued stocks
            filters = {
                "fa_pe_u": "15",      # P/E under 15
                "fa_pb_u": "2",       # P/B under 2
                "fa_debteq_u": "1",   # Debt/Equity under 1
                "fa_div_o": "1"       # Dividend yield over 1%
            }
            
            # Get stocks matching the filters
            stocks = await get_screener_results(filters)
            
            # Analyze each stock for value indicators
            for stock in stocks[:max_results * 2]:
                if "Ticker" in stock:
                    ticker = stock["Ticker"]
                    
                    # Get fundamental analysis
                    analysis = await get_stock_analysis(ticker)
                    
                    # Check for value criteria
                    if isinstance(analysis, dict) and "recommendation" in analysis:
                        if analysis["recommendation"] in ["Buy", "Hold"]:
                            # Add to opportunities with reason
                            opportunities.append({
                                "ticker": ticker,
                                "strategy": "value",
                                "price": stock.get("Price", "N/A"),
                                "signal": analysis["recommendation"],
                                "reasons": analysis.get("reasoning", []),
                                "fundamental_data": analysis.get("data_points", {})
                            })
        
        elif strategy.lower() == "breakout":
            # Look for stocks breaking out of resistance levels
            filters = {
                "ta_sma20_pa": "a",   # Price above SMA20
                "ta_sma50_pa": "a",   # Price above SMA50
                "ta_highlow52w_pb": "nh",  # Near 52-week high
                "sh_avgvol": "o200"   # Average volume over 200K
            }
            
            # Get stocks matching the filters
            stocks = await get_screener_results(filters)
            
            # Analyze each stock for breakout indicators
            for stock in stocks[:max_results * 2]:
                if "Ticker" in stock:
                    ticker = stock["Ticker"]
                    
                    # Get technical indicators
                    tech_analysis = await analyze_technical_indicators(ticker)
                    
                    # Check for breakout criteria
                    if isinstance(tech_analysis, dict) and "indicators" in tech_analysis:
                        indicators = tech_analysis["indicators"]
                        
                        # Check for high relative volume (breakout confirmation)
                        if "relative_volume" in indicators and indicators["relative_volume"] > 1.5:
                            # Add to opportunities with reason
                            opportunities.append({
                                "ticker": ticker,
                                "strategy": "breakout",
                                "price": stock.get("Price", "N/A"),
                                "signal": "Potential Breakout",
                                "reasons": [
                                    f"Volume is {indicators['relative_volume']}x average",
                                    "Price near 52-week high",
                                    "Trading above key moving averages"
                                ],
                                "technical_data": indicators
                            })
        
        elif strategy.lower() == "oversold":
            # Look for oversold stocks that might bounce
            filters = {
                "ta_rsi_14": "ob30",  # RSI under 30 (oversold)
                "sh_price": "o5",     # Price over $5
                "sh_avgvol": "o100"   # Average volume over 100K
            }
            
            # Get stocks matching the filters
            stocks = await get_screener_results(filters)
            
            # Analyze each stock for oversold indicators
            for stock in stocks[:max_results * 2]:
                if "Ticker" in stock:
                    ticker = stock["Ticker"]
                    
                    # Get technical indicators
                    tech_analysis = await analyze_technical_indicators(ticker)
                    
                    # Check for oversold criteria
                    if isinstance(tech_analysis, dict) and "indicators" in tech_analysis:
                        indicators = tech_analysis["indicators"]
                        
                        if "rsi" in indicators and indicators["rsi"] < 30:
                            # Add to opportunities with reason
                            opportunities.append({
                                "ticker": ticker,
                                "strategy": "oversold",
                                "price": stock.get("Price", "N/A"),
                                "signal": "Potential Bounce",
                                "reasons": [
                                    f"RSI at {indicators['rsi']} indicates oversold conditions",
                                    "Potential for mean reversion"
                                ],
                                "technical_data": indicators
                            })
        
        elif strategy.lower() == "earnings":
            # Look for stocks with upcoming earnings that might move
            filters = {
                "earningsdate": "thisweek",  # Earnings this week
                "sh_avgvol": "o500"          # Average volume over 500K
            }
            
            # Get stocks matching the filters
            stocks = await get_screener_results(filters)
            
            # Analyze each stock for earnings play potential
            for stock in stocks[:max_results * 2]:
                if "Ticker" in stock:
                    ticker = stock["Ticker"]
                    
                    # Get stock data
                    stock_data = await get_stock_data(ticker)
                    
                    if isinstance(stock_data, dict) and "Earnings" in stock_data:
                        earnings_date = stock_data["Earnings"]
                        
                        # Add to opportunities
                        opportunities.append({
                            "ticker": ticker,
                            "strategy": "earnings",
                            "price": stock.get("Price", "N/A"),
                            "signal": "Earnings Play",
                            "reasons": [
                                f"Earnings date: {earnings_date}",
                                "Potential for price movement around earnings announcement"
                            ],
                            "earnings_date": earnings_date
                        })
        
        else:
            return {"error": f"Unknown strategy: {strategy}. Available strategies: momentum, value, breakout, oversold, earnings"}
        
        # Limit results to max_results
        opportunities = opportunities[:max_results]
        
        # Store in cache
        cache[cache_key] = {
            "timestamp": current_time,
            "data": opportunities
        }
        
        return opportunities
        
    except Exception as e:
        logger.error(f"Error scanning for {strategy} opportunities: {str(e)}")
        return {"error": str(e)}

@mcp.tool()
async def place_order(
    ticker: str, 
    order_type: str,
    quantity: int,
    price: Optional[float] = None,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None
) -> Dict[str, Any]:
    """Place a trading order through connected broker API.
    
    Args:
        ticker: Stock ticker symbol
        order_type: Type of order (buy, sell, buy_call, buy_put)
        quantity: Number of shares or contracts
        price: Limit price (if applicable)
        stop_loss: Stop loss price
        take_profit: Take profit price
        
    Returns:
        Order confirmation details
    """
    # Implementation would connect to broker API and place order
    pass

@mcp.tool()
async def analyze_options_chain(ticker: str) -> Dict[str, Any]:
    """Analyze options chain for a specific stock.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Analysis of options including implied volatility, best opportunities
    """
    logger.info(f"Analyzing options chain for {ticker}")
    
    cache_key = f"options_chain_{ticker}"
    current_time = time.time()
    
    # Check if data is in cache and still fresh
    if cache_key in cache and current_time - cache[cache_key]["timestamp"] < CACHE_TIMEOUT:
        logger.info(f"Using cached options data for {ticker}")
        return cache[cache_key]["data"]
    
    try:
        # First get the basic stock data
        stock_data = await get_stock_data(ticker)
        
        if "error" in stock_data:
            return stock_data
        
        # Get current price
        current_price = 0
        if "Price" in stock_data:
            current_price = float(stock_data["Price"].replace("$", ""))
        
        # Fetch options data from Finviz
        url = f"{FINVIZ_URL}/options.ashx?t={ticker}"
        html_content = await make_request(url)
        
        if not html_content:
            return {"error": f"Could not fetch options data for {ticker}"}
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Initialize results
        results = {
            "ticker": ticker,
            "current_price": current_price,
            "implied_volatility": None,
            "call_options": [],
            "put_options": [],
            "opportunities": {
                "calls": [],
                "puts": []
            },
            "summary": {}
        }
        
        # Extract implied volatility if available
        iv_element = soup.find(text=lambda t: t and "Implied Volatility:" in t)
        if iv_element:
            iv_text = iv_element.strip()
            iv_value = iv_text.split("Implied Volatility:")[1].strip().split("%")[0]
            try:
                results["implied_volatility"] = float(iv_value)
            except ValueError:
                results["implied_volatility"] = None
        
        # Find options tables
        option_tables = soup.find_all('table', {'class': 'table-options'})
        
        # Process call options
        call_options = []
        put_options = []
        
        for table in option_tables:
            # Determine if this is calls or puts table
            table_type = None
            header = table.find_previous('h3')
            if header and "Call Options" in header.text:
                table_type = "call"
            elif header and "Put Options" in header.text:
                table_type = "put"
            else:
                continue
            
            # Get headers
            headers = []
            header_row = table.find('tr')
            if header_row:
                for th in header_row.find_all('td'):
                    headers.append(th.text.strip())
            
            # Process option rows
            rows = table.find_all('tr')[1:]  # Skip header row
            for row in rows:
                cells = row.find_all('td')
                if len(cells) >= len(headers):
                    option = {}
                    for i, header in enumerate(headers):
                        option[header] = cells[i].text.strip()
                    
                    # Add calculated fields
                    try:
                        option["strike"] = float(option.get("Strike", "0"))
                        option["last"] = float(option.get("Last", "0"))
                        option["bid"] = float(option.get("Bid", "0"))
                        option["ask"] = float(option.get("Ask", "0"))
                        option["volume"] = int(option.get("Volume", "0").replace(",", ""))
                        option["open_interest"] = int(option.get("Open Int", "0").replace(",", ""))
                        
                        # Calculate mid price
                        option["mid"] = (option["bid"] + option["ask"]) / 2
                        
                        # Calculate distance from current price (%)
                        if current_price > 0:
                            distance = ((option["strike"] - current_price) / current_price) * 100
                            option["distance_pct"] = round(distance, 2)
                        
                        # Calculate liquidity score (0-10)
                        liquidity = min(10, option["volume"] / 100) if option["volume"] > 0 else 0
                        option["liquidity_score"] = round(liquidity, 1)
                        
                    except (ValueError, KeyError):
                        # Skip calculated fields if data is missing
                        pass
                    
                    # Add to appropriate list
                    if table_type == "call":
                        call_options.append(option)
                    else:
                        put_options.append(option)
        
        # Sort options by strike price
        call_options.sort(key=lambda x: x.get("strike", 0))
        put_options.sort(key=lambda x: x.get("strike", 0))
        
        # Add to results
        results["call_options"] = call_options
        results["put_options"] = put_options
        
        # Find opportunities
        
        # For calls - look for high volume, near the money options
        for option in call_options:
            if "distance_pct" in option and "liquidity_score" in option:
                # Near the money (within 10%)
                if -5 <= option["distance_pct"] <= 15:
                    # Decent liquidity
                    if option["liquidity_score"] >= 3:
                        # Add as opportunity
                        opportunity = {
                            "strike": option["strike"],
                            "expiration": option.get("Expiration", "Unknown"),
                            "price": option["mid"],
                            "volume": option.get("volume", 0),
                            "open_interest": option.get("open_interest", 0),
                            "distance_pct": option["distance_pct"],
                            "liquidity_score": option["liquidity_score"],
                            "reason": "Near the money call with good liquidity"
                        }
                        
                        # Add additional reasons
                        if option.get("volume", 0) > option.get("open_interest", 0) * 0.2:
                            opportunity["reason"] += ", high relative volume"
                        
                        results["opportunities"]["calls"].append(opportunity)
        
        # For puts - look for high volume, near the money options
        for option in put_options:
            if "distance_pct" in option and "liquidity_score" in option:
                # Near the money (within 10%)
                if -15 <= option["distance_pct"] <= 5:
                    # Decent liquidity
                    if option["liquidity_score"] >= 3:
                        # Add as opportunity
                        opportunity = {
                            "strike": option["strike"],
                            "expiration": option.get("Expiration", "Unknown"),
                            "price": option["mid"],
                            "volume": option.get("volume", 0),
                            "open_interest": option.get("open_interest", 0),
                            "distance_pct": option["distance_pct"],
                            "liquidity_score": option["liquidity_score"],
                            "reason": "Near the money put with good liquidity"
                        }
                        
                        # Add additional reasons
                        if option.get("volume", 0) > option.get("open_interest", 0) * 0.2:
                            opportunity["reason"] += ", high relative volume"
                        
                        results["opportunities"]["puts"].append(opportunity)
        
        # Generate summary
        summary = {
            "total_calls": len(call_options),
            "total_puts": len(put_options),
            "call_put_ratio": len(call_options) / max(1, len(put_options)),
            "most_active_calls": sorted(call_options, key=lambda x: x.get("volume", 0), reverse=True)[:3],
            "most_active_puts": sorted(put_options, key=lambda x: x.get("volume", 0), reverse=True)[:3],
            "sentiment": "Neutral"
        }
        
        # Determine sentiment based on call/put ratio
        if summary["call_put_ratio"] > 1.5:
            summary["sentiment"] = "Bullish"
        elif summary["call_put_ratio"] < 0.7:
            summary["sentiment"] = "Bearish"
        
        results["summary"] = summary
        
        # Store in cache
        cache[cache_key] = {
            "timestamp": current_time,
            "data": results
        }
        
        return results
        
    except Exception as e:
        logger.error(f"Error analyzing options chain for {ticker}: {str(e)}")
        return {"error": str(e)}

@mcp.tool()
async def get_portfolio_status() -> Dict[str, Any]:
    """Get current portfolio status including positions and performance.
    
    Returns:
        Portfolio details including positions, values, and performance metrics
    """
    logger.info("Retrieving portfolio status")
    
    cache_key = "portfolio_status"
    current_time = time.time()
    
    # Check if data is in cache and still fresh (shorter timeout for portfolio data)
    if cache_key in cache and current_time - cache[cache_key]["timestamp"] < 300:  # 5 minutes
        logger.info("Using cached portfolio data")
        return cache[cache_key]["data"]
    
    try:
        # In a real implementation, this would connect to a brokerage API
        # For this demo, we'll simulate a portfolio with some positions
        
        # Simulated portfolio positions
        positions = [
            {"ticker": "AAPL", "quantity": 10, "entry_price": 150.00, "date_opened": "2023-01-15"},
            {"ticker": "MSFT", "quantity": 5, "entry_price": 280.00, "date_opened": "2023-02-10"},
            {"ticker": "GOOGL", "quantity": 8, "entry_price": 120.00, "date_opened": "2023-03-05"},
            {"ticker": "AMZN", "quantity": 12, "entry_price": 100.00, "date_opened": "2023-04-20"},
            {"ticker": "NVDA", "quantity": 15, "entry_price": 200.00, "date_opened": "2023-05-12"}
        ]
        
        # Get current prices for all positions
        total_value = 0
        total_cost = 0
        position_details = []
        
        for position in positions:
            ticker = position["ticker"]
            quantity = position["quantity"]
            entry_price = position["entry_price"]
            cost_basis = entry_price * quantity
            
            # Get current stock data
            stock_data = await get_stock_data(ticker)
            
            if "error" in stock_data:
                # If we can't get current data, use entry price as fallback
                current_price = entry_price
                logger.warning(f"Could not get current price for {ticker}, using entry price")
            else:
                # Extract current price
                current_price = float(stock_data.get("Price", "0").replace("$", ""))
            
            # Calculate position metrics
            market_value = current_price * quantity
            profit_loss = market_value - cost_basis
            profit_loss_pct = (profit_loss / cost_basis) * 100 if cost_basis > 0 else 0
            
            # Get technical indicators for position
            tech_analysis = await analyze_technical_indicators(ticker)
            
            # Add position details
            position_detail = {
                "ticker": ticker,
                "quantity": quantity,
                "entry_price": entry_price,
                "current_price": current_price,
                "cost_basis": cost_basis,
                "market_value": market_value,
                "profit_loss": profit_loss,
                "profit_loss_pct": profit_loss_pct,
                "date_opened": position["date_opened"],
                "days_held": (datetime.now() - datetime.strptime(position["date_opened"], "%Y-%m-%d")).days,
                "technical_signal": tech_analysis.get("overall_signal", "Neutral") if isinstance(tech_analysis, dict) else "Neutral"
            }
            
            position_details.append(position_detail)
            total_value += market_value
            total_cost += cost_basis
        
        # Calculate overall portfolio metrics
        total_profit_loss = total_value - total_cost
        total_profit_loss_pct = (total_profit_loss / total_cost) * 100 if total_cost > 0 else 0
        
        # Simulated cash balance and historical performance
        cash_balance = 25000.00
        total_portfolio_value = total_value + cash_balance
        
        # Simulated historical performance
        historical_performance = {
            "1d": 0.5,    # 0.5% daily change
            "1w": 1.2,    # 1.2% weekly change
            "1m": 2.8,    # 2.8% monthly change
            "3m": 5.3,    # 5.3% 3-month change
            "ytd": 8.7,   # 8.7% year-to-date change
            "1y": 12.4    # 12.4% 1-year change
        }
        
        # Sector allocation
        sector_allocation = {}
        for position in position_details:
            ticker = position["ticker"]
            stock_data = await get_stock_data(ticker)
            
            if isinstance(stock_data, dict) and "Sector" in stock_data:
                sector = stock_data["Sector"]
                if sector not in sector_allocation:
                    sector_allocation[sector] = 0
                sector_allocation[sector] += position["market_value"]
        
        # Convert sector allocation to percentages
        for sector in sector_allocation:
            sector_allocation[sector] = (sector_allocation[sector] / total_value) * 100
        
        # Risk metrics
        beta_weighted = 0
        for position in position_details:
            ticker = position["ticker"]
            stock_data = await get_stock_data(ticker)
            
            if isinstance(stock_data, dict) and "Beta" in stock_data and stock_data["Beta"] != "-":
                beta = float(stock_data["Beta"])
                weight = position["market_value"] / total_value
                beta_weighted += beta * weight
        
        # Assemble portfolio status
        portfolio_status = {
            "timestamp": datetime.now().isoformat(),
            "total_value": total_portfolio_value,
            "invested_value": total_value,
            "cash_balance": cash_balance,
            "cash_percentage": (cash_balance / total_portfolio_value) * 100,
            "total_profit_loss": total_profit_loss,
            "total_profit_loss_percentage": total_profit_loss_pct,
            "positions": position_details,
            "position_count": len(position_details),
            "sector_allocation": sector_allocation,
            "historical_performance": historical_performance,
            "risk_metrics": {
                "beta_weighted": beta_weighted,
                "sharpe_ratio": 1.2,  # Simulated Sharpe ratio
                "max_drawdown": -8.5  # Simulated maximum drawdown percentage
            },
            "recommendations": []
        }
        
        # Generate portfolio recommendations
        if cash_balance < 0.1 * total_portfolio_value:
            portfolio_status["recommendations"].append({
                "type": "warning",
                "message": "Cash balance is low (less than 10% of portfolio). Consider raising cash levels."
            })
        
        # Check for overconcentration
        for position in position_details:
            position_weight = position["market_value"] / total_value * 100
            if position_weight > 20:
                portfolio_status["recommendations"].append({
                    "type": "warning",
                    "message": f"Position in {position['ticker']} is {position_weight:.1f}% of portfolio, consider reducing for diversification."
                })
        
        # Check for positions with technical sell signals
        for position in position_details:
            if position["technical_signal"] in ["Sell", "Strong Sell"]:
                portfolio_status["recommendations"].append({
                    "type": "action",
                    "message": f"Consider selling {position['ticker']} based on {position['technical_signal']} technical signal."
                })
        
        # Store in cache
        cache[cache_key] = {
            "timestamp": current_time,
            "data": portfolio_status
        }
        
        return portfolio_status
        
    except Exception as e:
        logger.error(f"Error retrieving portfolio status: {str(e)}")
        return {"error": str(e)}

@mcp.tool()
async def backtest_strategy(
    strategy: str,
    tickers: List[str],
    start_date: str,
    end_date: str
) -> Dict[str, Any]:
    """Backtest a trading strategy on historical data.
    
    Args:
        strategy: Trading strategy to test
        tickers: List of stock tickers to test on
        start_date: Start date for backtest (YYYY-MM-DD)
        end_date: End date for backtest (YYYY-MM-DD)
        
    Returns:
        Backtest results including performance metrics
    """
    # Implementation would run backtest on historical data
    pass

@mcp.tool()
async def set_trading_parameters(
    risk_per_trade: float,
    max_positions: int,
    trading_hours: Dict[str, str]
) -> Dict[str, str]:
    """Configure trading parameters for the bot.
    
    Args:
        risk_per_trade: Maximum risk percentage per trade
        max_positions: Maximum number of concurrent positions
        trading_hours: Trading hours configuration
        
    Returns:
        Confirmation of settings
    """
    # Implementation would set trading parameters
    pass

@mcp.tool()
async def get_stocks_by_category(category: str) -> List[Dict[str, Any]]:
    """Get a list of stocks belonging to a specific category or industry.
    
    Args:
        category: The category or industry to search for (e.g., "Gold", "Rare Earth", "Semiconductors")
        
    Returns:
        A list of stocks in the specified category with their basic information.
    """
    cache_key = f"category_{category.lower().replace(' ', '_')}"
    current_time = time.time()
    
    # Check if data is in cache and still fresh
    if cache_key in cache and current_time - cache[cache_key]["timestamp"] < CACHE_TIMEOUT:
        logger.info(f"Using cached data for category: {category}")
        return cache[cache_key]["data"]
    
    logger.info(f"Fetching stocks for category: {category}")
    try:
        # Map common categories to Finviz industry/sector filters
        category_mapping = {
            "gold": {"industry": "gold"},
            "silver": {"industry": "silver"},
            "rare earth": {"industry": "nonmetallic mining"},
            "semiconductors": {"industry": "semiconductor"},
            "ai": {"industry": "computer software"},  # Approximate match
            "electric vehicles": {"industry": "auto manufacturers"},  # Approximate match
            "solar": {"industry": "solar"},
            "oil": {"industry": "oil & gas"},
            "banks": {"industry": "banks"},
            "biotech": {"industry": "biotechnology"},
            "cloud": {"industry": "software - application"},  # Approximate match
            "retail": {"sector": "consumer defensive", "industry": "retail"},
            "airlines": {"industry": "airlines"},
            "steel": {"industry": "steel"}
        }
        
        filters = {}
        category_lower = category.lower()
        
        # Try to find an exact match in our mapping
        if category_lower in category_mapping:
            for key, value in category_mapping[category_lower].items():
                if key == "industry":
                    filters["ind_industry"] = value
                elif key == "sector":
                    filters["sec_sector"] = value
        else:
            # If no exact match, try to use the category as an industry name
            filters["ind_industry"] = category_lower
        
        # Get screener results with these filters
        results = await get_screener_results(filters)
        
        # If no results found, try a keyword search in description
        if not results or (isinstance(results, dict) and "error" in results):
            logger.info(f"No direct industry match for {category}, trying keyword search")
            
            # Get all stocks and filter manually
            all_stocks = await get_screener_results({})
            if isinstance(all_stocks, list):
                filtered_results = []
                
                # Get detailed data for each stock and check description
                for stock in all_stocks[:30]:  # Limit to first 30 to avoid too many requests
                    if "Ticker" in stock:
                        ticker = stock["Ticker"]
                        stock_data = await get_stock_data(ticker)
                        
                        # Check if category keywords are in company description
                        if isinstance(stock_data, dict) and "Company" in stock_data:
                            description = stock_data.get("Company", "").lower()
                            if category_lower in description:
                                filtered_results.append(stock)
                
                results = filtered_results
        
        # Store in cache
        cache[cache_key] = {
            "timestamp": current_time,
            "data": results
        }
        
        return results
        
    except Exception as e:
        logger.error(f"Error fetching stocks for category {category}: {str(e)}")
        return {"error": str(e)}


if __name__ == "__main__":
    # Initialize and run the server with stdio transport for MCP
    mcp.run(transport='stdio')
    # After initializing the server
    print("Starting Finviz MCP server", file=sys.stderr)
    print(f"Python executable: {sys.executable}", file=sys.stderr)