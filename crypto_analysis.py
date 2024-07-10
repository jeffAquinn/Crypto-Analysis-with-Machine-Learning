# crypto_analysis.py
import ccxt
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import numpy as np
from datetime import datetime
import pytz
import schedule
import time
from rf import run_rf  # Import the function from rf.py

# Google Sheets credentials
SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_FILE = 'service_account.json'
SPREADSHEET_NAME = 'CCXT-DATA'  # Update with your Google Sheet name

# Initialize Google Sheets client
credentials = ServiceAccountCredentials.from_json_keyfile_name(SERVICE_ACCOUNT_FILE, SCOPES)
gc = gspread.authorize(credentials)
sh = gc.open(SPREADSHEET_NAME)

# CCXT exchanges
exchanges = {
    'Binance': ccxt.binance(),
    'Bybit': ccxt.bybit(),
    'HitBTC': ccxt.hitbtc(),
    'Coinbase': ccxt.coinbaseadvanced()
}

# Trading pairs
TRADING_PAIRS = {
    'BTC/USDT': 'BTC',
    'SOL/USDT': 'SOL',
    'ATOM/USDT': 'ATOM'
}

def fetch_market_data(exchange, symbol):
    try:
        ticker = exchange.fetch_ticker(symbol)
        l2_orderbook = exchange.fetch_l2_order_book(symbol)
        trades = exchange.fetch_trades(symbol)

        return {
            'symbol': symbol,
            'l2_orderbook': l2_orderbook,
            'trades': trades,
            'ticker': ticker
        }
    except Exception as e:
        print(f"Error fetching data for {symbol} on {exchange.name}: {str(e)}")
        return None

def analyze_liquidity(l2_orderbook, price):
    bids = l2_orderbook['bids']
    asks = l2_orderbook['asks']

    bid_liquidity = sum(bid[1] for bid in bids)
    ask_liquidity = sum(ask[1] for ask in asks)

    return {
        'bid_liquidity': bid_liquidity,
        'ask_liquidity': ask_liquidity,
        'bid_ask_ratio': bid_liquidity / ask_liquidity if ask_liquidity else float('inf'),
        'bid_liquidity_usd': bid_liquidity * price,
        'ask_liquidity_usd': ask_liquidity * price
    }

def analyze_leverage_ratio(trades):
    longs = sum(trade['amount'] for trade in trades if trade['side'] == 'buy')
    shorts = sum(trade['amount'] for trade in trades if trade['side'] == 'sell')

    return {
        'long_ratio': longs / (longs + shorts) if (longs + shorts) else 0,
        'short_ratio': shorts / (longs + shorts) if (longs + shorts) else 0
    }

def fetch_and_analyze_data():
    all_data = {symbol: [] for symbol in TRADING_PAIRS.keys()}

    for exchange_name, exchange in exchanges.items():
        for symbol in TRADING_PAIRS.keys():
            data = fetch_market_data(exchange, symbol)
            if data:
                price = data['ticker']['last']
                liquidity_analysis = analyze_liquidity(data['l2_orderbook'], price)
                leverage_analysis = analyze_leverage_ratio(data['trades'])

                # Adding L2 order book data converted to USD
                l2_bids_usd = [bid[0] * bid[1] for bid in data['l2_orderbook']['bids'][:5]]
                l2_asks_usd = [ask[0] * ask[1] for ask in data['l2_orderbook']['asks'][:5]]

                # Ensure there are 5 levels, fill missing with 0
                while len(l2_bids_usd) < 5:
                    l2_bids_usd.append(0)
                while len(l2_asks_usd) < 5:
                    l2_asks_usd.append(0)

                all_data[symbol].append({
                    'exchange': exchange_name,
                    'symbol': symbol,
                    'price': price,
                    'volume': data['ticker']['baseVolume'],
                    'bid_liquidity': liquidity_analysis['bid_liquidity'],
                    'ask_liquidity': liquidity_analysis['ask_liquidity'],
                    'bid_ask_ratio': liquidity_analysis['bid_ask_ratio'],
                    'bid_liquidity_usd': liquidity_analysis['bid_liquidity_usd'],
                    'ask_liquidity_usd': liquidity_analysis['ask_liquidity_usd'],
                    'long_ratio': leverage_analysis['long_ratio'],
                    'short_ratio': leverage_analysis['short_ratio'],
                    'L2 Bid 1 USD': l2_bids_usd[0],
                    'L2 Bid 2 USD': l2_bids_usd[1],
                    'L2 Bid 3 USD': l2_bids_usd[2],
                    'L2 Bid 4 USD': l2_bids_usd[3],
                    'L2 Bid 5 USD': l2_bids_usd[4],
                    'L2 Ask 1 USD': l2_asks_usd[0],
                    'L2 Ask 2 USD': l2_asks_usd[1],
                    'L2 Ask 3 USD': l2_asks_usd[2],
                    'L2 Ask 4 USD': l2_asks_usd[3],
                    'L2 Ask 5 USD': l2_asks_usd[4]
                })

    dfs = {symbol: pd.DataFrame(data) for symbol, data in all_data.items()}
    for symbol, df in dfs.items():
        df.columns = [
            'Exchange', 'Symbol', 'Price', 'Volume', 'Bid Liquidity', 'Ask Liquidity',
            'Bid Ask Ratio', 'Bid Liquidity USD', 'Ask Liquidity USD', 'Long Ratio', 'Short Ratio',
            'L2 Bid 1 USD', 'L2 Bid 2 USD', 'L2 Bid 3 USD', 'L2 Bid 4 USD', 'L2 Bid 5 USD',
            'L2 Ask 1 USD', 'L2 Ask 2 USD', 'L2 Ask 3 USD', 'L2 Ask 4 USD', 'L2 Ask 5 USD'
        ]
    return dfs

def update_google_sheets():
    dfs = fetch_and_analyze_data()

    # Replace NaN and infinite values with suitable defaults
    for symbol, df in dfs.items():
        df.replace([np.nan, np.inf, -np.inf], 0, inplace=True)

    # Add current date in the format "Jul/7/2024"
    mountain_time = datetime.now(pytz.timezone('US/Mountain'))
    date_str = mountain_time.strftime('%b/%d/%Y')

    try:
        for symbol, sheet_name in TRADING_PAIRS.items():
            df = dfs[symbol]

            worksheet = sh.worksheet(sheet_name)
            existing_data = worksheet.get_all_values()
            headers = ['Date'] + df.columns.values.tolist()
            
            new_data = [[date_str] + row for row in df.values.tolist()]

            # Prepare data to update
            updated_data = [headers] + new_data + existing_data[1:]

            # Update worksheet with the new data
            worksheet.clear()
            worksheet.update(updated_data)
            print(f"Data for {symbol} updated successfully.")
    except Exception as e:
        print(f"Error updating Google Sheets: {str(e)}")

def run_scheduler():
    update_google_sheets()  # Initial run
    run_rf()  # Run rf.py logic after updating Google Sheets

    # Schedule the update every hour
    schedule.every(1).hours.do(update_google_sheets)
    schedule.every(1).hours.do(run_rf)  # Schedule rf.py logic to run after updating Google Sheets

    while True:
        # Get current time in US/Mountain timezone
        now = datetime.now(pytz.timezone('US/Mountain'))
        if now.strftime('%H:%M') == '07:00':
            update_google_sheets()
            run_rf()  # Run rf.py logic after updating Google Sheets
        schedule.run_pending()
        time.sleep(60)  # Sleep for 60 seconds

if __name__ == "__main__":
    run_scheduler()
