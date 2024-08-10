import ccxt
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import numpy as np
from datetime import datetime
import pytz
import schedule
import time
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)

# Google Sheets credentials
SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_FILE = 'service_account.json'
SPREADSHEET_NAME = 'CCXT-DATA'  # Update with your Google Sheet name

# Initialize Google Sheets client
try:
    credentials = ServiceAccountCredentials.from_json_keyfile_name(SERVICE_ACCOUNT_FILE, SCOPES)
    gc = gspread.authorize(credentials)
    sh = gc.open(SPREADSHEET_NAME)
    logging.info(f"Successfully connected to Google Sheets: {SPREADSHEET_NAME}")
except Exception as e:
    logging.error(f"Failed to connect to Google Sheets: {str(e)}")
    sys.exit(1)

# CCXT exchanges
exchanges = {
    'Binance': ccxt.binance(),
    'Bybit': ccxt.bybit(),
    'HitBTC': ccxt.hitbtc(),
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
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=100)  # Fetch 100 hours of OHLCV data

        return {
            'symbol': symbol,
            'L2 Orderbook': l2_orderbook,
            'Trades': trades,
            'Ticker': ticker,
            'OHLCV': ohlcv
        }
    except Exception as e:
        logging.error(f"Error fetching data for {symbol} on {exchange.name}: {str(e)}")
        return None

def calculate_vwap(df):
    df['Cumulative_TPV'] = (df['Close Price'] * df['Volume']).cumsum()
    df['Cumulative_Volume'] = df['Volume'].cumsum()
    df['VWAP'] = df['Cumulative_TPV'] / df['Cumulative_Volume']
    return df['VWAP'].iloc[-1] if not df['VWAP'].empty else np.nan

def calculate_wavetrend(df, channel_length=9, average_length=12, wt_ma_length=3):
    hlc3 = (df['High Price'] + df['Low Price'] + df['Close Price']) / 3
    esa = hlc3.ewm(span=channel_length, adjust=False).mean()
    de = abs(hlc3 - esa).ewm(span=channel_length, adjust=False).mean()
    ci = (hlc3 - esa) / (0.015 * de)
    tci = ci.ewm(span=average_length, adjust=False).mean()
    wt1 = tci
    wt2 = wt1.rolling(window=wt_ma_length).mean()
    return wt1.iloc[-1] if not wt1.empty else np.nan, wt2.iloc[-1] if not wt2.empty else np.nan

def calculate_mfi(df, period=14):
    typical_price = (df['High Price'] + df['Low Price'] + df['Close Price']) / 3
    money_flow = typical_price * df['Volume']
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
    positive_mf = positive_flow.rolling(window=period).sum()
    negative_mf = negative_flow.rolling(window=period).sum()
    mfi = 100 * (positive_mf / (positive_mf + negative_mf))
    return mfi.iloc[-1] if not mfi.empty else np.nan

def analyze_liquidity(l2_orderbook, price):
    bids = l2_orderbook['bids']
    asks = l2_orderbook['asks']

    bid_liquidity = sum(bid[1] for bid in bids)
    ask_liquidity = sum(ask[1] for ask in asks)
    bid_ask_spread = asks[0][0] - bids[0][0] if bids and asks else None
    market_depth = bid_liquidity + ask_liquidity

    return {
        'bid_liquidity': bid_liquidity,
        'ask_liquidity': ask_liquidity,
        'bid_ask_ratio': bid_liquidity / ask_liquidity if ask_liquidity else float('inf'),
        'bid_liquidity_usd': bid_liquidity * price,
        'ask_liquidity_usd': ask_liquidity * price,
        'bid_ask_spread': bid_ask_spread,
        'market_depth': market_depth
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
                # Convert OHLCV data to DataFrame
                ohlcv_df = pd.DataFrame(data['OHLCV'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                ohlcv_df['timestamp'] = pd.to_datetime(ohlcv_df['timestamp'], unit='ms')
                ohlcv_df.set_index('timestamp', inplace=True)

                # Calculate additional columns
                ohlcv_df.rename(columns={'close': 'Close Price', 'high': 'High Price', 'low': 'Low Price', 'volume': 'Volume'}, inplace=True)
                ohlcv_df['Price'] = ohlcv_df['Close Price']  # Assuming close price as the last price

                vwap = calculate_vwap(ohlcv_df)
                wavetrend1, wavetrend2 = calculate_wavetrend(ohlcv_df)
                mfi = calculate_mfi(ohlcv_df)

                # Fetch the latest ticker data for the most recent row
                ticker = data['Ticker']
                price = ticker['last']
                open_price = ticker['open']
                close_price = ticker['close']
                high_price = ticker['high']
                low_price = ticker['low']

                liquidity_analysis = analyze_liquidity(data['L2 Orderbook'], price)
                leverage_analysis = analyze_leverage_ratio(data['Trades'])

                # Create DataFrame with all required data including indicators
                all_data[symbol].append({
                    'Exchange': exchange_name,
                    'Symbol': symbol,
                    'Price': price,
                    'Open Price': open_price,
                    'Close Price': close_price,
                    'High Price': high_price,
                    'Low Price': low_price,
                    'Volume': ticker['baseVolume'],
                    'Bid Liquidity USD': liquidity_analysis['bid_liquidity_usd'],
                    'Ask Liquidity USD': liquidity_analysis['ask_liquidity_usd'],
                    'Long Ratio': leverage_analysis['long_ratio'],
                    'Short Ratio': leverage_analysis['short_ratio'],
                    'Market Depth': liquidity_analysis['market_depth'],
                    'Bid Ask Spread': liquidity_analysis['bid_ask_spread'],
                    'Bid Ask Ratio': liquidity_analysis['bid_ask_ratio'],
                    'L2 Bid 1 USD': data['L2 Orderbook']['bids'][0][0] * data['L2 Orderbook']['bids'][0][1] if len(data['L2 Orderbook']['bids']) > 0 else 0,
                    'L2 Bid 2 USD': data['L2 Orderbook']['bids'][1][0] * data['L2 Orderbook']['bids'][1][1] if len(data['L2 Orderbook']['bids']) > 1 else 0,
                    'L2 Bid 3 USD': data['L2 Orderbook']['bids'][2][0] * data['L2 Orderbook']['bids'][2][1] if len(data['L2 Orderbook']['bids']) > 2 else 0,
                    'L2 Bid 4 USD': data['L2 Orderbook']['bids'][3][0] * data['L2 Orderbook']['bids'][3][1] if len(data['L2 Orderbook']['bids']) > 3 else 0,
                    'L2 Bid 5 USD': data['L2 Orderbook']['bids'][4][0] * data['L2 Orderbook']['bids'][4][1] if len(data['L2 Orderbook']['bids']) > 4 else 0,
                    'L2 Ask 1 USD': data['L2 Orderbook']['asks'][0][0] * data['L2 Orderbook']['asks'][0][1] if len(data['L2 Orderbook']['asks']) > 0 else 0,
                    'L2 Ask 2 USD': data['L2 Orderbook']['asks'][1][0] * data['L2 Orderbook']['asks'][1][1] if len(data['L2 Orderbook']['asks']) > 1 else 0,
                    'L2 Ask 3 USD': data['L2 Orderbook']['asks'][2][0] * data['L2 Orderbook']['asks'][2][1] if len(data['L2 Orderbook']['asks']) > 2 else 0,
                    'L2 Ask 4 USD': data['L2 Orderbook']['asks'][3][0] * data['L2 Orderbook']['asks'][3][1] if len(data['L2 Orderbook']['asks']) > 3 else 0,
                    'L2 Ask 5 USD': data['L2 Orderbook']['asks'][4][0] * data['L2 Orderbook']['asks'][4][1] if len(data['L2 Orderbook']['asks']) > 4 else 0,
                    'VWAP': vwap,
                    'WaveTrend1': wavetrend1,
                    'WaveTrend2': wavetrend2,
                    'MFI': mfi
                })

    dfs = {symbol: pd.DataFrame(data) for symbol, data in all_data.items()}
    for symbol, df in dfs.items():
        df.columns = [
            'Exchange', 'Symbol', 'Price', 'Open Price', 'Close Price', 'High Price', 'Low Price', 'Volume',
            'Bid Liquidity USD', 'Ask Liquidity USD', 'Long Ratio', 'Short Ratio',
            'Market Depth', 'Bid Ask Spread', 'Bid Ask Ratio', 'L2 Bid 1 USD', 'L2 Bid 2 USD', 'L2 Bid 3 USD',
            'L2 Bid 4 USD', 'L2 Bid 5 USD', 'L2 Ask 1 USD', 'L2 Ask 2 USD', 'L2 Ask 3 USD', 'L2 Ask 4 USD', 'L2 Ask 5 USD',
            'VWAP', 'WaveTrend1', 'WaveTrend2', 'MFI'
        ]
    return dfs

def update_google_sheets():
    try:
        dfs = fetch_and_analyze_data()
    except Exception as e:
        logging.error(f"Error fetching and analyzing data: {str(e)}")
        return

    # Replace NaN and infinite values with suitable defaults
    for symbol, df in dfs.items():
        df.replace([np.nan, np.inf, -np.inf], 0, inplace=True)

    # Add current date in the format "Jul/7/2024"
    mountain_time = datetime.now(pytz.timezone('US/Mountain'))
    date_str = mountain_time.strftime('%b/%d/%Y')

    for symbol, sheet_name in TRADING_PAIRS.items():
        df = dfs[symbol]

        try:
            worksheet = sh.worksheet(sheet_name)
        except gspread.exceptions.WorksheetNotFound:
            logging.warning(f"Worksheet '{sheet_name}' not found. Creating it.")
            worksheet = sh.add_worksheet(title=sheet_name, rows="1000", cols="50")

        try:
            existing_data = worksheet.get_all_values()
            headers = ['Date'] + df.columns.values.tolist()

            new_data = [[date_str] + row.tolist() for _, row in df.iterrows()]

            # Prepare data to update
            updated_data = [headers] + new_data + existing_data[1:]

            # Update worksheet with the new data
            worksheet.clear()
            worksheet.update(updated_data)
            logging.info(f"Data for {symbol} updated successfully.")
        except Exception as e:
            logging.error(f"Error updating {symbol} sheet: {str(e)}")

def run_scheduler():
    logging.info("Starting initial update...")
    update_google_sheets()  # Initial run
    logging.info("Initial update completed.")
    
    # Schedule the update every 30 minutes
    schedule.every(30).minutes.do(update_google_sheets)

    while True:
        now = datetime.now(pytz.timezone('US/Mountain'))
        if now.strftime('%H:%M') == '07:00':
            logging.info("Running daily update at 07:00 Mountain Time.")
            update_google_sheets()
        schedule.run_pending()
        time.sleep(60)  # Sleep for 60 seconds

if __name__ == "__main__":
    logging.info("Script started.")
    try:
        run_scheduler()
    except Exception as e:
        logging.error(f"Unexpected error in main script execution: {str(e)}")
        sys.exit(1)
