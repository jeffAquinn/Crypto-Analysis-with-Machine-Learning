import ccxt
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import numpy as np

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
    'binance': ccxt.binance(),
    'bybit': ccxt.bybit(),
    'hitbtc': ccxt.hitbtc(),
    'coinbaseadvanced' : ccxt.coinbaseadvanced()
}

# Trading pairs
TRADING_PAIRS = ['BTC/USDT', 'SOL/USDT', 'ATOM/USDT']

def fetch_market_data(exchange, symbol):
    try:
        ticker = exchange.fetch_ticker(symbol)
        orderbook = exchange.fetch_order_book(symbol)
        trades = exchange.fetch_trades(symbol)

        return {
            'symbol': symbol,
            'orderbook': orderbook,
            'trades': trades,
            'ticker': ticker
        }
    except Exception as e:
        print(f"Error fetching data for {symbol} on {exchange.name}: {str(e)}")
        return None

def analyze_liquidity(orderbook, price):
    bids = orderbook['bids']
    asks = orderbook['asks']

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
    all_data = []

    for exchange_name, exchange in exchanges.items():
        for symbol in TRADING_PAIRS:
            data = fetch_market_data(exchange, symbol)
            if data:
                price = data['ticker']['last']
                liquidity_analysis = analyze_liquidity(data['orderbook'], price)
                leverage_analysis = analyze_leverage_ratio(data['trades'])

                all_data.append({
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
                    'short_ratio': leverage_analysis['short_ratio']
                })

    df = pd.DataFrame(all_data)
    df.columns = [
        'Exchange', 'Symbol', 'Price', 'Volume', 'Bid Liquidity', 'Ask Liquidity', 
        'Bid Ask Ratio', 'Bid Liquidity USD', 'Ask Liquidity USD', 'Long Ratio', 'Short Ratio'
    ]
    return df

def update_google_sheets():
    df = fetch_and_analyze_data()

    # Replace NaN and infinite values with suitable defaults
    df.replace([np.nan, np.inf, -np.inf], 0, inplace=True)

    try:
        worksheet = sh.get_worksheet(0)  # Assuming the first sheet is where you want to update data
        worksheet.clear()  # Clear existing data
        worksheet.update([df.columns.values.tolist()] + df.values.tolist())
        print("Data updated successfully.")
    except Exception as e:
        print(f"Error updating Google Sheets: {str(e)}")

if __name__ == "__main__":
    update_google_sheets()
