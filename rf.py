import ccxt
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import numpy as np
from datetime import datetime
import pytz
import schedule
import time
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

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

# Initialize Random Forest models
clf = RandomForestClassifier(n_estimators=100, random_state=42)
reg = RandomForestRegressor(n_estimators=100, random_state=42)

# Account details
initial_balance = 1000
risk_tolerance = 0.02  # 2%
stop_loss_pct = 0.03  # 3%
take_profit_pct = 0.09  # 9%

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

                all_data[symbol].append({
                    'exchange': exchange_name,
                    'symbol': symbol,
                    'price': price,
                    'volume': data['ticker']['baseVolume'],
                    'bid_liquidity': liquidity_analysis['bid_liquidity'],
                    'ask_liquidity': liquidity_analysis['ask_liquidity'],
                    'bid_ask_ratio': liquidity_analysis['bid_ask_ratio'],
                    'long_ratio': leverage_analysis['long_ratio'],
                    'short_ratio': leverage_analysis['short_ratio']
                })

    dfs = {symbol: pd.DataFrame(data) for symbol, data in all_data.items()}
    for symbol, df in dfs.items():
        df.columns = [
            'Exchange', 'Symbol', 'Price', 'Volume', 'Bid Liquidity', 'Ask Liquidity',
            'Bid Ask Ratio', 'Long Ratio', 'Short Ratio'
        ]
    return dfs

def train_and_predict(df, symbol):
    if df.empty:
        return None, None

    df['Price_Change'] = df['Price'].diff().shift(-1)
    df['Bullish_Bearish'] = (df['Price_Change'] > 0).astype(int)

    features = ['Price', 'Volume', 'Bid Liquidity', 'Ask Liquidity', 'Bid Ask Ratio', 'Long Ratio', 'Short Ratio']
    df = df.dropna()

    if len(df) < 2:
        return None, None

    X = df[features]
    y_classification = df['Bullish_Bearish']
    y_regression = df['Price'].shift(-1).dropna()

    clf.fit(X[:-1], y_classification[:-1])
    reg.fit(X[:-1], y_regression)

    X_latest = X.iloc[-1].values.reshape(1, -1)
    bullish_bearish_pred = clf.predict(X_latest)[0]
    price_pred = reg.predict(X_latest)[0]

    return bullish_bearish_pred, price_pred

def update_google_sheets():
    dfs = fetch_and_analyze_data()

    # Replace NaN and infinite values with suitable defaults
    for symbol, df in dfs.items():
        df.replace([np.nan, np.inf, -np.inf], 0, inplace=True)

    # Add current date in the format "Jul/7/2024"
    mountain_time = datetime.now(pytz.timezone('US/Mountain'))
    date_str = mountain_time.strftime('%b/%d/%Y')

    try:
        worksheet = sh.worksheet('Random Forest')
        existing_data = worksheet.get_all_values()

        # Initialize account balance
        if existing_data:
            account_balance = float(existing_data[-1][5])
        else:
            account_balance = initial_balance

        for symbol, sheet_name in TRADING_PAIRS.items():
            df = dfs[symbol]

            bullish_bearish_pred, price_pred = train_and_predict(df, symbol)

            if bullish_bearish_pred is None or price_pred is None:
                continue

            bullish_bearish = 'Bullish' if bullish_bearish_pred == 1 else 'Bearish'
            entry_price = df['Price'].iloc[-1]
            stop_loss_price = entry_price * (1 - stop_loss_pct)
            take_profit_price = entry_price * (1 + take_profit_pct)

            # Update account balance based on prediction
            if bullish_bearish_pred == 1 and price_pred >= take_profit_price:
                account_balance *= (1 + take_profit_pct)
            elif bullish_bearish_pred == 0 and price_pred <= stop_loss_price:
                account_balance *= (1 - stop_loss_pct)

            new_row = [
                initial_balance,
                bullish_bearish,
                price_pred,
                entry_price,
                stop_loss_price,
                take_profit_price,
                account_balance
            ]

            headers = ['Starting Balance', 'Bullish/Bearish', 'Prediction Price', 'Entry Price', 'Stop Loss Price', 'Take Profit Price', 'Account Balance']
            updated_data = [headers] + existing_data + [new_row]

            worksheet.clear()
            worksheet.update(updated_data)

            # Color coding for Bullish/Bearish
            cell_range = worksheet.range(f'B{len(existing_data) + 2}:B{len(existing_data) + 2}')
            for cell in cell_range:
                cell.value = bullish_bearish
                cell.color = (0.0, 1.0, 0.0, 1.0) if bullish_bearish == 'Bullish' else (1.0, 0.0, 0.0, 1.0)
            worksheet.update_cells(cell_range)
            
            print(f"Data for {symbol} updated successfully.")
    except Exception as e:
        print(f"Error updating Google Sheets: {str(e)}")

def run_scheduler():
    update_google_sheets()  # Initial run

    # Schedule the update every 4 hours
    schedule.every(4).hours.do(update_google_sheets)

    while True:
        schedule.run_pending()
        time.sleep(60)  # Sleep for 60 seconds

if __name__ == "__main__":
    run_scheduler()
