import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from datetime import datetime
import pytz
import logging
import time

# Configuration
SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_FILE = 'service_account.json'
SPREADSHEET_NAME = 'CCXT-DATA'
DATA_SHEET_NAME = 'BTC'
RESULT_SHEET_NAME = 'Random Forest'
INITIAL_ACCOUNT_BALANCE = 1000.0

# Initialize Google Sheets client
logging.basicConfig(level=logging.INFO)
credentials = ServiceAccountCredentials.from_json_keyfile_name(SERVICE_ACCOUNT_FILE, SCOPES)
gc = gspread.authorize(credentials)
sh = gc.open(SPREADSHEET_NAME)

def fetch_data():
    """Fetch and preprocess data from Google Sheets."""
    try:
        worksheet = sh.worksheet(DATA_SHEET_NAME)
        records = worksheet.get_all_records()
        if not records:
            raise ValueError("No records found in the data sheet.")
        df = pd.DataFrame(records)
        df = preprocess_data(df)
        logging.info("Data fetched and preprocessed successfully.")
        return df
    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def preprocess_data(df):
    """Preprocess data for model training."""
    relevant_columns = [
        'Date', 'Price', 'Open Price', 'Close Price', 'High Price', 'Low Price', 'Volume',
        'Bid Ask Spread', 'Market Depth', 'Bid Ask Ratio', 'Bid Liquidity USD', 'Ask Liquidity USD',
        'Long Ratio', 'Short Ratio', 'L2 Bid 1 USD', 'L2 Bid 2 USD', 'L2 Bid 3 USD',
        'L2 Bid 4 USD', 'L2 Bid 5 USD', 'L2 Ask 1 USD', 'L2 Ask 2 USD', 'L2 Ask 3 USD',
        'L2 Ask 4 USD', 'L2 Ask 5 USD'
    ]
    df = df[relevant_columns].apply(pd.to_numeric, errors='coerce')
    df['Price'] = df['Price'].fillna(method='ffill')
    df = calculate_vwap(df)
    df = calculate_wavetrend(df)
    df = calculate_mfi(df)
    return df

def calculate_vwap(df):
    """Calculate VWAP."""
    df['Cumulative_TPV'] = (df['Price'] * df['Volume']).cumsum()
    df['Cumulative_Volume'] = df['Volume'].cumsum()
    df['VWAP'] = df['Cumulative_TPV'] / df['Cumulative_Volume']
    return df

def calculate_wavetrend(df, channel_length=9, average_length=12, wt_ma_length=3):
    """Calculate WaveTrend Oscillator."""
    hlc3 = (df['High Price'] + df['Low Price'] + df['Close Price']) / 3
    esa = hlc3.ewm(span=channel_length, adjust=False).mean()
    de = abs(hlc3 - esa).ewm(span=channel_length, adjust=False).mean()
    ci = (hlc3 - esa) / (0.015 * de)
    tci = ci.ewm(span=average_length, adjust=False).mean()
    df['WaveTrend1'] = tci
    df['WaveTrend2'] = tci.rolling(window=wt_ma_length).mean()
    return df

def calculate_mfi(df, period=14):
    """Calculate MFI."""
    typical_price = (df['High Price'] + df['Low Price'] + df['Close Price']) / 3
    money_flow = typical_price * df['Volume']
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
    positive_mf = positive_flow.rolling(window=period).sum()
    negative_mf = negative_flow.rolling(window=period).sum()
    df['MFI'] = 100 * (positive_mf / (positive_mf + negative_mf))
    return df

def train_and_predict(df):
    """Train models and predict price and direction."""
    df.fillna(0, inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    
    features = df.columns.difference(['Price'])
    X = df[features]
    y_price = df['Price'].shift(-1).fillna(0)
    y_direction = (y_price > df['Price']).astype(int)
    
    if len(X) < 10:
        logging.info("Insufficient data for training.")
        return None, None, 0
    
    X_train, X_test, y_train_price, y_test_price = train_test_split(X, y_price, test_size=0.2, random_state=42)
    _, _, y_train_direction, y_test_direction = train_test_split(X, y_direction, test_size=0.2, random_state=42)
    
    # Random Forest Regressor
    rf_regressor = GridSearchCV(RandomForestRegressor(random_state=42), {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }, cv=3)
    rf_regressor.fit(X_train, y_train_price)
    price_pred = rf_regressor.predict(X_test)
    
    # Random Forest Classifier
    rf_classifier = GridSearchCV(RandomForestClassifier(random_state=42), {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }, cv=3)
    rf_classifier.fit(X_train, y_train_direction)
    direction_pred = rf_classifier.predict(X_test)
    
    accuracy = accuracy_score(y_test_direction, direction_pred)
    logging.info(f'Direction Prediction Accuracy: {accuracy:.2f}')
    
    return price_pred[-1], direction_pred[-1], accuracy

def calculate_trade_parameters(current_price, predicted_price):
    """Calculate trade parameters."""
    long_entry = (current_price + predicted_price) / 2
    long_stop_loss = long_entry * 0.98
    long_take_profit = long_entry * 1.09
    short_entry = (current_price + predicted_price) / 2
    short_stop_loss = short_entry * 1.02
    short_take_profit = short_entry * 0.91
    return long_entry, long_stop_loss, long_take_profit, short_entry, short_stop_loss, short_take_profit

def evaluate_trade(account_balance, direction, trade_params, df):
    """Evaluate trade based on market conditions."""
    def get_last_mfi(df):
        return df['MFI'].iloc[-2] if len(df) > 1 else None

    latest_values = {
        'VWAP': df['VWAP'].iloc[-1],
        'WaveTrend1': df['WaveTrend1'].iloc[-1],
        'WaveTrend2': df['WaveTrend2'].iloc[-1],
        'MFI': df['MFI'].iloc[-1],
        'LastMFI': get_last_mfi(df)
    }

    trade_direction = 0
    if (latest_values['VWAP'] > -0.8 and latest_values['WaveTrend1'] > latest_values['WaveTrend2'] and 
        latest_values['MFI'] > latest_values['LastMFI']):
        trade_direction = 1
    elif (latest_values['VWAP'] < 0.8 and latest_values['WaveTrend1'] < latest_values['WaveTrend2'] and 
          latest_values['MFI'] < latest_values['LastMFI']):
        trade_direction = -1
    
    if direction == trade_direction:
        entry_price, stop_loss_price, take_profit_price = trade_params
        risk = 0.02 * account_balance
        reward = 0.09 * account_balance
        current_price = df['Price'].iloc[-1]
        trade_outcome, profit_loss = 'No Trade', 0
        
        if entry_price and stop_loss_price and take_profit_price:
            if direction == 1 and current_price >= take_profit_price:
                trade_outcome, profit_loss = 'Win', reward
            elif direction == -1 and current_price <= take_profit_price:
                trade_outcome, profit_loss = 'Win', reward
            elif direction == 1 and current_price <= stop_loss_price:
                trade_outcome, profit_loss = 'Loss', -risk
            elif direction == -1 and current_price >= stop_loss_price:
                trade_outcome, profit_loss = 'Loss', -risk
            else:
                trade_outcome = 'Open'
            
            account_balance += profit_loss
        
        return entry_price, stop_loss_price, take_profit_price, risk, reward, trade_outcome, profit_loss
    else:
        return None, None, None, 0, 0, 'No Trade', 0

def update_results(trade_data):
    """Update trade results in the 'Random Forest' worksheet."""
    try:
        worksheet = sh.worksheet(RESULT_SHEET_NAME)
        existing_data = worksheet.get_all_values()
        headers = [
            'Date', 'Asset', 'Account Balance', 'Trade Type', 'Trade Outcome', 'Profit/Loss',
            'Predicted Price', 'Entry Price', 'Stop Loss Price', 'Take Profit Price'
        ]
        updated_data = [headers] + [trade_data] + existing_data[1:]
        worksheet.clear()
        worksheet.update('A1', updated_data)
        logging.info("Trade results updated successfully.")
    except Exception as e:
        logging.error(f"Error updating results: {e}")

def run_rf():
    """Main function to run the trading logic."""
    logging.info("Starting trading logic...")
    while True:
        df = fetch_data()
        if df.empty:
            logging.error("No data available. Retrying in 1 hour...")
            time.sleep(3600)
            continue

        current_price = df['Price'].iloc[-1]
        predicted_price, predicted_direction, accuracy = train_and_predict(df)

        if predicted_price is None or predicted_direction is None:
            logging.info("Insufficient data for prediction. Retrying in 1 hour...")
            time.sleep(3600)
            continue
        
        trade_params = calculate_trade_parameters(current_price, predicted_price)
        trade_data = {
            'Date': datetime.now(pytz.timezone('US/Mountain')).strftime('%b/%d/%Y'),
            'Asset': 'BTC',
            'Account Balance': INITIAL_ACCOUNT_BALANCE,
            'Trade Type': 'Long' if predicted_direction == 1 else 'Short',
            'Trade Outcome': 'No Trade',
            'Profit/Loss': 0,
            'Predicted Price': predicted_price,
            'Entry Price': trade_params[0],
            'Stop Loss Price': trade_params[1],
            'Take Profit Price': trade_params[2]
        }
        
        entry_price, stop_loss_price, take_profit_price, risk, reward, trade_outcome, profit_loss = evaluate_trade(
            INITIAL_ACCOUNT_BALANCE, predicted_direction, trade_params, df
        )
        
        trade_data.update({
            'Trade Outcome': trade_outcome,
            'Profit/Loss': profit_loss
        })
        
        update_results(list(trade_data.values()))

        logging.info("Waiting for next update cycle...")
        time.sleep(3600)

if __name__ == "__main__":
    run_rf()
