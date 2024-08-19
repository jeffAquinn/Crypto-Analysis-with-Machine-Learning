import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, mean_squared_error
from datetime import datetime
import pytz

# Google Sheets credentials
SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_FILE = 'service_account.json'
SPREADSHEET_NAME = 'CCXT-DATA'  # Update with your Google Sheet name

# Initialize Google Sheets client
credentials = ServiceAccountCredentials.from_json_keyfile_name(SERVICE_ACCOUNT_FILE, SCOPES)
gc = gspread.authorize(credentials)
sh = gc.open(SPREADSHEET_NAME)

# Sheets corresponding to each trading pair
SHEET_NAME = 'BTC'  # Only BTC sheet is used

# Trade status tracking dictionary
trade_status = {SHEET_NAME: False}

# Initialize account balance
account_balance = 1000  # Set an initial value for account balance

def get_data(sheet_name):
    worksheet = sh.worksheet(sheet_name)
    data = worksheet.get_all_values()
    
    if not data:
        return pd.DataFrame()  # Return empty DataFrame if no data found
    
    header = data[0]
    data = data[1:]
    df = pd.DataFrame(data, columns=header)
    
    relevant_columns = [
        'Date', 'Price', 'Open Price', 'Close Price', 'High Price', 'Low Price', 'Volume', 
        'Bid Ask Spread', 'Market Depth', 
        'Bid Ask Ratio', 'Bid Liquidity USD', 'Ask Liquidity USD', 
        'Long Ratio', 'Short Ratio', 'L2 Bid 1 USD', 'L2 Bid 2 USD', 'L2 Bid 3 USD', 
        'L2 Bid 4 USD', 'L2 Bid 5 USD', 'L2 Ask 1 USD', 'L2 Ask 2 USD', 'L2 Ask 3 USD', 
        'L2 Ask 4 USD', 'L2 Ask 5 USD', 'VWAP', 'WaveTrend1', 'WaveTrend2', 'MFI'
    ]
    
    if not set(relevant_columns).issubset(df.columns):
        print(f"Missing columns in {sheet_name}")
        return pd.DataFrame()
    
    df = df[relevant_columns]
    
    numeric_columns = [
        'Price', 'Open Price', 'Close Price', 'High Price', 'Low Price', 'Volume', 
        'Bid Ask Spread', 'Market Depth', 
        'Bid Ask Ratio', 'Bid Liquidity USD', 'Ask Liquidity USD', 
        'Long Ratio', 'Short Ratio', 'L2 Bid 1 USD', 'L2 Bid 2 USD', 'L2 Bid 3 USD', 
        'L2 Bid 4 USD', 'L2 Bid 5 USD', 'L2 Ask 1 USD', 'L2 Ask 2 USD', 'L2 Ask 3 USD', 
        'L2 Ask 4 USD', 'L2 Ask 5 USD', 'VWAP', 'WaveTrend1', 'WaveTrend2', 'MFI'
    ]
    
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col].str.replace(',', ''), errors='coerce')
    
    df['Price'] = df['Price'].ffill()
    
    return df

def train_predict(df):
    if df.empty:
        return None, None, 0  # Return default values for empty DataFrame
    
    df.fillna(0, inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    
    features = [
        'Price', 'Open Price', 'Close Price', 'High Price', 'Low Price', 'Volume', 
        'Bid Liquidity USD', 'Ask Liquidity USD', 'Bid Ask Spread', 'Market Depth', 
        'Bid Ask Ratio', 'Long Ratio', 'Short Ratio', 
        'L2 Bid 1 USD', 'L2 Bid 2 USD', 'L2 Bid 3 USD', 
        'L2 Bid 4 USD', 'L2 Bid 5 USD', 'L2 Ask 1 USD', 'L2 Ask 2 USD', 'L2 Ask 3 USD', 
        'L2 Ask 4 USD', 'L2 Ask 5 USD', 'VWAP', 'WaveTrend1', 'WaveTrend2', 'MFI'
    ]
    
    X = df[features]
    y_price = df['Price'].shift(-1).fillna(0)
    y_direction = (y_price > df['Price']).astype(int)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    price_preds = []
    direction_preds = []
    direction_accuracies = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train_price, y_test_price = y_price.iloc[train_index], y_price.iloc[test_index]
        y_train_direction, y_test_direction = y_direction.iloc[train_index], y_direction.iloc[test_index]
        
        rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_regressor.fit(X_train, y_train_price)
        price_pred = rf_regressor.predict(X_test)
        price_preds.extend(price_pred)
        
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_classifier.fit(X_train, y_train_direction)
        direction_pred = rf_classifier.predict(X_test)
        direction_preds.extend(direction_pred)
        
        direction_accuracy = accuracy_score(y_test_direction, direction_pred)
        direction_accuracies.append(direction_accuracy)
    
    final_price_pred = np.mean(price_preds)
    final_direction_pred = np.round(np.mean(direction_preds)).astype(int)
    final_direction_accuracy = np.mean(direction_accuracies)
    
    print(f'Direction Prediction Accuracy: {final_direction_accuracy:.2f}')
    
    return final_price_pred, final_direction_pred, final_direction_accuracy

def calculate_entry_stop_loss_take_profit(current_price, predicted_price):
    long_entry = (current_price + predicted_price) / 2
    long_stop_loss = long_entry * 0.98
    long_take_profit = long_entry * 1.09
    
    short_entry = (current_price + predicted_price) / 2
    short_stop_loss = short_entry * 1.02
    short_take_profit = short_entry * 0.91
    
    return long_entry, long_stop_loss, long_take_profit, short_entry, short_stop_loss, short_take_profit

# Your existing imports and code remain unchanged

def check_long_trade_conditions(df):
    # Check MFI increasing over the last three days
    mfi_increasing = df['MFI'].iloc[-3:].is_monotonic_increasing
    vwap_increasing = df['VWAP'].iloc[-2:].is_monotonic_increasing
    wavetrend_increasing = df['WaveTrend1'].iloc[-2:].is_monotonic_increasing and df['WaveTrend2'].iloc[-2:].is_monotonic_increasing
    
    return mfi_increasing and vwap_increasing and wavetrend_increasing

def check_short_trade_conditions(df):
    # Check MFI decreasing over the last three days
    mfi_decreasing = df['MFI'].iloc[-3:].is_monotonic_decreasing
    vwap_decreasing = df['VWAP'].iloc[-2:].is_monotonic_decreasing
    wavetrend_decreasing = df['WaveTrend1'].iloc[-2:].is_monotonic_decreasing and df['WaveTrend2'].iloc[-2:].is_monotonic_decreasing
    
    return mfi_decreasing and vwap_decreasing and wavetrend_decreasing

def execute_trade(account_balance, df, sheet_name):
    latest_price = df['Price'].iloc[-1]
    yesterday_close = df['Close Price'].iloc[-2]
    yesterday_high = df['High Price'].iloc[-2]
    yesterday_low = df['Low Price'].iloc[-2]
    
    trade_outcome = 'No Trade'
    trade_direction = 0
    first_entry_price = second_entry_price = None
    
    # Check long trade conditions
    if check_long_trade_conditions(df) and not trade_status[sheet_name]:
        trade_direction = 1
        first_entry_price = yesterday_close
        second_entry_price = yesterday_low
    # Check short trade conditions
    elif check_short_trade_conditions(df) and not trade_status[sheet_name]:
        trade_direction = -1
        first_entry_price = yesterday_close
        second_entry_price = yesterday_high
    
    if trade_direction != 0:
        return execute_single_trade(account_balance, latest_price, first_entry_price, second_entry_price, trade_direction, sheet_name)
    
    return None, None, None, 0, 0, trade_outcome, 0, account_balance

def execute_single_trade(account_balance, current_price, first_entry_price, second_entry_price, trade_direction, sheet_name):
    global trade_status
    
    risk = 0.02 * account_balance  # Risk 2% of the account balance
    reward = 0.09 * account_balance  # Target 9% profit
    profit_loss = 0
    trade_outcome = 'Open'
    
    if trade_direction == 1:  # Long trade
        stop_loss_price = min(first_entry_price, second_entry_price) * 0.98
        take_profit_price = max(first_entry_price, second_entry_price) * 1.09
        if current_price >= take_profit_price:
            trade_outcome = 'Profit'
            profit_loss = reward
            trade_status[sheet_name] = False
        elif current_price <= stop_loss_price:
            trade_outcome = 'Loss'
            profit_loss = -risk
            trade_status[sheet_name] = False
    elif trade_direction == -1:  # Short trade
        stop_loss_price = max(first_entry_price, second_entry_price) * 1.02
        take_profit_price = min(first_entry_price, second_entry_price) * 0.91
        if current_price <= take_profit_price:
            trade_outcome = 'Profit'
            profit_loss = reward
            trade_status[sheet_name] = False
        elif current_price >= stop_loss_price:
            trade_outcome = 'Loss'
            profit_loss = -risk
            trade_status[sheet_name] = False
    
    account_balance += profit_loss
    
    # Log the trade decision
    print(f"Trade {'Long' if trade_direction == 1 else 'Short'} | First Entry: {first_entry_price} | Second Entry: {second_entry_price} | Stop: {stop_loss_price} | Take Profit: {take_profit_price} | Outcome: {trade_outcome} | P/L: {profit_loss}")
    
    return first_entry_price, second_entry_price, stop_loss_price, risk, reward, trade_outcome, profit_loss, account_balance

def update_google_sheets_with_predictions():
    mountain_time = datetime.now(pytz.timezone('US/Mountain'))
    date_str = mountain_time.strftime('%b/%d/%Y')
    
    global account_balance
    
    try:
        worksheet_name = "Random Forest"
        worksheet = sh.worksheet(worksheet_name)
        existing_data = worksheet.get_all_values()

        headers = [
            'Date', 'Sheet Name', 'Account Balance', 'Trade Type', 'Trade Outcome', 'Profit/Loss',
            'Predicted Price', 'Entry Price', 'Stop Loss Price', 'Take Profit Price'
        ]

        if not existing_data or existing_data[0] != headers:
            worksheet.insert_row(headers, 1)

        new_rows = []
        df = get_data(SHEET_NAME)
        if df.empty:
            print(f"No data for {SHEET_NAME}")
        else:
            current_price = df.iloc[-1]['Price']
            predicted_price, predicted_direction, direction_accuracy = train_predict(df)

            entry_price, stop_loss_price, take_profit_price, risk, reward, trade_outcome, profit_loss, account_balance = execute_trade(
                account_balance, df, SHEET_NAME
            )

            trade_type = 'Long' if predicted_direction == 1 else 'Short'

            new_row = [
                date_str, SHEET_NAME, account_balance, trade_type, trade_outcome, profit_loss,
                predicted_price, entry_price, stop_loss_price, take_profit_price
            ]
            new_rows.append(new_row)

        if new_rows:
            for row in new_rows:
                worksheet.insert_row(row, 2)
            print("Data updated successfully.")
        else:
            print("No new data to update.")

    except Exception as e:
        print(f"Error updating Google Sheets: {str(e)}")


def format_cell_range(sheet, cell_range, cell_format):
    body = {
        "requests": [
            {
                "repeatCell": {
                    "range": {
                        "sheetId": sheet.id,
                        "startRowIndex": cell_range['startRowIndex'],
                        "endRowIndex": cell_range['endRowIndex'],
                        "startColumnIndex": cell_range['startColumnIndex'],
                        "endColumnIndex": cell_range['endColumnIndex']
                    },
                    "cell": {
                        "userEnteredFormat": cell_format
                    },
                    "fields": "userEnteredFormat.textFormat.foregroundColor"
                }
            }
        ]
    }
    sh.batch_update(body)

def apply_formatting():
    worksheet_name = "Random Forest"
    worksheet = sh.worksheet(worksheet_name)
    cells = worksheet.range('D2:D')

    for cell in cells:
        if cell.value == 'Long':
            cell_format = {
                "textFormat": {
                    "foregroundColor": {
                        "red": 0.14,
                        "green": 0.47,
                        "blue": 0.0
                    }
                }
            }
            format_cell_range(worksheet, {
                "startRowIndex": cell.row - 1,
                "endRowIndex": cell.row,
                "startColumnIndex": 3,
                "endColumnIndex": 4
            }, cell_format)
        elif cell.value == 'Short':
            cell_format = {
                "textFormat": {
                    "foregroundColor": {
                        "red": 0.6,
                        "green": 0.0,
                        "blue": 0.0
                    }
                }
            }
            format_cell_range(worksheet, {
                "startRowIndex": cell.row - 1,
                "endRowIndex": cell.row,
                "startColumnIndex": 3,
                "endColumnIndex": 4
            }, cell_format)

def run_rf():
    update_google_sheets_with_predictions()
    apply_formatting()

if __name__ == "__main__":
    run_rf()
