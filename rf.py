import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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
SHEET_NAMES = {
    'BTC/USDT': 'BTC',
    'SOL/USDT': 'SOL',
    'ATOM/USDT': 'ATOM'
}

def get_data(sheet_name):
    worksheet = sh.worksheet(sheet_name)
    data = worksheet.get_all_values()
    
    # Extract the header (first row) for column names
    header = data[0]
    
    # Extract the data starting from the second row
    data = data[1:]
    
    # Convert to DataFrame
    df = pd.DataFrame(data, columns=header)
    
    # Specify the relevant columns
    relevant_columns = [
        'Date', 'Price', 'Open Price', 'Close Price', 'High Price', 'Low Price', 'Volume', 
        'Bid Ask Spread', 'Market Depth', 
        'Bid Ask Ratio', 'Bid Liquidity USD', 'Ask Liquidity USD', 
        'Long Ratio', 'Short Ratio', 'L2 Bid 1 USD', 'L2 Bid 2 USD', 'L2 Bid 3 USD', 
        'L2 Bid 4 USD', 'L2 Bid 5 USD', 'L2 Ask 1 USD', 'L2 Ask 2 USD', 'L2 Ask 3 USD', 
        'L2 Ask 4 USD', 'L2 Ask 5 USD'
    ]
    
    # Filter the DataFrame to keep only the relevant columns
    df = df[relevant_columns]
    
    # Convert numeric columns to appropriate data types
    numeric_columns = [
        'Price', 'Open Price', 'Close Price', 'High Price', 'Low Price', 'Volume', 
        'Bid Ask Spread', 'Market Depth', 
        'Bid Ask Ratio', 'Bid Liquidity USD', 'Ask Liquidity USD', 
        'Long Ratio', 'Short Ratio', 'L2 Bid 1 USD', 'L2 Bid 2 USD', 'L2 Bid 3 USD', 
        'L2 Bid 4 USD', 'L2 Bid 5 USD', 'L2 Ask 1 USD', 'L2 Ask 2 USD', 'L2 Ask 3 USD', 
        'L2 Ask 4 USD', 'L2 Ask 5 USD'
    ]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col].str.replace(',', ''), errors='coerce')
    
    # Handle NaN values in the Price column by forward filling
    df['Price'] = df['Price'].ffill()
    
    return df

def train_predict(df):
    # Fill NaN values generated from the conversion with 0 or another suitable value
    df.fillna(0, inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    
    # Features for training
    features = [
        'Price', 'Open Price', 'Close Price', 'High Price', 'Low Price', 'Volume', 
        'Bid Liquidity USD', 'Ask Liquidity USD', 'Bid Ask Spread', 'Market Depth', 
        'Bid Ask Ratio', 'Long Ratio', 'Short Ratio', 
        'L2 Bid 1 USD', 'L2 Bid 2 USD', 'L2 Bid 3 USD', 
        'L2 Bid 4 USD', 'L2 Bid 5 USD', 'L2 Ask 1 USD', 'L2 Ask 2 USD', 'L2 Ask 3 USD', 
        'L2 Ask 4 USD', 'L2 Ask 5 USD'
    ]
    
    X = df[features]
    y_price = df['Price'].shift(-1).fillna(0)  # Next period price for regression
    y_direction = (y_price > df['Price']).astype(int)  # 1 if price goes up, 0 if price goes down
    
    X_train, X_test, y_train_price, y_test_price = train_test_split(X, y_price, test_size=0.2, random_state=42)
    _, _, y_train_direction, y_test_direction = train_test_split(X, y_direction, test_size=0.2, random_state=42)
    
    # Train Random Forest for price prediction
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regressor.fit(X_train, y_train_price)
    price_pred = rf_regressor.predict(X_test)
    
    # Train Random Forest for direction prediction
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train_direction)
    direction_pred = rf_classifier.predict(X_test)
    
    # Evaluate accuracy
    direction_accuracy = accuracy_score(y_test_direction, direction_pred)
    print(f'Direction Prediction Accuracy: {direction_accuracy:.2f}')
    
    return price_pred[-1], direction_pred[-1], direction_accuracy  # Return the last prediction and accuracy

def calculate_entry_stop_loss_take_profit(current_price, predicted_price):
    # Long Entry
    long_entry = (current_price + predicted_price) / 2
    long_stop_loss = long_entry * 0.98
    long_take_profit = long_entry * 1.09
    
    # Short Entry
    short_entry = (current_price + predicted_price) / 2
    short_stop_loss = short_entry * 1.02
    short_take_profit = short_entry * 0.91
    
    return long_entry, long_stop_loss, long_take_profit, short_entry, short_stop_loss, short_take_profit

def execute_trade(account_balance, direction, current_price):
    if direction == 1:  # Bullish (Long)
        entry_price = current_price
        stop_loss_price = entry_price * 0.98
        take_profit_price = entry_price * 1.09
        risk = 0.02 * account_balance
        reward = 0.09 * account_balance
    else:  # Bearish (Short)
        entry_price = current_price
        stop_loss_price = entry_price * 1.02
        take_profit_price = entry_price * 0.91
        risk = 0.02 * account_balance
        reward = 0.09 * account_balance

    return entry_price, stop_loss_price, take_profit_price, risk, reward

def update_google_sheets_with_predictions():
    # Get the current date in the format "Jul/7/2024"
    mountain_time = datetime.now(pytz.timezone('US/Mountain'))
    date_str = mountain_time.strftime('%b/%d/%Y')
    
    try:
        worksheet_name = "Random Forest"
        worksheet = sh.worksheet(worksheet_name)
        existing_data = worksheet.get_all_values()
        headers = [
            'Date', 'Sheet Name', 'Account Balance', 'Trade Type', 'Trade Outcome', 'Profit/Loss',
            'Entry Price', 'Stop Loss Price', 'Take Profit Price', 'Long Trades Taken', 'Short Trades Taken'
        ]
        
        account_balance = 1000  # Initial balance
        
        new_rows = []
        for symbol, sheet_name in SHEET_NAMES.items():
            df = get_data(sheet_name)
            current_price = df.iloc[-1]['Price']
            predicted_price, predicted_direction, direction_accuracy = train_predict(df)

            # Execute trade based on predictions
            entry_price, stop_loss_price, take_profit_price, risk, reward = execute_trade(
                account_balance, predicted_direction, current_price
            )

            if predicted_direction == 1:  # Long trade
                trade_type = 'Long'
                trade_outcome = 'Profit' if current_price >= take_profit_price else 'Loss'
                profit_loss = reward if trade_outcome == 'Profit' else -risk
                account_balance += profit_loss
                long_trades_taken = profit_loss if profit_loss > 0 else 0
                short_trades_taken = 0
            else:  # Short trade
                trade_type = 'Short'
                trade_outcome = 'Profit' if current_price <= take_profit_price else 'Loss'
                profit_loss = reward if trade_outcome == 'Profit' else -risk
                account_balance += profit_loss
                long_trades_taken = 0
                short_trades_taken = -profit_loss if profit_loss < 0 else 0
            
            new_row = [
                date_str, sheet_name, account_balance, trade_type, trade_outcome, profit_loss,
                entry_price, stop_loss_price, take_profit_price, long_trades_taken, short_trades_taken
            ]
            new_rows.append(new_row)
        
        updated_data = [headers] + new_rows + existing_data[1:]
        worksheet.clear()
        worksheet.update(updated_data)
        
        print(f"Data updated successfully.")
        
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

# Function to be called from crypto_analysis.py
def run_rf():
    update_google_sheets_with_predictions()
    apply_formatting()

if __name__ == "__main__":
    run_rf()
