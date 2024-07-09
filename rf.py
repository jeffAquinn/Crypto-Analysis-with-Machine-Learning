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
    
    # Convert numeric columns to appropriate data types
    numeric_columns = ['Price', 'Volume', 'Bid Liquidity', 'Ask Liquidity', 'Bid Ask Ratio', 'Long Ratio', 'Short Ratio']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def train_predict(df):
    # Fill NaN values generated from the conversion with 0 or another suitable value
    df.fillna(0, inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    
    # Features for training
    features = ['Price', 'Volume', 'Bid Liquidity', 'Ask Liquidity', 'Bid Ask Ratio', 'Long Ratio', 'Short Ratio']
    
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
    
    return price_pred[-1], direction_pred[-1]  # Return the last prediction as the next period's prediction

def update_google_sheets_with_predictions():
    for symbol, sheet_name in SHEET_NAMES.items():
        df = get_data(sheet_name)
        
        # Get predictions
        predicted_price, predicted_direction = train_predict(df)
        
        # Calculate Entry Price, Stop Loss, Take Profit
        current_price = df.iloc[-1]['Price']
        entry_price = (current_price + predicted_price) / 2  # Enhance entry price based on current and predicted price
        stop_loss_price = entry_price * 0.97  # 3% stop loss
        take_profit_price = entry_price * 1.09  # 9% take profit
        
        # Risk management
        account_balance = 1000  # Initial account balance
        risk_amount = account_balance * 0.02  # 2% risk
        
        # Append new predictions to the "Random Forest" Google Sheet
        worksheet_name = "Random Forest"
        worksheet = sh.worksheet(worksheet_name)
        
        # Ensure headers are set
        headers = ['Date', 'Sheet Name', 'Account Balance', 'Bullish or Bearish', 
                   'Prediction Price', 'Entry Price', 'Stop Loss Price', 'Take Profit Price']
        
        existing_headers = worksheet.row_values(1)
        
        if existing_headers != headers:
            worksheet.clear()
            worksheet.append_row(headers)
        
        # Insert new row with data below headers
        new_row = {
            'Date': datetime.now(pytz.timezone('US/Mountain')).strftime('%b/%d/%Y'),
            'Sheet Name': sheet_name,
            'Account Balance': account_balance,
            'Bullish or Bearish': 'Bullish' if predicted_direction == 1 else 'Bearish',
            'Prediction Price': predicted_price,
            'Entry Price': entry_price,
            'Stop Loss Price': stop_loss_price,
            'Take Profit Price': take_profit_price
        }
        worksheet.append_row(list(new_row.values()))
        
        # Format Bullish/Bearish cell
        cell = worksheet.find(new_row['Bullish or Bearish'])
        if new_row['Bullish or Bearish'] == 'Bullish':
            worksheet.format(cell.address, {'backgroundColor': {'red': 0, 'green': 1, 'blue': 0}})
        else:
            worksheet.format(cell.address, {'backgroundColor': {'red': 1, 'green': 0, 'blue': 0}})

# Function to be called from crypto_analysis.py
def run_rf():
    update_google_sheets_with_predictions()

if __name__ == "__main__":
    run_rf()
