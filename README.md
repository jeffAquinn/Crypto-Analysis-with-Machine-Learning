# Crypto Market Analysis Project

## Overview

This project fetches and analyzes cryptocurrency market data from top exchanges using the CCXT library. It collects information on trading pairs, calculates liquidity metrics, and determines long/short ratios. The data is then processed, analyzed using machine learning techniques, and uploaded to a Google Sheet for easy visualization and further analysis.

## ** You will have to run a VPN if you can't KYC with the exchanges! **
## Features

- Fetches market data from multiple exchanges (Binance, Bybit, HitBTC)
- Analyzes trading pairs (BTC/USDT, SOL/USDT, ATOM/USDT)
- Calculates liquidity metrics (bid/ask liquidity, bid/ask ratio)
- Determines long/short ratios based on recent trades
- Uses Random Forest algorithm for price prediction and market direction
- Simulates trades based on predictions
- Uploads processed data and predictions to Google Sheets

## Requirements

-ccxt
-pandas
-gspread
-oauth2client
-numpy
-schedule
-pytz
-scikit-learn
-Flask
-requests

## Setup

1. Clone the repository:
git clone https://github.com/your-username/crypto-market-analysis.git
cd crypto-market-analysis
Copy
2. Create a virtual environment (optional but recommended):
python -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate
Copy
3. Install the required packages:
pip install ccxt pandas gspread oauth2client numpy scikit-learn pytz
Copy
4. Set up Google Sheets API:
a. Go to the Google Cloud Console (https://console.cloud.google.com/)
b. Create a new project or select an existing one
c. Enable the Google Sheets API for your project
d. Create a service account:
   - Go to "IAM & Admin" > "Service Accounts"
   - Click "Create Service Account"
   - Fill in the details and grant it the "Editor" role
e. Create a key for the service account:
   - Select the service account you just created
   - Go to the "Keys" tab
   - Click "Add Key" > "Create new key"
   - Choose JSON as the key type and download the key file
f. Rename the downloaded JSON key file to `service_account.json` and place it in the project root directory

5. Create a new Google Sheet:
a. Go to Google Sheets (https://sheets.google.com)
b. Create a new spreadsheet
c. Rename it to "CCXT-DATA" (or choose your own name)
d. Create worksheets named "BTC", "SOL", "ATOM", and "Random Forest"
e. Share the spreadsheet with the email address of the service account (found in the `service_account.json` file)

6. Update the configuration:
- Open `crypto_analysis.py` and `rf.py`
- Update the `SPREADSHEET_NAME` variable with your Google Sheet name if you chose a different name

7. Set up a VPN (if necessary):
If you're unable to complete KYC with the exchanges, you may need to use a VPN to access their APIs. Set up a VPN service of your choice and ensure it's running before executing the script.

## Usage

1. Ensure your virtual environment is activated (if you created one).

2. Run the main script:
python crypto_analysis.py
Copy
This script will:
- Fetch data from the specified exchanges
- Process and analyze the data
- Update the designated Google Sheet with the collected data
- Run the Random Forest analysis (by calling `rf.py`)
- Update the Google Sheet with predictions and simulated trade results

3. The script is set to run every 6 hours. You can modify the schedule in the `run_scheduler()` function of `crypto_analysis.py` if needed.

4. To run the Random Forest analysis separately:
python rf.py
Copy
This will update the "Random Forest" worksheet with the latest predictions and simulated trade results.

## Understanding the Output

- The "BTC", "SOL", and "ATOM" worksheets contain the raw data collected from the exchanges.
- The "Random Forest" worksheet contains:
- Date of the prediction
- Sheet name (corresponding to the cryptocurrency)
- Account balance after the simulated trade
- Trade type (Long or Short)
- Trade outcome (Profit or Loss)
- Profit/Loss amount
- Entry price
- Stop loss price
- Take profit price

## Customization

- To add or remove exchanges, modify the `exchanges` dictionary in `crypto_analysis.py`.
- To change the trading pairs, update the `TRADING_PAIRS` dictionary in `crypto_analysis.py` and `SHEET_NAMES` in `rf.py`.
- Adjust the Random Forest parameters in the `train_predict()` function of `rf.py`.
- Modify the trade execution logic in the `execute_trade()` function of `rf.py`.

## Disclaimer

This project is for educational and research purposes only. It is not financial advice, and you should not use it to make real trading decisions without thorough testing and understanding of the risks involved.

## Contributing

Contributions to improve the project or add new features are welcome. Please submit a pull request or open an issue to discuss proposed changes.

## License

[MIT License](https://opensource.org/licenses/MIT)