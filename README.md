Crypto Market Analysis Project
Overview
This project fetches and analyzes cryptocurrency market data from top exchanges using the CCXT library. It collects information on trading pairs, calculates liquidity metrics, and determines long/short ratios. The data is then processed, analyzed using machine learning techniques, and uploaded to a Google Sheet for easy visualization and further analysis.

If you find this code helpful and you want to give back please send crypto to the following address. Thanks for any support and much love.
BTC: 3LvkjKVkSfLCKfxWPdsBokXnmsBhRrtmFk
XRP: rUwrkngQTjKhrBf9FdC1wGTBsbgC8yhweK *No Memo*
ETH: 0x4f18f6ead4d8c22a2e757257ad647b5d2d110cd4

You will have to run a VPN if you can't KYC with the exchanges!
Features
Fetches market data from multiple exchanges (Binance, Bybit, HitBTC)
Analyzes trading pairs (BTC/USDT, SOL/USDT, ATOM/USDT)
Calculates liquidity metrics (bid/ask liquidity, bid/ask ratio)
Determines long/short ratios based on recent trades
Uses Random Forest algorithm for price prediction and market direction
Simulates trades based on predictions
Uploads processed data and predictions to Google Sheets
Ensures that the most recent data is always displayed at the top of the Google Sheet
Requirements
ccxt
pandas
gspread
oauth2client
numpy
schedule
pytz
scikit-learn
Flask
requests
Setup
Clone the repository:

bash
Copy code
git clone https://github.com/your-username/crypto-market-analysis.git
cd crypto-market-analysis
Create a virtual environment (optional but recommended):

bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate
Install the required packages:

bash
Copy code
pip install ccxt pandas gspread oauth2client numpy scikit-learn pytz
Set up Google Sheets API:

Go to the Google Cloud Console: https://console.cloud.google.com/
Create a new project or select an existing one
Enable the Google Sheets API for your project
Create a service account:
Go to "IAM & Admin" > "Service Accounts"
Click "Create Service Account"
Fill in the details and grant it the "Editor" role
Create a key for the service account:
Select the service account you just created
Go to the "Keys" tab
Click "Add Key" > "Create new key"
Choose JSON as the key type and download the key file
Rename the downloaded JSON key file to service_account.json and place it in the project root directory
Create a new Google Sheet:

Go to Google Sheets: https://sheets.google.com
Create a new spreadsheet
Rename it to "CCXT-DATA" (or choose your own name)
Create worksheets named "BTC", "SOL", "ATOM", and "Random Forest"
Share the spreadsheet with the email address of the service account (found in the service_account.json file)
Update the configuration:

Open crypto_analysis.py and rf.py
Update the SPREADSHEET_NAME variable with your Google Sheet name if you chose a different name
Set up a VPN (if necessary):

If you're unable to complete KYC with the exchanges, you may need to use a VPN to access their APIs. Set up a VPN service of your choice and ensure it's running before executing the script.
Usage
Ensure your virtual environment is activated (if you created one).

Run the main script:

bash
Copy code
python crypto_analysis.py
This script will:

Fetch data from the specified exchanges
Process and analyze the data
Update the designated Google Sheet with the collected data
Run the Random Forest analysis (by calling rf.py)
Update the Google Sheet with predictions and simulated trade results
The script is set to run every 6 hours. You can modify the schedule in the run_scheduler() function of crypto_analysis.py if needed.

To run the Random Forest analysis separately:

bash
Copy code
python rf.py
This will update the "Random Forest" worksheet with the latest predictions and simulated trade results.

Understanding the Output
The "BTC", "SOL", and "ATOM" worksheets contain the raw data collected from the exchanges.
The "Random Forest" worksheet contains:
Date of the prediction
Sheet name (corresponding to the cryptocurrency)
Account balance after the simulated trade
Trade type (Long or Short)
Trade outcome (Profit or Loss)
Profit/Loss amount
Predicted price
Entry price
Stop loss price
Take profit price
Customization
To add or remove exchanges, modify the exchanges dictionary in crypto_analysis.py.
To change the trading pairs, update the TRADING_PAIRS dictionary in crypto_analysis.py and SHEET_NAMES in rf.py.
Adjust the Random Forest parameters in the train_predict() function of rf.py.
Modify the trade execution logic in the execute_trade() function of rf.py.
Changes and Improvements
Data Insertion Order: The Random Forest predictions and simulated trades are now inserted at the top of the "Random Forest" worksheet, ensuring that the most recent data is always visible first.
Trade Outcome Formatting: The script now applies conditional formatting to the "Trade Type" column, coloring "Long" trades in green and "Short" trades in red for better visibility.
Disclaimer
This project is for educational and research purposes only. It is not financial advice, and you should not use it to make real trading decisions without thorough testing and understanding of the risks involved.

Contributing
Contributions to improve the project or add new features are welcome. Please submit a pull request or open an issue to discuss proposed changes.

Feture Updates
Going to add security features 
Going to add an exchange API file for automated BOT trading.
Looking to improve the code overall for trading performance and machine learning performance.

License
MIT License

