# Crypto Market Analysis Project

## Overview

This project fetches and analyzes cryptocurrency market data from top exchanges using the CCXT library. It collects information on trading pairs, calculates liquidity metrics, and determines long/short ratios. The data is then processed, analyzed using machine learning techniques, and uploaded to a Google Sheet for easy visualization and further analysis.

## Support

If you find this code helpful and want to give back, please send crypto to the following addresses:

- BTC: 3LvkjKVkSfLCKfxWPdsBokXnmsBhRrtmFk
- XRP: rUwrkngQTjKhrBf9FdC1wGTBsbgC8yhweK (No Memo)
- ETH: 0x4f18f6ead4d8c22a2e757257ad647b5d2d110cd4

**Note:** You may need to run a VPN if you can't complete KYC with the exchanges.

## Features

- Fetches market data from multiple exchanges (Binance, Bybit, HitBTC)
- Analyzes trading pairs (BTC/USDT, SOL/USDT, ATOM/USDT)
- Calculates liquidity metrics (bid/ask liquidity, bid/ask ratio)
- Determines long/short ratios based on recent trades
- Uses Random Forest algorithm for price prediction and market direction
- Simulates trades based on predictions
- Uploads processed data and predictions to Google Sheets
- Ensures that the most recent data is always displayed at the top of the Google Sheet

## Requirements

- ccxt
- pandas
- gspread
- oauth2client
- numpy
- schedule
- pytz
- scikit-learn
- Flask
- requests

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/crypto-market-analysis.git
   cd crypto-market-analysis
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install ccxt pandas gspread oauth2client numpy scikit-learn pytz
   ```

4. Set up Google Sheets API:
   - Go to the [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select an existing one
   - Enable the Google Sheets API for your project
   - Create a service account and key (download as JSON)
   - Rename the key file to `service_account.json` and place it in the project root directory

5. Create a new Google Sheet:
   - Create a new spreadsheet and name it "CCXT-DATA" (or choose your own name)
   - Create worksheets named "BTC", "SOL", "ATOM", and "Random Forest"
   - Share the spreadsheet with the email address of the service account

6. Update the configuration:
   - Open `crypto_analysis.py` and `rf.py`
   - Update the `SPREADSHEET_NAME` variable with your Google Sheet name if different

7. Set up a VPN (if necessary):
   - If you're unable to complete KYC with the exchanges, set up a VPN service

## Usage

1. Ensure your virtual environment is activated (if created).

2. Run the main script:
   ```bash
   python crypto_analysis.py
   ```

3. To run the Random Forest analysis separately:
   ```bash
   python rf.py
   ```

## Understanding the Output

- The "BTC", "SOL", and "ATOM" worksheets contain raw data from the exchanges.
- The "Random Forest" worksheet contains predictions and simulated trade results.

## Customization

- Modify the `exchanges` dictionary in `crypto_analysis.py` to add or remove exchanges.
- Update the `TRADING_PAIRS` dictionary in `crypto_analysis.py` and `SHEET_NAMES` in `rf.py` to change trading pairs.
- Adjust Random Forest parameters in the `train_predict()` function of `rf.py`.
- Modify trade execution logic in the `execute_trade()` function of `rf.py`.

## Recent Improvements

- Data insertion order: Most recent data appears at the top of the "Random Forest" worksheet.
- Trade outcome formatting: "Long" trades are colored green, "Short" trades are red.

## Disclaimer

This project is for educational and research purposes only. It is not financial advice.

## Contributing

Contributions are welcome. Please submit a pull request or open an issue to discuss proposed changes.

## Future Updates

- Add security features
- Add an exchange API file for automated BOT trading
- Improve overall code for trading and machine learning performance

## License

MIT License