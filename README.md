# Crypto Market Analysis Project

## Overview

This project fetches and analyzes cryptocurrency market data from top exchanges using the CCXT library. It collects information on trading pairs, calculates liquidity metrics, and determines long/short ratios. The data is then processed and uploaded to a Google Sheet for easy visualization and further analysis.

## **Have to run a VPN if you can't KYC with the exchanges!**

## Features

- Fetches market data from multiple exchanges (Binance, Bybit, HitBTC, Coinbase Advanced)
- Analyzes trading pairs (BTC/USDT, SOL/USDT, ATOM/USDT)
- Calculates liquidity metrics (bid/ask liquidity, bid/ask ratio)
- Determines long/short ratios based on recent trades
- Uploads processed data to Google Sheets

## Future Goals

The project aims to incorporate machine learning techniques to predict market direction based on:
- Total trading volume
- Short and long ratios
- Data from the top 4 crypto exchanges

This addition will help in identifying potential market trends and making more informed trading decisions.

## Requirements

- Python 3.7+
- CCXT library
- pandas
- gspread
- oauth2client
- numpy

## Setup

1. Clone the repository: 
      git clone https://github.com/jeffAquinn/Crypto-Analysis-with-Machine-Learning
   
3. Install the required packages:
   ```
   pip install ccxt pandas gspread oauth2client numpy
   ```
4. Set up Google Sheets API:
   - Create a Google Cloud project
   - Enable Google Sheets API
   - Create a service account and download the JSON key
   - Rename the key to `service_account.json` and place it in the project root
5. Update the `SPREADSHEET_NAME` variable in the script with your Google Sheet name

## Usage

Run the script:

```
python crypto_analysis.py
```

The script will fetch data from the specified exchanges, process it, and update the designated Google Sheet.

## Contributing

Contributions to improve the project or add new features are welcome. Please submit a pull request or open an issue to discuss proposed changes.

## License

[MIT License](https://opensource.org/licenses/MIT)
