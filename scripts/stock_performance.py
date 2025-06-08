
"""
Script to find the top N S&P 500 stocks by percentage gain over the last 30 days.
AHN, Jun 2025
"""

import argparse
import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

from pdb import set_trace as BP


#-----------------------
def main():
    
    # Get number of days and top N from command line arguments
    parser = argparse.ArgumentParser(description='Find top N S&P 500 stocks by gain over a number of days')
    parser.add_argument('--topn', type=int, default=20, help='Number of top stocks to display', required=True)
    parser.add_argument('--days', type=int, default=30, help='Number of days to calculate gains over', required=True)
    args = parser.parse_args()
    N = args.topn
    days = args.days

    tickers = get_sp500_tickers()
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)

    # yfinance handles up to 200 tickers per call, so split
    batches = [tickers[i:i + 100] for i in range(0, len(tickers), 100)]
    price_data = pd.DataFrame()

    for batch in batches:
        df = fetch_prices(batch, start_date, end_date)
        price_data = pd.concat([price_data, df], axis=1)

    valid_data = price_data.dropna(axis=1)
    gains = calculate_gains(price_data)
    best_gains = gains.head(N)
    best_tickers = best_gains.index.tolist()
    best_info = [ { 'industry': yf.Ticker(t).info.get('industry', 'N/A'), 'name': yf.Ticker(t).info.get('longName', 'N/A') } for t in best_tickers ]
    
    for idx, ticker in enumerate(best_tickers):
        industry = best_info[idx]['industry']
        name = best_info[idx]['name']
        gain = best_gains[ticker]
        print(f"{gain:.2f}% {ticker} ({name}) - Industry: {industry}")

#-----------------------
def get_sp500_tickers():
    """ Get SP500 tickers from wikipedia """
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'id': 'constituents'})
    tickers = [row.find_all('td')[0].text.strip() for row in table.find_all('tr')[1:]]
    return tickers

#--------------------------------
def calculate_gains(price_df):
    gains = (price_df.iloc[-1] / price_df.iloc[0] - 1) * 100
    return gains.sort_values(ascending=False)

#----------------------------
def clean_ticker(ticker):
    return ticker.replace('.', '-')

#----------------------------------------
def fetch_prices(tickers, start, end):
    """ Download adjusted stock prices from yfinance """
    tickers = [clean_ticker(t) for t in tickers]
    data = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)['Close']
    if isinstance(data, pd.Series):  # Single ticker case
        print('Warning: Single ticker')
        data = data.to_frame()
    return data.dropna(axis=1, how='all')  # Drop tickers with all-NaN data


# Stuff below still to do. Checks for increased trade volume

import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

# --- Step 1: Get S&P 500 tickers ---
def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    soup = BeautifulSoup(requests.get(url).text, 'html.parser')
    table = soup.find('table', {'id': 'constituents'})
    return [row.find_all('td')[0].text.strip().replace('.', '-') for row in table.find_all('tr')[1:]]

# --- Step 2: Get volume data and compute volume ratio ---
def get_high_volume_tickers(tickers, period='30d', threshold=2.0):
    high_volume = []
    for batch in [tickers[i:i+50] for i in range(0, len(tickers), 50)]:
        df = yf.download(batch, period=period, progress=False, group_by='ticker', threads=True)
        
        if isinstance(df.columns, pd.MultiIndex):
            for ticker in batch:
                try:
                    vol = df[ticker]['Volume']
                    avg_vol = vol[:-1].mean()
                    last_vol = vol[-1]
                    ratio = last_vol / avg_vol if avg_vol > 0 else 0
                    if ratio >= threshold:
                        high_volume.append({
                            'Ticker': ticker,
                            'Last Volume': int(last_vol),
                            'Avg Volume': int(avg_vol),
                            'Volume Ratio': round(ratio, 2)
                        })
                except Exception:
                    continue
    return pd.DataFrame(high_volume).sort_values('Volume Ratio', ascending=False)


# Main execution
if __name__ == '__main__':
    main()
    
    