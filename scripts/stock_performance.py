
"""
Script to find the top N S&P 500 stocks by percentage gain over the last 30 days.
AHN, Jun 2025
"""

import argparse
import os
import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from io import StringIO 

from pdb import set_trace as BP


#--------------
def usage():
    name = os.path.basename( __file__)
    msg = f'''
    Name:
      {name}: Find the best N S&P 500 stocks by gains over a given period.

    Synopsis:
      {name} --days <days> --topn <topn> 

    Example:
      python {name} --days 30 --topn 10

''' 
    msg += '\n '
    return msg 

#-----------------------
def main():
    
    # Get number of days and top N from command line arguments
    parser = argparse.ArgumentParser( usage=usage())
    parser.add_argument('--topn', type=int, default=20, help='Number of top stocks to display', required=True)
    parser.add_argument('--days', type=int, default=30, help='Number of days to calculate gains over', required=True)
    args = parser.parse_args()
    N = args.topn
    days = args.days

    tickers = get_sp500_tickers()
    tickers = tickers[:2] # For testing, limit to 2 tickers
    metadata = get_sp500_metadata()
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)

    # yfinance handles up to 200 tickers per call, so split
    batches = [tickers[i:i + 100] for i in range(0, len(tickers), 100)]
    price_data = pd.DataFrame()

    for batch in batches:
        df = fetch_prices(batch, start_date, end_date) # end_date is exclusive. Dates when market closed are not included.
        price_data = pd.concat([price_data, df], axis=1)

    valid_data = price_data.dropna(axis=1)
    BP()
    gains = calculate_gains(valid_data)
    best_gains = gains.head(N)
    best_tickers = best_gains.index.tolist()
    best_info = [ { 'industry': metadata[t]['Industry'], 'name': metadata[t]['Company'] } for t in best_tickers ]

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

#------------------------------------------------------------------------------------
def get_sp500_metadata():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    table_html = soup.find('table', {'id': 'constituents'})
    df = pd.read_html(StringIO(str(table_html)))[0]

    # Clean ticker symbols (for yfinance compatibility: '.' → '-')
    df['Symbol'] = df['Symbol'].str.replace('.', '-', regex=False)
    df = df.rename(columns={'Symbol': 'Ticker', 'Security': 'Company', 'GICS Sector': 'Sector', 'GICS Sub-Industry': 'Industry'})

    metadata_dict = {
        row['Ticker']: {
            'Company': row['Company'],
            'Sector': row['Sector'],
            'Industry': row['Industry']
        }
        for _, row in df.iterrows()
    }
    return metadata_dict

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


# Main execution
if __name__ == '__main__':
    main()
    
    