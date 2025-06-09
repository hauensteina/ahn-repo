
"""
Script to simulate trading every day with different parameters.
Buy or hold the top N_STOCKS. Top N_STOCKS by something like vol_5 / vol_30 * incr_5 
where vol is trade volume and incr is increase in value. The number indicates the number of days to look back.
AHN, Jun 2025
"""

"""
Example initial json file for simulating four investors with different strategies:

{
    "last_trade_date": "2025-06-06",
    "investors": [
        {
            "name": "incr_5_days",
            "parameters": {
                "n_stocks": 4,
                "n_days_spike": 5,
                "criterion": "incr"
            },
            "portfolio_log": [
                {
                    "date": "2025-06-06",
                    "cash": 100000,
                    "stocks": {
                    }
                }
            ]
        },
        {
            "name": "incr_2_days",
            "parameters": {
                "n_stocks": 4,
                "n_days_spike": 2,
                "criterion": "incr"
            },
            "portfolio_log": [
                {
                    "date": "2025-06-06",
                    "cash": 100000,
                    "stocks": {
                    }
                }
            ]
        },
        {
            "name": "vol_x_incr_5_days",
            "parameters": {
                "n_stocks": 4,
                "n_days_spike": 5,
                "criterion": "vol_x_incr"
            },
            "portfolio_log": [
                {
                    "date": "2025-06-06",
                    "cash": 100000,
                    "stocks": {
                    }
                }
            ]
        },
        {
            "name": "vol_x_incr_2_days",
            "parameters": {
                "n_stocks": 4,
                "n_days_spike": 2,
                "criterion": "vol_x_incr"
            },
            "portfolio_log": [
                {
                    "date": "2025-06-06",
                    "cash": 100000,
                    "stocks": {
                    }
                }
            ]
        }
    ]
}

"""

import argparse
import os,json
import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import time
from io import StringIO 
import copy
from zoneinfo import ZoneInfo
import pandas_market_calendars as mcal
from datetime import datetime

from pdb import set_trace as BP

N_DAYS_LONG = 30

#--------------
def usage():
    name = os.path.basename( __file__)
    msg = f'''
    Name:
      {name}: Simulate different trading strategies. 

    Description:
        A set of traders with different strategies, and their portfolio status, is defined in trading_tool_investors.json.  
        The script runs in an endless loop waiting for noon (12:00 PM) New York time.
        If the markets are open, it executes trades based on the defined strategies.
        The current state of the portfolios is saved back to trading_tool_investors.json.
        
        Each investor holds n_stocks different tickers. Each day, we find the best (according to strategy) n_stocks and buy them.
        Stocks not in the top n_stocks are sold. 

    Synopsis:
      {name} --run

    Example:
      python {name} --run

''' 
    msg += '\n '
    return msg 

#-----------------------
def main():
    
    parser = argparse.ArgumentParser( usage=usage())
    parser.add_argument('--run', action='store_true', help='Run the trading simulation', required=True)
    args = parser.parse_args()
    
    # Sleep and check occasionally whether it is 9 am and the stock market is open
    while True:
        while True: # Make a block to break out of
            state  = json.load(open('trading_tool_investors.json', 'r'))
            if state['last_trade_date'] == today():
                print("Already traded today. Doing nothing.")
                break    
            east_coast_time = datetime.now(ZoneInfo("America/New_York"))
            if not (east_coast_time.hour >= 12 and east_coast_time.minute > 3): break
            if not is_market_open():
                print(f"{east_coast_time} Market is closed, waiting for next trading day...")
                break
            print(f"{east_coast_time} It's noon, starting the trading simulation...")
            trade(state)
            break
        time.sleep(30)
        
# sort_stocks() defines the way we pick stocks.  
#------------------------------------------------------------------------------------
def sort_stocks(ticker_data, N_STOCKS, criterion, long_start_date, n_days_spike):
    """
    Sort tickers based on recent changes in price and/or volume.
    """

    avg_volume_by_ticker = ticker_data['Volume'].mean()
    recent_volume_by_ticker = ticker_data['Volume'].iloc[-1]
    volrat_by_ticker = recent_volume_by_ticker / avg_volume_by_ticker # yesterdays volume over avg 30 day volume
    recent_price_increase_by_ticker = ticker_data['Close'].iloc[-1] / ticker_data['Close'].iloc[-n_days_spike] # yesterdays price over price n days ago

    if criterion == 'incr':
        crit_by_ticker = recent_price_increase_by_ticker
    elif criterion == 'volume':
        crit_by_ticker = recent_volume_by_ticker
    elif criterion == 'vol_x_incr':
        crit_by_ticker = volrat_by_ticker  * (recent_price_increase_by_ticker - 1)
    elif criterion == 'volrat':
        crit_by_ticker = volrat_by_ticker
    else:
        raise ValueError(f"Unknown criterion: {criterion}")

    # Sort crit_by_ticker in descending order
    crit_by_ticker = dict(sorted(crit_by_ticker.items(), key=lambda x: x[1], reverse=True))
    res = crit_by_ticker
    return res

#--------------------------------------------------------
def is_market_open():
    """ Check if the stock market is open (9:30 AM to 4:00 PM ET) """
    
    # Get NYSE calendar
    nyse = mcal.get_calendar('NYSE')
    # Current time in Eastern Time
    east_coast_time = datetime.now(ZoneInfo("America/New_York"))

    # Check if now is within a market open window
    schedule = nyse.schedule(start_date=east_coast_time.date(), end_date=east_coast_time.date())
    
    # Check if today is a trading day and market is open now
    if schedule.empty: return False
    is_open = schedule.iloc[0]['market_open'] <= east_coast_time <= schedule.iloc[0]['market_close']

    return is_open

#--------------------------------------------------------
def trade(state):
    investors = state['investors']

    END_DATE = datetime.now(ZoneInfo("America/New_York")) 
    LONG_START_DATE = END_DATE - timedelta(days=N_DAYS_LONG)

    tickers = get_sp500_tickers()
    ticker_data = get_data(tickers, LONG_START_DATE, END_DATE)
    metadata = get_sp500_metadata()

    for investor in investors:
        trade_portfolio(investor, ticker_data, metadata, LONG_START_DATE)

    print_portfolio_values(investors, ticker_data)
    print()

    # Save the updated investors data back to the file
    state['last_trade_date'] = today()
    with open('trading_tool_investors.json', 'w') as f:
        json.dump(state, f, indent=4)

#-----------------------------------------------------------------------
def today():
    """ Returns current New York date in 'YYYY-MM-DD' format """
    return datetime.now(ZoneInfo("America/New_York")).strftime('%Y-%m-%d')

#-----------------------------------------------------------------------
def trade_portfolio(investor, ticker_data, metadata, long_start_date):
    print( '\nTrading for Investor: ', investor['name'])
    criterion = investor['parameters']['criterion']
    n_stocks = investor['parameters']['n_stocks']
    n_days_spike = investor['parameters']['n_days_spike']
    top_stocks = dict(list(sort_stocks(ticker_data, n_stocks, criterion, long_start_date, n_days_spike).items())[:n_stocks])
    # Sell any stocks that are not in the top N
    my_stocks = investor['portfolio_log'][-1]['stocks']
    # Get a copy of the last log entry
    last_log_entry = investor['portfolio_log'][-1]
    new_log_entry = copy.deepcopy(last_log_entry)
    new_log_entry['date'] = today()

    # Sell stocks that are not in the top N
    for ticker in my_stocks:
        if ticker not in top_stocks:
            print(f"Selling {ticker} from {investor['name']}'s portfolio")
            sell_ticker(investor, ticker, ticker_data, new_log_entry, last_log_entry)

    # Buy equal amounts of the top N stocks we don't already have
    stocks_to_buy = [ticker for ticker in top_stocks if ticker not in my_stocks]
    print(f"Buying {len(stocks_to_buy)} stocks for {investor['name']}: {', '.join(stocks_to_buy)}")
    cash = new_log_entry['cash']
    # Calculate equal investment amount for each stock
    investment_per_stock = cash / len(stocks_to_buy)
    for ticker in stocks_to_buy:
        buy_ticker(investor, ticker, investment_per_stock, ticker_data, new_log_entry)
        
    # Update the portfolio log with the new log entry
    investor['portfolio_log'].append(new_log_entry)
    
#-----------------------------------------------------------------------------------
def buy_ticker(investor, ticker, investment_amount, ticker_data, new_log_entry):
    """
    Buy a ticker for the investor's portfolio.
    This function should update the portfolio_log with the purchase details.
    """
    # Get the last price of the ticker
    last_price = ticker_data['Close'].iloc[-1][ticker]
    # Calculate number of shares to buy
    shares = int(investment_amount / last_price)

    if shares <= 0:
        print(f"Not enough cash to buy {ticker}. Skipping.")
        return

    # Update portfolio log
    new_log_entry['stocks'][ticker] = {
        'shares': shares,
        'price_bought': last_price,
        'price_sold': None
    }

    # Update cash and log
    new_log_entry['cash'] -= shares * last_price
    print(f"Bought {shares} shares of {ticker} at {last_price:.2f}, remaining cash: {new_log_entry['cash']:.2f}")

#--------------------------------------------------------------------------------
def sell_ticker(investor, ticker, ticker_data, new_log_entry, old_log_entry):
    """
    Sell a ticker from the investor's portfolio.
    This function should update the portfolio_log with the sale details.
    """
    # Get the last price of the ticker
    last_price = ticker_data['Close'].iloc[-1][ticker]
    # Update portfolio log
    shares = new_log_entry['stocks'][ticker]['shares']
    cash = new_log_entry['cash'] + (shares * last_price)

    # Remove the ticker from stocks
    del new_log_entry['stocks'][ticker]
    # Update the last price sold
    old_log_entry['stocks'][ticker]['price_sold'] = last_price
    # Update cash and log
    new_log_entry['cash'] = cash
    print(f"Sold {shares} shares of {ticker} at {last_price:.2f}, new cash balance: {cash:.2f}")
    
#-------------------------------------------------------
def print_portfolio_values(investors, ticker_data):
    """
    Print the current value of each investor's portfolio.
    """
    for investor in investors:
        last_log_entry = investor['portfolio_log'][-1]
        total_value = last_log_entry['cash']
        print(f"\nPortfolio value for {investor['name']}:")
        print(f"Cash: {last_log_entry['cash']:.2f}")
        for ticker, stock_info in last_log_entry['stocks'].items():
            shares = stock_info['shares']
            price_bought = stock_info['price_bought']
            price_sold = stock_info.get('price_sold', None)
            current_price = ticker_data['Close'].iloc[-1][ticker]
            value = shares * current_price
            total_value += value
            print(f"{ticker}: {shares} shares bought at {price_bought:.2f}, current price: {current_price:.2f}, value: {value:.2f}")
            if price_sold:
                print(f"  Sold at: {price_sold:.2f}")
        print(f"Total portfolio value: {total_value:.2f}")

#-------------------------
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



#---------------------------------------------------------------------------------
def get_data(tickers, start_date, end_date):
    """
    Fetch historical stock data for the given tickers and date range.
    """
    # yfinance handles up to 200 tickers per call, so split
    batches = [tickers[i:i + 100] for i in range(0, len(tickers), 100)]
    price_volume_data = pd.DataFrame()
    for batch in batches:
        df = fetch_ticker_data(batch, start_date, end_date)
        price_volume_data = pd.concat([price_volume_data, df], axis=1)

    res = price_volume_data.dropna(axis=1) # Drop tickers with all-NaN data
    return res

#------------------------------------------------------
def fetch_ticker_data(tickers, start_date, end_date):
    """ Download adjusted stock prices from yfinance """
    tickers = [clean_ticker(t) for t in tickers]
    # end_date is exclusive. Dates when market closed are not included.
    data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True) 
    if isinstance(data, pd.Series):  # Single ticker case
        print('Warning: Single ticker')
        data = data.to_frame()
    return data.dropna(axis=1, how='all')  # Drop tickers with all-NaN data

#----------------------------
def clean_ticker(ticker):
    return ticker.replace('.', '-')


if __name__ == '__main__':
    #trade() # test run
    main()
    
    