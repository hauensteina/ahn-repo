
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
from datetime import datetime, timedelta, date
import time
from io import StringIO 
import copy
from zoneinfo import ZoneInfo
import pandas_market_calendars as mcal

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
        
    Options:
        --run: Run the trading simulation every day at noon New York time, starting today.
        --simulate: Use data from the past year to simulate a year of trading

    Synopsis:
      {name} --mode [ run | simulate ]

    Examples:
        python {name} --mode run
        python {name} --mode simulate
    
''' 
    msg += '\n '
    return msg 

#-----------------------
def main():
    parser = argparse.ArgumentParser( usage=usage())
    parser.add_argument('--mode', choices=['run', 'simulate'], default='simulate', help='Mode of operation')
    args = parser.parse_args()

    if args.mode == 'run':
        print("Running the trading simulation...")
        run()
    else: 
        print("Trading with last year's data...")
        simulate()
        
#--------------------------------------------------------        
def simulate():
    """
    Simulate trading with historical data for the past year.
    This function will read the initial state from trading_tool_investors.json,
    fetch historical data for the past year, and simulate trades based on the defined strategies.
    """
    
    # Load initial state
    state = json.load(open('trading_tool_investors.json', 'r'))
    
    # Set start and end dates for simulation
    end_date = datetime.now(ZoneInfo("America/New_York"))
    start_date = end_date - timedelta(days=365)  # One year of data
    
    tickers = get_sp500_tickers()
    #metadata = get_sp500_metadata()
    # See if we have a pickle file with the data
    #close_price_data, volume_data = get_data(tickers, start_date, end_date)
    try:
        pickle = "sp500_year_ending_mon_2025-06-09.pkl"
        with open(pickle, 'rb') as f: data = pd.read_pickle(f)
        close_price_data = data['Close']
        volume_data = data['Volume']
        print(f"Loaded ticker data from file: {pickle}")
        end_date = datetime.fromisoformat('2025-06-09')
        start_date = end_date - timedelta(days=365)
    except FileNotFoundError:
        print(f"Getting ticker data from yfinance")
        close_price_data, volume_data = get_data(tickers, start_date, end_date)

    days = close_price_data.index[30:]
    for idx,day in enumerate(days):
        print(f"\nDay {idx+1}: {day.strftime('%Y-%m-%d')}")
        print('===============================================================')
        start_date = day - timedelta(days=30)  # one month of data
        close_30_days = close_price_data.loc[start_date:day]
        volume_30_days = volume_data.loc[start_date:day]
        for investor in state['investors']:
            # Simulate trading for each day
            trade_portfolio(investor, close_30_days, volume_30_days)
        print_portfolio_values(state['investors'], close_30_days)

#-----------------------------------------------------------------------
def run():
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
# close_price_data and volume_data cover the previous 30 days.
#-----------------------------------------------------------------------------------------
def sort_stocks(close_price_data, volume_data, criterion, n_days_spike):
    """
    Sort tickers based on recent changes in price and/or volume.
    """

    avg_volume_by_ticker = volume_data.mean()
    recent_volume_by_ticker = volume_data.iloc[-1]
    volrat_by_ticker = recent_volume_by_ticker / avg_volume_by_ticker # yesterdays volume over avg 30 day volume
    recent_price_increase_by_ticker = close_price_data.iloc[-1] / close_price_data.iloc[-n_days_spike] # yesterdays price over price n days ago

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
    close_price_data, volume_data = get_data(tickers, LONG_START_DATE, END_DATE)
    metadata = get_sp500_metadata()

    for investor in investors:
        trade_portfolio(investor, close_price_data, volume_data)

    print_portfolio_values(investors, close_price_data)
    print()

    # Save the updated investors data back to the file
    state['last_trade_date'] = today()
    with open('trading_tool_investors.json', 'w') as f:
        json.dump(state, f, indent=4)

#-----------------------------------------------------------------------
def today():
    """ Returns current New York date in 'YYYY-MM-DD' format """
    return datetime.now(ZoneInfo("America/New_York")).strftime('%Y-%m-%d')

#------------------------------------------------------------------------------------
def trade_portfolio(investor, close_price_data, volume_data):
    print( '\nTrading for Investor: ', investor['name'])
    print('--------------------------------------------------')
    criterion = investor['parameters']['criterion']
    n_stocks = investor['parameters']['n_stocks']
    n_days_spike = investor['parameters']['n_days_spike']
    top_stocks = dict(list(sort_stocks(close_price_data, volume_data, criterion, n_days_spike).items())[:n_stocks])
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
            sell_ticker(investor, ticker, close_price_data, new_log_entry, last_log_entry)

    # Buy equal amounts of the top N stocks we don't already have
    stocks_to_buy = [ticker for ticker in top_stocks if ticker not in my_stocks]
    print(f"\nBuying {len(stocks_to_buy)} stocks for {investor['name']}: {', '.join(stocks_to_buy)}")
    cash = new_log_entry['cash']
    # Calculate equal investment amount for each stock
    investment_per_stock = cash / len(stocks_to_buy)
    for ticker in stocks_to_buy:
        buy_ticker(investor, ticker, investment_per_stock, close_price_data, new_log_entry)

    print()
    # Update the portfolio log with the new log entry
    investor['portfolio_log'].append(new_log_entry)
    
#-----------------------------------------------------------------------------------
def buy_ticker(investor, ticker, investment_amount, close_price_data, new_log_entry):
    """
    Buy a ticker for the investor's portfolio.
    This function should update the portfolio_log with the purchase details.
    """
    # Get the last price of the ticker
    last_price = close_price_data.iloc[-1][ticker]
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
def sell_ticker(investor, ticker, close_price_data, new_log_entry, old_log_entry):
    """
    Sell a ticker from the investor's portfolio.
    This function should update the portfolio_log with the sale details.
    """
    # Get the last price of the ticker
    last_price = close_price_data.iloc[-1][ticker]
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
def print_portfolio_values(investors, close_price_data):
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
            current_price = close_price_data.iloc[-1][ticker]
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
    close_price_data = pd.DataFrame()
    volume_data = pd.DataFrame()
    for batch in batches:
        df = fetch_ticker_data(batch, start_date, end_date)
        close_price = df['Close']
        volume = df['Volume']
        close_price_data = pd.concat([close_price_data, close_price], axis=1)
        volume_data = pd.concat([volume_data, volume], axis=1)

    return close_price_data, volume_data

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
    
    