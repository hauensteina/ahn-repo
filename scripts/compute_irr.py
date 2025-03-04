
'''
Demonstrate IRR computation for stock portfolio.
AHN, Sep 2024
'''

"""
Suppose I start with 100 dollars at the beginning of year 1.
Then I invest another 100 at the beginning of year two.
Year 1 had growth 1.1, year 2 had growth 1.2 .

After year 1, I have 100 * 1.1 = 110 dollars.
After year 2, I have (100 + 110) * 1.2 = 210 + 42 = 252 dollars.

As a TRANSACTIONS list:

TRANSACTIONS = [
    { 'account':'CMA', 'dollars': 100, 'date': '2022', balance_after: 100 },
    { 'account':'CMA', 'dollars': 100, 'date': '2023', balance_after: 210 },
    { 'account':'CMA', 'dollars': 0, 'date': '2024', balance_after: 252 },
]

N = 2
n in {0,1,2}

IRR:

(c_0 * g + c_1) * g  = NPV 
c_0 * g^2 + c_1 * g = NPV
c_0 * g^2 + c_1 * g - NPV = 0

c_0 happens at the beginning of year 0.
c_1 happens at the beginning of year 1.
NPV is the value at the beginning of year 2 and we could say c_2 = -NPV for mathematical elegance.

In general:

sum_{n=0 to N} { c_n * g^(N-n) }  = 0

Solve for g.

c_0 = 100
c_1 = 100
NPV = 252
c_2 = -252

c_0  * g^2 + c_1 * g + c_2 = 0
100 * g^2 + 100 * g - 252 = 0

>>> import numpy as np
>>> coeff = [ 100, 100, -252 ]
>>> np.roots(coeff)
array([-2.1643317,  1.1643317])

So IRR is 1.1643317 - 1 = 16%

If g is the annual growth factor, but cash flows can happen daily (i.e. N and n are days), replace 
g^(N-n) 
with
( g^{ 1 / 365 } ) ^ { N - n } =  g^{ (N-n) / 365 } 

"""

from collections import defaultdict
import copy
from pdb import set_trace as BP
from datetime import datetime, timedelta
from datetime import date
import os
import argparse
import numpy as np
from pprint import pprint

# Negative prices means sold, positive means bought.
# Account is SEP or CMA
# Balance is *after* the transaction
TRANSACTIONS =  [
    { 'ticker': 'VOO', 'date': '2022-12-09', 'dollars': 95678.0, 'account':'CMA', 'balance_after':95678.0 },
    { 'ticker': 'VOO', 'date': '2022-12-21', 'dollars': 99808.0, 'account':'CMA' , 'balance_after':198320.0 },

    { 'ticker': 'VOO', 'date': '2023-01-18', 'dollars': -100436.0, 'account':'CMA', 'balance_after':109320.0 },
    { 'ticker': 'VOO', 'date': '2024-02-06', 'dollars': 49507.0, 'account':'CMA', 'balance_after':182501.0 },

    { 'ticker': 'NVDA', 'date': '2024-02-27', 'dollars': 49479.0, 'account':'CMA', 'balance_after':231980.0 },
    { 'ticker': 'NVDA', 'date': '2024-03-26', 'dollars': 9403.0, 'account':'CMA', 'balance_after':265450.0 },

    { 'ticker': 'VOO', 'date': '2024-04-02', 'dollars': 14912.0, 'account':'CMA', 'balance_after':265716.0 },

    { 'ticker': 'TSLA', 'date': '2024-05-29', 'dollars': 4946.0, 'account':'CMA', 'balance_after':289296.0 },

    { 'ticker': 'TSLA', 'date': '2024-06-14', 'dollars': 10178.0, 'account':'CMA', 'balance_after':313411.0 },

    { 'ticker': 'TSLA', 'date': '2024-07-03', 'dollars': 5032.0, 'account':'CMA', 'balance_after':318443.0 },

    { 'ticker': 'COST', 'date': '2024-09-12', 'dollars': 24165.0, 'account':'CMA', 'balance_after':352221.0 },
    { 'ticker': 'VOO', 'date': '2024-09-12', 'dollars': 24709.0, 'account':'CMA', 'balance_after':376930.0 },
    
    { 'ticker': 'VOO', 'date': '2023-09-13', 'dollars': 46542.0, 'account':'SEP', 'balance_after':46542.0 },
    { 'ticker': 'COST', 'date': '2024-09-13', 'dollars': 7203.0, 'account':'SEP', 'balance_after':67390.0 },

    { 'ticker': 'VOO', 'date': '2024-09-30', 'dollars': 99674, 'account':'CMA', 'balance_after':488492.0 },
    { 'ticker': 'dummy', 'date': '2024-09-30', 'dollars': 0.0, 'account':'SEP', 'balance_after':68409.0 },
    
    { 'ticker': 'TSLA', 'date': '2024-10-08', 'dollars': -26082.0, 'account':'CMA', 'balance_after':462410.0 },
    { 'ticker': 'VOO', 'date': '2024-10-10', 'dollars': 28603.0, 'account':'CMA', 'balance_after':498319.0 },
    
    { 'ticker': 'dummy', 'date': '2024-10-29', 'dollars': 0.0, 'account':'SEP', 'balance_after':69804.0 },
    { 'ticker': 'dummy', 'date': '2024-10-29', 'dollars': 0.0, 'account':'CMA', 'balance_after':507515.0 },
    
    { 'ticker': 'dummy', 'date': '2024-12-05', 'dollars': 0.0, 'account':'SEP', 'balance_after':73242.0 },
    { 'ticker': 'dummy', 'date': '2024-12-05', 'dollars': 0.0, 'account':'CMA', 'balance_after':529192.0 },

    { 'ticker': 'AAPL', 'date': '2024-12-09', 'dollars': 14752.0, 'account':'CMA', 'balance_after':538681.0 },

    { 'ticker': 'dummy', 'date': '2025-01-06', 'dollars': 0.0, 'account':'SEP', 'balance_after':72021.0 },
    { 'ticker': 'dummy', 'date': '2025-01-06', 'dollars': 0.0, 'account':'CMA', 'balance_after':542763.0 },
    
    { 'ticker': 'dummy', 'date': '2025-01-30', 'dollars': 0.0, 'account':'SEP', 'balance_after':73121.0 },
    { 'ticker': 'dummy', 'date': '2025-01-30', 'dollars': 0.0, 'account':'CMA', 'balance_after':528735.0 },

    { 'ticker': 'NVDA', 'date': '2025-01-31', 'dollars': -86513.0, 'account':'CMA', 'balance_after':434271.0 },
    { 'ticker': 'FXF', 'date': '2025-02-04', 'dollars': 86586.0, 'account':'CMA', 'balance_after':521739.0 },
    
    { 'ticker': 'dummy', 'date': '2025-03-03', 'dollars': 0.0, 'account':'SEP', 'balance_after':71483.0 },
    { 'ticker': 'dummy', 'date': '2025-03-03', 'dollars': 0.0, 'account':'CMA', 'balance_after':515022.0 },
]

# Remember transaction order for sorting
for idx,t in enumerate(TRANSACTIONS):
    # Left pad id with zeros
    t['id'] = f'{idx:04}'
    
ACCOUNTS = ['CMA', 'SEP']

#--------------
def usage():
    name = os.path.basename( __file__)
    msg = f'''
    Name:
      {name}: Compute IRR for our portfolio

    Synopsis:
      {name} --from_date <date> --to_date <date> [--account <account>] 

    Description:
      Compute IRR between the given dates, annualized and absolute. 
      Account is SEP or CMA. If not given, compute for total portfolio.
      Use the Dietz method for IRR computation.
      Compute dollars earned in the period.
      
    Examples:
      python {name} --from_date 2022-12-09 --to_date 2024-09-15 

''' 
    msg += '\n '
    return msg 

#-------------
def main():
    mindate = min([ t['date'] for t in TRANSACTIONS])
    maxdate = max([ t['date'] for t in TRANSACTIONS])
    
    parser = argparse.ArgumentParser( usage=usage())
    parser.add_argument( '--from_date', default=mindate)
    parser.add_argument( '--to_date', default=maxdate)
    parser.add_argument( '--account', default='')
    args = parser.parse_args()
        
    add_total_transactions()
    
    minmax_dates = get_minmax_dates( args.from_date, args.to_date)
    transactions = filter_transactions( minmax_dates)
    
    if args.account:
        transactions = [ t for t in transactions if t['account'] == args.account]

    if len( [t for t in transactions if t['account'] == '#TOTAL']) < 2:
        print( 'Not enough transactions for IRR computation. Need at least 2')
        return
    
    print()
    print("Period from", args.from_date, "to", args.to_date)
    print()
    print("Closest Account Periods:")
    for a in ACCOUNTS + ['#TOTAL']:
        print( f'{a}: {minmax_dates[a][0]} to {minmax_dates[a][1]}')
    print()
    
    dietz_irr = compute_dietz_irr( transactions, args.from_date, args.to_date)
    for a in dietz_irr:
        print( f'Dietz IRR for account {a}: {dietz_irr[a]:.2f}')
    print()

    irr = compute_irr( transactions)
    for a in irr:
        print( f'IRR for account {a}: {irr[a]:.2f}')
    print()
    
    dollars, dollars_end = compute_revenue_dollars( transactions, args.from_date, args.to_date) # Revenue by account 
    for a in dollars:
        if '_annualized' in a: continue
        print( f'Dollars earned for account {a}: {dollars[a]:.2f} annualized: {dollars[a + "_annualized"]:.2f}')
    print()
    for a in dollars_end:
        print( f'End value for account {a}: {dollars_end[a]:.2f}')
    print()

    externals = compute_externals()
    for a in externals:
        print( f'Forever externals for account {a}: {externals[a]:.2f}')
    print()
    final = final_value()
    for a in final:
        print( f'Current value for account {a}: {final[a]:.2f}')
    print()
    
#-------------------------------------------    
def filter_transactions( minmax_dates):
    """ Remove transactions outside the minmax_dates range """
    res = []
    for t in TRANSACTIONS:
        if minmax_dates[t['account']][0] <= t['date'] <= minmax_dates[t['account']][1]:
            res.append( t)
    return res
    
#-------------------------------------------    
def get_minmax_dates(from_date, to_date):
    """ For each account, return earliest and latest transaction date to span the period """
    minmax_dates = {}
    for a in ACCOUNTS + ['#TOTAL']:
        # Find the transaction date closest to from_date and to_date
        account_dates = [ t['date'] for t in TRANSACTIONS if t['account'] == a]
        closest_from_date = min( account_dates, key=lambda x: abs(datetime.strptime(x, '%Y-%m-%d') - datetime.strptime(from_date, '%Y-%m-%d')))
        closest_to_date = min( account_dates, key=lambda x: abs(datetime.strptime(x, '%Y-%m-%d') - datetime.strptime(to_date, '%Y-%m-%d')))
        minmax_dates[a] = (closest_from_date, closest_to_date)

    return minmax_dates
    
#--------------------------------
def add_total_transactions():
    """ Add transactions for #TOTAL portfolio """
    global TRANSACTIONS
    # Compute balance_after across all accounts for each transaction
    for idx,t in enumerate(TRANSACTIONS):
        t['total_balance_after'] = total_other_balances_for_transaction( t) + t['balance_after']
        
    total_transactions = sorted(copy.deepcopy(TRANSACTIONS), key=lambda x: x['date'] + x['id'])
    for idx,t in enumerate(total_transactions):
        t['balance_after'] = t['total_balance_after'] 
        t['account'] = '#TOTAL'
        
    # for t in total_transactions:
    #     pprint.pprint(t)

    TRANSACTIONS.extend( total_transactions)

#--------------------------------------------------------
def total_other_balances_for_transaction( transaction):
    """ Return total balance of all accounts except account at date datestr """
    total = 0.0
    for a in ACCOUNTS:
        if a == transaction['account']: continue
        transactions = [ t for t in TRANSACTIONS 
                        if t['account'] == a 
                        and t['date'] + t['id'] < transaction['date'] + transaction['id']]
        if not transactions: continue
        total += transactions[-1]['balance_after']
    return total

#----------------------------------------------------------------
def compute_revenue_dollars( transactions, from_date, to_date):
    """ Compute revenue by account """
    days_in_period = period_len( from_date, to_date)
    dollars_end = {}
    dollars_start = {}
    externals = {}
    dollars = {}
    for a in ACCOUNTS + ['#TOTAL']:
        account_transactions = [ t for t in transactions if t['account'] == a]
        if not account_transactions: continue
        externals[a] = sum([ t['dollars'] for t in account_transactions])
        dollars_end[a] = account_transactions[-1]['balance_after']
        dollars_start[a] = account_transactions[0]['balance_after'] - account_transactions[0]['dollars']
        dollars[a] = dollars_end[a] - dollars_start[a] - externals[a]
        
    # Compute annualized revenue
    for a in ACCOUNTS + ['#TOTAL']:
        if not a in dollars: continue
        dollars[a + '_annualized'] = dollars[a] / days_in_period * 365    
    return dollars, dollars_end

#----------------------------
def compute_externals():
    """ Sum externals over all time """
    externals = {}
    for a in ACCOUNTS + ['#TOTAL']:
        account_transactions = [ t for t in TRANSACTIONS if t['account'] == a]
        if not account_transactions: continue
        externals[a] = sum([ t['dollars'] for t in account_transactions])
    return externals

#--------------------
def final_value():
    """ Show latest value of portfolio """
    dollars = {}
    for a in ACCOUNTS + ['#TOTAL']:
        account_transactions = [ t for t in TRANSACTIONS if t['account'] == a]
        if not account_transactions: continue
        dollars[a] = account_transactions[-1]['balance_after']
    return dollars

#-------------------------
def period_len( d1, d2):
    """ Return number of days between two date strings in iso format. d2 is larger than d1 """
    res = (datetime.strptime( d2, '%Y-%m-%d') - datetime.strptime( d1, '%Y-%m-%d')).days 
    return res

#-----------------------------------
def compute_irr(transactions):
    # For each account, for each day, compute balance before and after the transaction.
    # Transactions happen at the beginning of the day.
    irr = {}
    for a in ACCOUNTS + ['#TOTAL']:
        account_transactions = [ t for t in transactions if t['account'] == a]
        if len(account_transactions) < 2: continue
        # Days with transactions, as python dates
        days_with_transactions = [ datetime.strptime(t['date'], '%Y-%m-%d') for t in account_transactions]
        # All days in period, whether they had a transaction or not
        start_date = days_with_transactions[0]
        end_date = days_with_transactions[-1]
        all_days = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]
        
        by_day = { key: defaultdict( float) for key in all_days}
        for t in account_transactions:
            dt = datetime.strptime( t['date'], '%Y-%m-%d')
            by_day[dt]['dollars'] += t['dollars']     
            by_day[dt]['balance_after'] = t['balance_after']
            by_day[dt]['balance_before'] = by_day[dt]['balance_after'] - t['dollars']
        
        # Coefficient for each day in the period. Most will be zero.
        by_day_list = list(by_day.items())
        coefficients = []
        coefficients.append( by_day_list[0][1]['balance_before'])
        for dt, vals in by_day_list[:-1]:
            coefficients.append( vals['dollars'])
        coefficients.append( -by_day_list[-1][1]['balance_before'])
            
        roots = np.roots(coefficients)
        real_roots = roots[np.abs(roots.imag) < 1e-10].real
        solutions = [ s for s in real_roots if s > 0]
        # Now we have IRR per day. Annualize it.
        annualized = [ s ** 365 for s in solutions ]
        try:
            irr[a] = annualized[0]
        except Exception as e:
            BP()
            tt = 42
    return irr          

#-----------------------------------
def test_irr():
    def eq(a,b):
        if abs(a-b) < 0.0001:
            return True
        return False
        
    print('Testing IRR computation')    
        
    # One percent per 1 day 
    transactions = [
        { 'date': '1900-01-01', 'account':'CMA', 'dollars': 100.0, 'balance_after': 100.0 },
        { 'date': '1900-01-02', 'account':'CMA', 'dollars': 0.0, 'balance_after': 101.0 },
    ]
    res = compute_irr( transactions)
    annual_growth = res['CMA']
    computed_final_value = 100.0 * annual_growth ** (1 / 365)
    recorded_final_value = transactions[-1]['balance_after'] - transactions[-1]['dollars'] 

    if eq( computed_final_value, recorded_final_value):
        print( '  Test1 passed')
    else:
        print( f'  ERROR: Test1 failed. Computed final value: {computed_final_value} Recorded final value: {recorded_final_value}')

    # One percent per 10 days
    transactions = [
        { 'date': '1900-01-01', 'account':'CMA', 'dollars': 100.0, 'balance_after': 100.0 },
        { 'date': '1900-01-11', 'account':'CMA', 'dollars': 0.0, 'balance_after': 101.0 },
    ]
    res = compute_irr( transactions)
    annual_growth = res['CMA']
    computed_final_value = 100.0 * annual_growth ** (10 / 365)
    recorded_final_value = transactions[-1]['balance_after'] - transactions[-1]['dollars'] 

    if eq( computed_final_value, recorded_final_value):
        print( '  Test2 passed')
    else:
        print( f'  ERROR: Test2 failed. Computed final value: {computed_final_value} Recorded final value: {recorded_final_value}')
    
    # One percent per 10 days, then about half a percent in the next 10 days
    transactions = [
        { 'date': '1900-01-01', 'account':'CMA', 'dollars': 100.0, 'balance_after': 100.0 },
        { 'date': '1900-01-11', 'account':'CMA', 'dollars': 50.0, 'balance_after': 151.0 },
        { 'date': '1900-01-21', 'account':'CMA', 'dollars': 0.0, 'balance_after': 151.75 },
    ]
    res = compute_irr( transactions)
    annual_growth = res['CMA']
    computed_final_value = 100.0 * annual_growth ** (20 / 365) + 50.0 * annual_growth ** (10 / 365)
    recorded_final_value = transactions[-1]['balance_after'] - transactions[-1]['dollars']
    
    if eq( computed_final_value, recorded_final_value):
        print( '  Test3 passed')
    else:
        print( f'  ERROR: Test3 failed. Computed final value: {computed_final_value} Recorded final value: {recorded_final_value}')
    
#-----------------------------------------------------------    
def compute_dietz_irr( transactions, from_date, to_date):
    """ Compute IRR using the modified Dietz method, by account and total """
    days_in_period = period_len( from_date, to_date)

    dietz = {}
    dollars_start = {}
    for a in ACCOUNTS + ['#TOTAL']:
        account_transactions = [ t for t in transactions if t['account'] == a]
        if not account_transactions: continue
        # Market value at start of period
        dollars_start[a] = account_transactions[0]['balance_after'] - account_transactions[0]['dollars']
        dollars_end = account_transactions[-1]['balance_after']
        denominator = dollars_start[a]
        external_sum = 0.0
        for idx,t in enumerate(account_transactions):
            transaction_weight = ( days_in_period - period_len( from_date, account_transactions[idx]['date']) ) / days_in_period 
            denominator += transaction_weight * account_transactions[idx]['dollars']
            external_sum += account_transactions[idx]['dollars']
            
        dietz[a] = 1 + (dollars_end - dollars_start[a] - external_sum) / denominator   
        dietz[a] ** ( 365 / days_in_period )
            
    return dietz
 
#test_irr()    
main()
