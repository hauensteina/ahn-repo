
'''
Demonstrate IRR computation for stock portfolio.
AHN, Sep 2024
'''

import copy
from pdb import set_trace as BP
from datetime import datetime
from datetime import date
import os
import argparse
import pprint

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
      Use both a numeric solution for IRR and the Dietz method.
      Compute dollars earned in the period.

    Example:
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

    if args.account:
        mindate = min([ t['date'] for t in TRANSACTIONS if t['account'] == args.account])
        maxdate = max([ t['date'] for t in TRANSACTIONS if t['account'] == args.account])
        
    args.from_date = max( args.from_date, mindate)
    args.to_date = min( args.to_date, maxdate)
        
    if args.from_date < mindate:
        args.from_date = mindate

    add_total_transactions()
    
    transactions = [ t for t in TRANSACTIONS if args.from_date <= t['date'] <= args.to_date]
    if args.account:
        transactions = [ t for t in transactions if t['account'] == args.account]

    if len( [t for t in transactions if t['account'] == '#TOTAL']) < 2:
        print( 'Not enough transactions for IRR computation. Need at least 2')
        return
    
    print()
    print("Period from", args.from_date, "to", args.to_date)
    print()
    dietz_irr = compute_dietz_irr( transactions, args.from_date, args.to_date)
    for a in dietz_irr:
        if '_annualized' in a: continue
        print( f'Dietz IRR for account {a}: {dietz_irr[a]:.2f} annualized: {dietz_irr[a + "_annualized"]:.2f}')
    print( f'Dietz IRR overall: {dietz_irr['#TOTAL']:.2f} annualized: {dietz_irr['#TOTAL_annualized']:.2f}')
    print()

    # irr = compute_irr( transactions)
    # print( f'IRR: {irr:.2f}%')
    
    dollars = compute_revenue_dollars( transactions, args.from_date, args.to_date) # Revenue by account 
    for a in dollars:
        if '_annualized' in a: continue
        print( f'Dollars earned for account {a}: {dollars[a]:.2f} annualized: {dollars[a + "_annualized"]:.2f}')
    print( f'Dollars earned total: {dollars["#TOTAL"]:.2f} annualized: {dollars["#TOTAL_annualized"]:.2f}')
    print()

#--------------------------------
def add_total_transactions():
    """ Add transactions for total portfolio """
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

#----------------------------------------------
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
    return dollars

def period_len( d1, d2):
    """ Return number of days between two date strings in iso format, plus 1. d2 is larger than d1 """
    res = (datetime.strptime( d2, '%Y-%m-%d') - datetime.strptime( d1, '%Y-%m-%d')).days + 1
    return res
    
    
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
        account_transactions = [ t for t in transactions if t['account'] == a ]
        if not account_transactions: continue
        for idx,t in enumerate(account_transactions):
            transaction_weight = ( days_in_period - period_len( from_date, account_transactions[idx]['date']) ) / days_in_period 
            denominator += transaction_weight * account_transactions[idx]['dollars']
            external_sum += account_transactions[idx]['dollars']
            
        dietz[a] = 1 + (dollars_end - dollars_start[a] - external_sum) / denominator   
    
    # Compute annualized returns
    for a in ACCOUNTS + ['#TOTAL']:
        if not a in dietz: continue
        dietz[a + '_annualized'] = dietz[a] ** ( 365 / days_in_period )  
    
    return dietz
    
main()
