
'''
Number of coin tosses needed on average to get two heads in a row and 
number of tosses needed to get three heads in a row  
AHN, Jun 2025
'''

import argparse
import os
import random

N_SIM = int(1E6) # Number of simulations to run

#--------------
def usage():
    name = os.path.basename( __file__)
    msg = f'''
    Name:
      {name}: Number of coin tosses needed on average to get two heads in a row 

    Example:
      {name} --run

''' 
    msg += '\n '
    return msg 

#-------------
def main():
    
    parser = argparse.ArgumentParser( usage=usage())
    parser.add_argument('--run', action='store_true', help='Run the simulation', required=True)
    args = parser.parse_args()

    run_simulation_2()
    compute_expected_value_2()
    run_simulation_3()
    compute_expected_value_3()

#-----------------------
def run_simulation_2():

    total_tosses = 0
    for _ in range(N_SIM):
        tosses = 0
        last_was_head = False
        
        while True:
            tosses += 1
            if random.choice(['H', 'T']) == 'H':
                if last_was_head:
                    break
                last_was_head = True
            else:
                last_was_head = False
        
        total_tosses += tosses
    
    average_tosses = total_tosses / N_SIM
    print(f'Average number of tosses to get two heads in a row: {average_tosses:.2f}')
    
#-----------------------    
def run_simulation_3():
    """ Run a simulation to compute the average number of tosses needed to get three heads in a row """
    total_tosses = 0
    for _ in range(N_SIM):
        tosses = 0
        head_count = 0
        
        while True:
            tosses += 1
            if random.choice(['H', 'T']) == 'H':
                head_count += 1
                if head_count == 3:
                    break
            else:
                head_count = 0
        
        total_tosses += tosses
    
    average_tosses = total_tosses / N_SIM
    print(f'Average number of tosses to get three heads in a row: {average_tosses:.2f}')    

#---------------------------------    
def compute_expected_value_2():
    """ Recursively compute the expected values for the number of tosses needed for 2 heads in a row """
    DEPTH = 64
    fibo = [0] * (DEPTH + 1)
    fibo[0] = 0
    fibo[1] = 2
    fibo[2] = 3
    for i in range(3, DEPTH + 1):
        fibo[i] = fibo[i - 1] + fibo[i - 2]
    
    prev_p_2_run = 0.0
    E = 0.0
    for length in range(1, DEPTH + 1):
        p_2_run = 1 - fibo[length] / (2 ** length)
        pdf = p_2_run - prev_p_2_run
        prev_p_2_run = p_2_run
        x_times_fx = length * pdf
        E += x_times_fx  
        #print(f'Length {length}: E = {E:.2f}, p(2 runs) = {p_2_run:.4f}, pdf = {pdf:.4f}, x * f(x) = {x_times_fx:.4f}')
        
    print(f'Expected number of tosses to get two heads in a row: {E:.2f}')

#--------------------------------    
def compute_expected_value_3():
    """ Recursively compute the expected values for the number of tosses needed for 3 heads in a row """
    DEPTH = 128
    count_no_3_runs = [0] * (DEPTH + 1)
    count_no_3_runs[0] = 1  # Base case: 0 tosses, no runs
    count_no_3_runs[1] = 2  # 1 toss, no runs (H or T)
    count_no_3_runs[2] = 4  # 2 tosses, no runs (HH, HT, TH, TT)
    for i in range(3, DEPTH + 1):
        #count_no_3_runs[i] = 2 * count_no_3_runs[i - 1] + 2 # copilot is an idiot
        count_no_3_runs[i] = count_no_3_runs[i - 1] + count_no_3_runs[i - 2] + count_no_3_runs[i - 3]
    
    prev_p_3_run = 0.0
    E = 0.0
    for length in range(1, DEPTH + 1):
        p_3_run = 1 - count_no_3_runs[length] / (2 ** length)
        pdf = p_3_run - prev_p_3_run
        prev_p_3_run = p_3_run
        x_times_fx = length * pdf
        E += x_times_fx  
        #print(f'Length {length}: E = {E:.2f}, p(2 runs) = {p_2_run:.4f}, pdf = {pdf:.4f}, x * f(x) = {x_times_fx:.4f}')

    print(f'Expected number of tosses to get three heads in a row: {E:.2f}')


main()
