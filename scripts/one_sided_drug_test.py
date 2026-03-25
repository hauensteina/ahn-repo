from pdb import set_trace as BP
from scipy.stats import binom

# Adjust these to your liking
p0 = 0.5
p1 = 0.8
alpha = 0.05
power_target = 0.9

significance_target = 1 - alpha

for N in range(100):
    for kc in range(N):
        p_false_positive = 1 - binom.cdf( k=kc-1, n=N, p=p0) # P ( kc or more responders with p0 )
        significance = 1 - p_false_positive # we want this large 
        if significance > significance_target: # enough responders to make the old p0 unlikely
            p_false_negative =  binom.cdf( k=kc-1, n=N, p=p1)  # P ( less than kc responders with p1 )
            power = 1 - p_false_negative # We also want this large
            if power > power_target:
                print( f'Found a viable study for p0={p0}, p1={p1}, alpha={alpha}, power_target={power_target}')
                print( f'Study size: {N} min responders for success: {kc} power: {power} significance: {significance}' )
                exit(0)       
