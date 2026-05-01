from scipy.stats import binom, norm
import math

# Adjust these to your liking
p0 = 0.5
p1 = 0.8
alpha = 0.05
power_target = 0.9

significance_target = 1 - alpha

#--------------------------------------------------
def fleiss_sample_size(p0, p1, alpha, power):
    """ Fleiss method for one-sided test of proportions, with optional continuity correction. """
    z_alpha = norm.ppf(1 - alpha)       # one-sided
    z_beta  = norm.ppf(power)
    p_bar = (p0 + p1) / 2
    numerator   = (z_alpha * math.sqrt(p0 * (1 - p0)) + z_beta * math.sqrt(p1 * (1 - p1))) ** 2
    denominator = (p1 - p0) ** 2
    n_raw = numerator / denominator
    # Fleiss continuity correction
    n_cc = (n_raw / 4) * (1 + math.sqrt(1 + 2 / (n_raw * abs(p1 - p0)))) ** 2
    return math.ceil(n_raw), math.ceil(n_cc)

n_fleiss, n_fleiss_cc = fleiss_sample_size(p0, p1, alpha, power_target)

# --- Exact binomial search ---
for N in range(1000):
    for kc in range(N):
        p_false_positive = 1 - binom.cdf(k=kc-1, n=N, p=p0)
        significance = 1 - p_false_positive
        if significance > significance_target:
            p_false_negative = binom.cdf(k=kc-1, n=N, p=p1)
            power = 1 - p_false_negative
            if power > power_target:
                print(f"{'Parameter':<30} {'Value'}")
                print("-" * 45)
                print(f"{'p0':<30} {p0}")
                print(f"{'p1':<30} {p1}")
                print(f"{'alpha':<30} {alpha}")
                print(f"{'power target':<30} {power_target}")
                print()
                print(f"{'Method':<30} {'N':>6}  {'Notes'}")
                print("-" * 55)
                print(f"{'Exact binomial':<30} {N:>6}  kc={kc}, power={power:.4f}, sig={significance:.4f}")
                print(f"{'Fleiss (no correction)':<30} {n_fleiss:>6}")
                print(f"{'Fleiss (continuity corr.)':<30} {n_fleiss_cc:>6}")
                exit(0)
                
                