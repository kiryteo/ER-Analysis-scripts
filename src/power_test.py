import numpy as np
from scipy.stats import mannwhitneyu
import statsmodels.stats.power as smp

def mannwhitneyu_power_analysis(effect_size, alpha, power, n1=None, n2=None):
    """
    Perform power analysis for the Mann-Whitney U test.

    Parameters:
        effect_size (float): The effect size (Cohen's d) to detect.
        alpha (float): The significance level (Type I error probability).
        power (float): The desired statistical power (1 - Type II error probability).
        n1 (int, optional): Sample size of group 1. If not provided, the function will calculate it.
        n2 (int, optional): Sample size of group 2. If not provided, the function will calculate it.

    Returns:
        tuple: Sample sizes required for group 1 and group 2.
    """
    if n1 is None and n2 is None:
        raise ValueError("At least one of n1 or n2 must be provided.")

    if n1 is not None and n2 is not None:
        nobs1, nobs2 = n1, n2
    elif n1 is None:
        # Estimate n1 based on n2 and effect size
        nobs2 = n2
        nobs1 = smp.tt_ind_solve_power(effect_size, nobs2, alpha, power, alternative='two-sided')
    else:
        # Estimate n2 based on n1 and effect size
        nobs1 = n1
        nobs2 = smp.tt_ind_solve_power(effect_size, nobs1, alpha, power, alternative='two-sided')

    return int(np.ceil(nobs1)), int(np.ceil(nobs2))

# Example usage:
effect_size = 0.5   # Cohen's d effect size
alpha = 0.05       # Significance level (Type I error probability)
power = 0.8        # Desired statistical power (1 - Type II error probability)

# Calculate sample sizes required for each group
n1, n2 = mannwhitneyu_power_analysis(effect_size, alpha, power, 1000, 1000)

print("Sample size for Group 1:", n1)
print("Sample size for Group 2:", n2)
