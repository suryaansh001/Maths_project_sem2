import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f

# Load the dataset
file_path = 'student_sleep_patterns.csv'  # Update with the correct file path
data = pd.read_csv(file_path)

# Ensure the dataset has the required columns
if 'University_Year' not in data.columns or 'Sleep_Duration' not in data.columns:
    print("The dataset must contain 'University_Year' and 'Sleep_Duration' columns.")
else:
    # Group data by university year and calculate required values
    groups = data.groupby('University_Year')['Sleep_Duration']
    group_means = groups.mean()
    group_counts = groups.count()
    group_sums = groups.sum()
    grand_mean = data['Sleep_Duration'].mean()

    # Total number of observations
    N = data['Sleep_Duration'].count()
    k = len(group_means)

    # Calculate SSB (Between-Group Sum of Squares)
    SS_B = sum(group_counts * (group_means - grand_mean) ** 2)

    # Calculate SSW (Within-Group Sum of Squares)
    SS_W = sum(groups.apply(lambda x: sum((x - x.mean()) ** 2)))

    # Total Sum of Squares (SS_T)
    SS_T = SS_B + SS_W

    # Degrees of freedom
    df_B = k - 1
    df_W = N - k

    # Mean squares
    MS_B = SS_B / df_B
    MS_W = SS_W / df_W

    # F-statistic
    F_statistic = MS_B / MS_W

    print(f"SSB (Between Groups): {SS_B}")
    print(f"SSW (Within Groups): {SS_W}")
    print(f"Total Sum of Squares: {SS_T}")
    print(f"Degrees of Freedom Between Groups: {df_B}")
    print(f"Degrees of Freedom Within Groups: {df_W}")
    print(f"F Statistic: {F_statistic}")

    # Critical value and hypothesis testing
    alpha = 0.05
    F_critical = f.ppf(1 - alpha, df_B, df_W)

    print(f"F Critical (alpha={alpha}): {F_critical}")

    if F_statistic > F_critical:
        print("Reject the null hypothesis: There is a significant difference between the groups.")
    else:
        print("Fail to reject the null hypothesis: No significant difference between the groups.")

print("descriptive analysis")
print(data.describe())

# 