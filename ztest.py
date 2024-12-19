import pandas as pd
from scipy.stats import f_oneway

# Load data
df = pd.read_csv('merged_disaster_co2_data.csv')

# Group CO2 emissions by disaster type
eq_co2 = df[df['Disaster Type'] == 'Earthquake']['Global CO2 Emissions']
dr_co2 = df[df['Disaster Type'] == 'Drought']['Global CO2 Emissions']
ad_co2 = df[df['Disaster Type'] == 'All disasters']['Global CO2 Emissions']
aex_eq_co2 = df[df['Disaster Type'] == 'All disasters excluding earthquakes']['Global CO2 Emissions']
aex_et_co2 = df[df['Disaster Type'] == 'All disasters excluding extreme temperature']['Global CO2 Emissions']

# ANOVA
f_stat, p_val = f_oneway(eq_co2, dr_co2, ad_co2, aex_eq_co2, aex_et_co2)

# Result
print(f"F-statistic: {f_stat}\nP-value: {p_val}")
if p_val < 0.05:
    print("Significant difference in CO2 emissions between disaster types.")
else:
    print("No significant difference in CO2 emissions between disaster types.")
