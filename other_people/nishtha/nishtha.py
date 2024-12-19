# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load the Excel file into a pandas DataFrame
# df = pd.read_excel('nishtha.xlsx')

# # Print the first 10 rows to verify
# print(df.head())

# # Keep only the columns 'STATES', 'YEAR', 'IMORTAL TRAFFICKING', and 'TOTAL CRIMES AGAINST WOMEN'
# df = df[['STATES', 'YEAR', 'IMORTAL TRAFFICKING', 'TOTAL CRIMES AGAINST WOMEN']]

# # Print the columns to verify
# print(df.columns)
# print(df.head())
# #save the cleaned data to a new csv file
# df.to_csv('cleaned_data.csv', index=False)
import pandas as pd

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('cleaned_data.csv')

# Pivot the DataFrame
pivot_df = df.pivot(index='STATES', columns='YEAR', values='TOTAL CRIMES AGAINST WOMEN')

# Rename the columns to 'year1', 'year2', etc.
pivot_df.columns = [f'year{col}' for col in pivot_df.columns]

# Print the transformed DataFrame
print(pivot_df)

# Save the transformed DataFrame to a new CSV file
pivot_df.to_csv('transformed_data.csv')