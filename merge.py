import pandas as pd

# Read the CSV files
df1 = pd.read_csv('CO2_Emissions_1960-2018.csv')
df2 = pd.read_csv('deaths-from-natural-disasters-by-type.csv')


# Merge the DataFrames on the 'year' column
merged_df = df1.merge(df2, on='year', how='outer').merge(df3, on='year', how='outer')

# Save the merged DataFrame to a new CSV file
merged_df.to_csv('merged_file.csv', index=False)