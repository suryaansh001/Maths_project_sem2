import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns




# Load the CSV file into a pandas DataFrame
df = pd.read_csv('merged_disaster_co2_data.csv')
df.columns = df.columns.str.strip()  # Strip any leading/trailing spaces from column names
df = df.dropna()  # Remove rows with missing values
df = df.drop(columns='Code')  # Drop the 'Code' column

# Remove certain disaster types for analysis
df = df[df['Disaster Type'] != 'All disasters excluding extreme temperature']
df = df[df['Disaster Type'] != 'All disasters excluding earthquakes']

# Print the first 10 rows to verify
print(df.head(10))

# 1. Bar chart for total deaths by disaster type
plt.figure(figsize=(12, 6))
total_deaths_by_disaster = df.groupby('Disaster Type')['Total Deaths'].sum().sort_values()
total_deaths_by_disaster.plot(kind='bar', color='skyblue')
plt.xlabel('Disaster Type')
plt.ylabel('Total Deaths')
plt.title('Total Deaths by Disaster Type')
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.show()

# 2. Stacked bar chart for total deaths by disaster type over the years
df_pivot = df.pivot_table(values='Total Deaths', index='Year', columns='Disaster Type', aggfunc='sum', fill_value=0)
df_pivot.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='tab20')
plt.xlabel('Year')
plt.ylabel('Total Deaths')
plt.title('Total Deaths by Disaster Type Over the Years')
plt.legend(title='Disaster Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Ensure you're working with the latest cleaned DataFrame
# 1. Bar chart for total deaths by disaster type
plt.figure(figsize=(12, 6))
total_deaths_by_disaster = df.groupby('Disaster Type')['Total Deaths'].sum().sort_values()
total_deaths_by_disaster.plot(kind='bar', color='skyblue')
plt.xlabel('Disaster Type')
plt.ylabel('Total Deaths')
plt.title('Total Deaths by Disaster Type')
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.show()

# 2. Stacked bar chart for total deaths by disaster type over the years
df_pivot = df.pivot_table(values='Total Deaths', index='Year', columns='Disaster Type', aggfunc='sum', fill_value=0)
df_pivot.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='tab20')
plt.xlabel('Year')
plt.ylabel('Total Deaths')
plt.title('Total Deaths by Disaster Type Over the Years')
plt.legend(title='Disaster Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.show()

# 3. Line plot for total deaths over the years
plt.figure(figsize=(12, 6))
df_yearly_deaths = df.groupby('Year')['Total Deaths'].sum()
df_yearly_deaths.plot(kind='line', marker='o', color='purple')
plt.xlabel('Year')
plt.ylabel('Total Deaths')
plt.title('Total Deaths Over the Years')
plt.grid(True)
plt.show()


# 5. Scatter plot: Total Deaths vs CO2 Emissions (if CO2 emissions column exists)
if 'CO2 Emissions' in df.columns:
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x='CO2 Emissions', y='Total Deaths', hue='Disaster Type', palette='Set1')
    plt.title('Total Deaths vs CO2 Emissions by Disaster Type')
    plt.xlabel('CO2 Emissions')
    plt.ylabel('Total Deaths')
    plt.grid(True)
    plt.show()

# 6. Bar chart for deaths by region (if "Region" or "Country" is in the dataset)
if 'Region' in df.columns:
    plt.figure(figsize=(12, 6))
    deaths_by_region = df.groupby('Region')['Total Deaths'].sum().sort_values(ascending=False)
    deaths_by_region.plot(kind='bar', color='green')
    plt.xlabel('Region')
    plt.ylabel('Total Deaths')
    plt.title('Total Deaths by Region')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)
    plt.show()
