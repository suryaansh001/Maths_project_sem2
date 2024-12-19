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

# 1. Plotting Total Deaths vs CO2 Emissions (Overall)
plt.figure(figsize=(12, 6))
plt.scatter(df['Global CO2 Emissions'], df['Total Deaths'])
plt.xlabel('Global CO2 Emissions')
plt.ylabel('Total Deaths')
plt.title('Scatter Plot of Total Deaths vs Global CO2 Emissions')
plt.grid(True)
plt.show()

# 2. Correlation between CO2 Emissions and Total Deaths (Overall)
correlation_total_deaths_co2 = df[['Total Deaths', 'Global CO2 Emissions']].corr()
print("Correlation between Total Deaths and CO2 Emissions:")
print(correlation_total_deaths_co2)

# 3. Plotting Total Deaths by Disaster Type vs CO2 Emissions

# Group by 'Disaster Type' and aggregate deaths
disaster_deaths = df.groupby('Disaster Type')['Total Deaths'].sum().reset_index()

# Plot the relationship between Deaths by Disaster Type and CO2 Emissions
plt.figure(figsize=(12, 6))
sns.barplot(x='Total Deaths', y='Disaster Type', data=disaster_deaths, palette='viridis')
plt.xlabel('Total Deaths')
plt.ylabel('Disaster Type')
plt.title('Total Deaths by Disaster Type')
plt.grid(True)
plt.show()
#plot line plot of total deaths by disaster type and co2 emissions
plt.figure(figsize=(12,6))
sns.lineplot(data=df, x='Year', y='Total Deaths', hue='Disaster Type')
sns.lineplot(data=df, x='Year', y='Global CO2 Emissions', color='black', label='CO2 Emissions')
plt.xlabel('Year')
plt.ylabel('Total Deaths')
plt.title('Total Deaths by Disaster Type and CO2 Emissions Over Time')
plt.legend()
plt.grid(True)
plt.show()


# 4. Correlation between CO2 Emissions and Deaths by Disaster Type
# You can calculate the correlation between CO2 emissions and total deaths for each disaster type
# First, group by 'Year' and 'Disaster Type' and calculate total deaths per disaster per year
disaster_yearly_deaths = df.groupby(['Year', 'Disaster Type'])[['Total Deaths', 'Global CO2 Emissions']].sum().reset_index()

# Now calculate the correlation for each disaster type
disaster_types = df['Disaster Type'].unique()
for disaster in disaster_types:
    disaster_data = disaster_yearly_deaths[disaster_yearly_deaths['Disaster Type'] == disaster]
    correlation = disaster_data[['Total Deaths', 'Global CO2 Emissions']].corr().iloc[0, 1]
    print(f"Correlation between {disaster} deaths and CO2 emissions: {correlation}")

# 5. Correlation heatmap between Deaths (Total Deaths, Extreme Weather, Floods, Droughts) and CO2 Emissions
# First, filter the data for specific disaster types like extreme weather, floods, and droughts
extreme_weather = df[df['Disaster Type'] == 'Extreme weather events']
floods = df[df['Disaster Type'] == 'Floods']
droughts = df[df['Disaster Type'] == 'Droughts']

# Create a new DataFrame for these specific disaster types
disaster_subsets = pd.DataFrame({
    'Extreme Weather Deaths': extreme_weather['Total Deaths'],
    'Flood Deaths': floods['Total Deaths'],
    'Drought Deaths': droughts['Total Deaths'],
    'Global CO2 Emissions': df['Global CO2 Emissions']
})

# Calculate the correlation matrix
correlation_matrix = disaster_subsets.corr()

# Plot the heatmap of correlations
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix: Deaths by Disaster Type vs CO2 Emissions')
plt.show()

# 6. Plotting Deaths from Extreme Weather, Floods, and Droughts Over Time
plt.figure(figsize=(12, 6))
plt.plot(extreme_weather['Year'], extreme_weather['Total Deaths'], label='Extreme Weather Deaths', marker='o')
plt.plot(floods['Year'], floods['Total Deaths'], label='Flood Deaths', marker='o')
plt.plot(droughts['Year'], droughts['Total Deaths'], label='Drought Deaths', marker='o')
plt.xlabel('Year')
plt.ylabel('Total Deaths')
plt.title('Deaths Over Time from Extreme Weather, Floods, and Droughts')
plt.legend()
plt.grid(True)
plt.show()

# 7. Plotting Deaths vs CO2 Emissions for Extreme Weather, Floods, and Droughts
plt.figure(figsize=(12, 6))
plt.scatter(extreme_weather['Global CO2 Emissions'], extreme_weather['Total Deaths'], label='Extreme Weather', alpha=0.7)
plt.scatter(floods['Global CO2 Emissions'], floods['Total Deaths'], label='Floods', alpha=0.7)
plt.scatter(droughts['Global CO2 Emissions'], droughts['Total Deaths'], label='Droughts', alpha=0.7)
plt.xlabel('Global CO2 Emissions')
plt.ylabel('Total Deaths')
plt.title('Scatter Plot: Deaths vs CO2 Emissions for Different Disaster Types')
plt.legend()
plt.grid(True)
plt.show()
