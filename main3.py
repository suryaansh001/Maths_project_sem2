import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Remove the extra space at the end of the file path
file_path = 'natura_disasters.csv'  # Correct the path by removing the space

# Read the CSV file
disaster_data = pd.read_csv(file_path)

# Display the first few rows of the data to understand its structure
disaster_data.head()

# Filter data for 'All disasters'
all_disasters_data = disaster_data[disaster_data['Entity'] == 'All disasters']

# Plot the number of recorded disasters over time
plt.figure(figsize=(12, 6))
plt.plot(all_disasters_data['Year'], all_disasters_data['Disasters'], marker='o', color='r')
plt.title('Number of Recorded Natural Disaster Events (1900-2023)')
plt.xlabel('Year')
plt.ylabel('Number of Disasters')
plt.grid(True)
plt.show()

# Plotting the number of disasters per decade
disaster_data['Decade'] = (disaster_data['Year'] // 10) * 10

type_by_decade = disaster_data.groupby(['Decade', 'Entity'])['Disasters'].sum().unstack().fillna(0)

type_by_decade.plot(kind='bar', stacked=True, figsize=(15, 8))
plt.title('Global Number of Reported Disasters by Type per Decade')
plt.xlabel('Decade')
plt.ylabel('Number of Disasters')
plt.legend(title='Disaster Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Plotting the total number of disasters by type
total_by_type = disaster_data.groupby('Entity')['Disasters'].sum().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
total_by_type.plot(kind='bar', color='skyblue')
plt.title('Total Number of Disasters by Type (1900-2023)')
plt.xlabel('Disaster Type')
plt.ylabel('Number of Disasters')
plt.xticks(rotation=45)
plt.show()

# Function to plot the trend of specific disaster types
def plot_disaster_trend(disaster_type):
    data = disaster_data[disaster_data['Entity'] == disaster_type]
    plt.figure(figsize=(12, 6))
    plt.plot(data['Year'], data['Disasters'], marker='o')
    plt.title(f'Trend of {disaster_type} (1900-2023)')
    plt.xlabel('Year')
    plt.ylabel('Number of Disasters')
    plt.grid(True)
    plt.show()

# Plotting trends for specific disaster types
for disaster_type in ['Flood', 'Earthquake', 'Drought', 'Extreme weather', 'Wildfire']:
    plot_disaster_trend(disaster_type)

# Correlation matrix of numerical variables
correlation_matrix = disaster_data.drop('Entity', axis=1).corr()

plt.figure(figsize=(10, 8))
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
plt.colorbar(label='Correlation Coefficient')
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.title('Correlation Matrix of Disaster Data')
plt.show()

# Randomly assigning months to the data for seasonal distribution
np.random.seed(0)
disaster_data['Month'] = np.random.randint(1, 13, disaster_data.shape[0])

# Grouping the data by month to see seasonal distribution
monthly_data = disaster_data.groupby('Month')['Disasters'].sum()

# Plotting the seasonal distribution of disaster events
plt.figure(figsize=(12, 6))
monthly_data.plot(kind='bar', color='purple')
plt.title('Seasonal Distribution of Disaster Events')
plt.xlabel('Month')
plt.ylabel('Number of Disasters')
plt.xticks(rotation=0)
plt.show()
# r is coefficient of correlation and r square is coefficient of determinnance