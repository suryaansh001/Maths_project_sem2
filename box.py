import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'merged_disaster_co2_data.csv'
data = pd.read_csv(file_path)

# Check for any missing values in the relevant columns
data = data.dropna(subset=['Disaster Type', 'Total Deaths', 'Year'])

# Get unique disaster types
disaster_types = data['Disaster Type'].unique()

# Plot data for each disaster type
for disaster in disaster_types:
    # Filter data for the specific disaster type
    disaster_data = data[data['Disaster Type'] == disaster]
    yearly_totals = disaster_data.groupby('Year')['Total Deaths'].sum().reset_index()

    # Create the line plot
    plt.figure(figsize=(10, 6))
    plt.plot(yearly_totals['Year'], yearly_totals['Total Deaths'], marker='o', label=disaster)
    plt.title(f'Year-Wise Deaths Due to {disaster}')
    plt.xlabel('Year')
    plt.ylabel('Total Deaths')
    plt.grid(True)
    plt.tight_layout()

    # Show the plot for this disaster type
    plt.show()
