import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_path = 'clean_water_and_sanitation.csv'  # Correct the path by removing the space

# Read the CSV file
data = pd.read_csv(file_path)

# Display the first few rows of the data to understand its structure
print(data.head())

# Drop the null values
#data = data.dropna()

# Plot the bar graph for the data column 'Year' and 'Share of the population using safely managed drinking water services'
plt.figure(figsize=(12, 6))
plt.bar(data['Year'], data['Share of the population using safely managed drinking water services'], color='b', label='Drinking Water')
plt.xlabel('Year')
plt.ylabel('Percentage (%)')
plt.title('Trend of Safely Managed Drinking Water Services')
plt.grid(True)
plt.legend()
plt.show()