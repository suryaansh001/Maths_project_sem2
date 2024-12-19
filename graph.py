import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('merged_natural_disasters_co2.csv')
#find the top 5 countries with the highest frequency of natural disasters
top_5_countries = df.groupby("Country")["Natural Disasters Frequency"].sum().sort_values(ascending=False).head(5)
print(top_5_countries)

# Aggregate data to calculate total frequency per year
frequency_by_year = df.groupby("Year")["Natural Disasters Frequency"].sum().reset_index()

# Plot the frequency by year
plt.figure(figsize=(10, 6))
plt.bar(frequency_by_year["Year"], frequency_by_year["Natural Disasters Frequency"], color='skyblue')
plt.title("Frequency of Natural Disasters by Year")
plt.xlabel("Year")
plt.ylabel("Frequency of Disasters")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
#plot the grpah of frequency of natural disasters by year in every country
df_pivot = df.pivot_table(values='Natural Disasters Frequency', index='Year', columns='Country', aggfunc='sum', fill_value=0)
df_pivot.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='tab20')
plt.xlabel('Year')
plt.ylabel('Frequency of Natural Disasters')
plt.title('Frequency of Natural Disasters by Year and Country')
plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Sample data for the specified countries
data = {
    "Country": ["China", "India", "Indonesia", "Japan", "Bangladesh"],
    "Frequency": [984, 688, 570, 378, 327]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Plot the graph
plt.figure(figsize=(10, 6))
plt.bar(df["Country"], df["Frequency"], color=['red', 'blue', 'green', 'orange', 'purple'])

# Add annotations to display frequencies above the bars
for index, value in enumerate(df["Frequency"]):
    plt.text(index, value + 10, str(value), ha='center', fontsize=10)

# Add title and labels
plt.title("Frequency of Natural Disasters by Country", fontsize=14)
plt.xlabel("Country", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
