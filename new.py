import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('merged_disaster_co2_data.csv')
df.columns = df.columns.str.strip()  # Strip any leading/trailing spaces from column names
df = df.dropna()
df = df.drop(columns='Code')
print(df.head(10))
#remove 'All disasters excluding extreme temperature' and 'All disasters excluding earthquakes' from the 'Entity' column ''
df = df[df['Disaster Type'] != 'All disasters excluding extreme temperature']
df = df[df['Disaster Type'] != 'All disasters excluding earthquakes']
print(df.head(10))
#plot the bar graph of frequency of disasters and total deaths
plt.figure(figsize=(12,6))
plt.bar(df['Disaster Type'],df['Total Deaths'])
plt.xlabel('Disaster Type')
plt.xticks(rotation=45)
plt.ylabel('Total Deaths')
plt.grid(True)
plt.show()
#plot the line graph of frequency of disasters and total deaths
plt.figure(figsize=(36,18))
plt.plot(df['Year'], df['Total Deaths'], marker='o')
plt.xlabel('Year')  
plt.ylabel('Total Deaths')
plt.title('Total Deaths from Natural Disasters (1900-2023)')
plt.grid(True)
plt.show()  
#scatter plot
plt.figure(figsize=(12,6))  
plt.scatter(df['Year'], df['Total Deaths'])
plt.xlabel('Year')
plt.ylabel('Total Deaths')
plt.title('Total Deaths from Natural Disasters (1900-2023)')
plt.grid(True)
plt.show()
plt.figure(figsize=(12,6))

plt.figure(figsize=(8,8))
df['Disaster Type'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Proportion of Different Disaster Types')
plt.ylabel('')
plt.show()
#bar chart of year and co2 emissions
plt.figure(figsize=(36,18))
plt.bar(df['Year'], df['Global CO2 Emissions'])
plt.xlabel('Year')  
plt.ylabel('CO2 Emissions') 
plt.title('CO2 Emissions (1900-2023)')
plt.grid(True)
plt.show()
# #histogram of co2 emissions and total deaths
# plt.figure(figsize=(12,6))
# plt.hist(df['Global CO2 Emissions'], bins=10)
# plt.xlabel('CO2 Emissions')
# plt.ylabel('Frequency')
# plt.title('Histogram of CO2 Emissions')
# plt.show()
# plt.figure(figsize=(12,6))
# plt.hist(df['Total Deaths'], bins=10)
# plt.xlabel('Total Deaths')
# plt.ylabel('Frequency')
# plt.title('Histogram of Total Deaths')
# plt.show()
# plt.show()
# plt.figure(figsize=(12,6))
# plt.hist(df['Total Deaths'], bins=10, color='blue')
# plt.xlabel('Total Deaths')
# plt.ylabel('Frequency')
# plt.title('Histogram of Total Deaths')
# plt.show()
# #line plot of year and co2 emissions
# plt.figure(figsize=(36,18))
# plt.plot(df['Year'], df['Global CO2 Emissions'], marker='o', color='red')
# plt.xlabel('Year')
# plt.ylabel('CO2 Emissions')
# plt.title('Global CO2 Emissions (1900-2023)')
# plt.grid(True)
# plt.show()
#line plot of year and co2 emissions
plt.figure(figsize=(36,18))
plt.plot(df['Year'], df['Global CO2 Emissions'], marker='o')
plt.xlabel('Year')
plt.ylabel('CO2 Emissions')
plt.title('Global CO2 Emissions (1900-2023)')
plt.grid(True)
plt.show()


plt.figure(figsize=(12,6))
plt.scatter(df['Global CO2 Emissions'], df['Total Deaths'])
plt.xlabel('Global CO2 Emissions')
plt.ylabel('Total Deaths')
plt.title('Scatter Plot of CO2 Emissions vs Total Deaths')
plt.grid(True)
plt.show()

fig, ax1 = plt.subplots(figsize=(12,6))

ax1.set_xlabel('Year')
ax1.set_ylabel('Total Deaths', color='tab:red')
ax1.plot(df['Year'], df['Total Deaths'], color='tab:red', label='Total Deaths')
ax1.tick_params(axis='y', labelcolor='tab:red')

ax2 = ax1.twinx()
ax2.set_ylabel('Global CO2 Emissions', color='tab:blue')
ax2.plot(df['Year'], df['Global CO2 Emissions'], color='tab:blue', label='CO2 Emissions')
ax2.tick_params(axis='y', labelcolor='tab:blue')

fig.tight_layout()
plt.title('Total Deaths and CO2 Emissions Over Time')
plt.grid(True)
plt.show()

import seaborn as sns

plt.figure(figsize=(8,6))
correlation_matrix = df[['Total Deaths', 'Global CO2 Emissions']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# #find the correlation of the data
# print(df.corr())
#plot thegraph betweeen deaths due to extreme weather and year
plt.figure(figsize=(12,6))
plt.plot(df['Year'], df['Total Deaths'], marker='o')
plt.xlabel('Year')
plt.ylabel('Total Deaths')
plt.title('Total Deaths from Extreme Weather Events (1900-2023)')
plt.grid(True)
plt.show()
#plot the graph between deaths due to flood and year
plt.figure(figsize=(12,6))
plt.plot(df['Year'], df['Total Deaths'], marker='o')
plt.xlabel('Year')
plt.ylabel('Total Deaths')
plt.title('Total Deaths from Floods (1900-2023)')
plt.grid(True)
plt.show()
#plot the graph between deaths due to drought and year
plt.figure(figsize=(12,6))
plt.plot(df['Year'], df['Total Deaths'], marker='o')
plt.xlabel('Year')
plt.ylabel('Total Deaths')
plt.title('Total Deaths from Droughts (1900-2023)')
plt.grid(True)
plt.show()
#plot the correlation between deaths due to extreme weather and co2 emissions
plt.figure(figsize=(12,6))
plt.scatter(df['Global CO2 Emissions'], df['Total Deaths'])
plt.xlabel('Global CO2 Emissions')
plt.ylabel('Total Deaths')
plt.title('Scatter Plot of CO2 Emissions vs Total Deaths')
plt.grid(True)
plt.show()
#plot the correlation between deaths due to flood and co2 emissions
plt.figure(figsize=(12,6))
plt.scatter(df['Global CO2 Emissions'], df['Total Deaths'])
plt.xlabel('Global CO2 Emissions')
plt.ylabel('Total Deaths')
plt.title('Scatter Plot of CO2 Emissions vs Total Deaths')
plt.grid(True)
plt.show()
#plot the correlation between deaths due to drought and co2 emissions
plt.figure(figsize=(12,6))
plt.scatter(df['Global CO2 Emissions'], df['Total Deaths'])
plt.xlabel('Global CO2 Emissions')
plt.ylabel('Total Deaths')
plt.title('Scatter Plot of CO2 Emissions vs Total Deaths')
plt.grid(True)
plt.show()
# #plot the correlation matrix
# plt.figure(figsize=(12,6))
# sns.heatmap(df.corr(), annot=True)
# plt.title('Correlation Matrix')
# plt.show()
