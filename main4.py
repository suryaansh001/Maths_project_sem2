import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("datasets/deaths-from-natural-disasters-by-type.csv")
print(df.head())
print ("-"*10)
#drop a row
df=df.drop(1)
df=df.drop(2)

print(df.head())

# Line plot
# plt.figure(figsize=(18,18))
# plt.plot(df['Year'], df['Disasters'], marker='o')

# plt.title('Number of Recorded Natural Disaster Events (1900-2023)')
# plt.xlabel('Year')
# plt.ylabel('Number of Disasters')
# plt.grid(True)
# plt.show()

# Bar plot
plt.figure(figsize=(18,18))
plt.bar(df['Year'], df['Disasters'])

plt.title('Number of Recorded Natural Disaster Events (1900-2024)')
plt.xlabel('Year')
plt.ylabel('Number of Disasters')
plt.grid(True)
plt.show()
#----------------------------------------------------------------------------
#plot the bar graph of frequency of disasters and disaster type
plt.figure(figsize=(12,6))
plt.bar(df['Entity'],df['Disasters'])
plt.xlabel('Disaster Type')
plt.xticks(rotation=45)
plt.ylabel('Number of Disasters')
plt.grid(True)
plt.show()
#-----------------------------------------------------------------------------
total_by_type = df.groupby('Entity')['Disasters'].sum().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
total_by_type.plot(kind='bar', color='skyblue')
plt.title('Total Number of Disasters by Type (1900-2023)')
plt.xlabel('Disaster Type')
plt.ylabel('Number of Disasters')
plt.xticks(rotation=45)
plt.show()

#--------------------------------------------------------------------------------
# Function to plot the trend of specific disaster types
def plot_disaster_trend(disaster_type):
    data = df[df['Entity'] == disaster_type]
    plt.figure(figsize=(12, 6))
    plt.plot(data['Year'], data['Disasters'], marker='o')
    plt.title(f'Trend of {disaster_type} (1900-2023)')
    plt.xlabel('Year')
    plt.ylabel('Number of Disasters')
    plt.grid(True)
    plt.show()
#call the function
plot_disaster_trend('Flood')
plot_disaster_trend('Earthquake')
plot_disaster_trend('Drought')
plot_disaster_trend('Extreme weather')
plot_disaster_trend('Wildfire')
#---------------------------------------------------------------------------------
#find all the descriptions of the data
print(df.describe())
#find the correlation of the data
print(df.corr())
#plot the correlation matrix
import seaborn as sns
sns.heatmap(df.corr(),annot=True)
plt.show()
#--------------------------------------------------------------------------------
#plot the histogram of the data
plt.figure(figsize=(12,6))
plt.hist(df['Disasters'],bins=10)
plt.xlabel('Number of Disasters')
plt.ylabel('Frequency')
plt.title('Histogram of Number of Disasters')
plt.show()