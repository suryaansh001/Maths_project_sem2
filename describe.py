import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'merged_disaster_co2_data.csv'
data = pd.read_csv(file_path)


data = data.dropna(subset=['Disaster Type', 'Total Deaths', 'Year'])
print(data.describe())