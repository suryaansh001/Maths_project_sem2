import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = "../datasets/clean_water_and_sanitation.csv"
data = pd.read_csv(file_path)

# Drop the null values
data = data.dropna()

# Initialize seaborn for aesthetics
sns.set(style="whitegrid")

# 1. Line Plot: Trends over Years
plt.figure(figsize=(12, 6))
sns.lineplot(data=data, x="Year", y="Share of the population using safely managed drinking water services", label='Drinking Water', color='blue')
sns.lineplot(data=data, x="Year", y="Share of the population using safely managed sanitation services", label='Sanitation', color='green')
plt.title("Trend of Safely Managed Drinking Water and Sanitation Services", fontsize=14)
plt.ylabel("Percentage (%)", fontsize=12)
plt.xlabel("Year", fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

# 2. Bar Plot: Yearly Comparison
plt.figure(figsize=(12, 6))
sns.barplot(data=data, x="Year", y="Share of the population using safely managed drinking water services", color='blue', label='Drinking Water')
sns.barplot(data=data, x="Year", y="Share of the population using safely managed sanitation services", color='green', label='Sanitation', alpha=0.7)
plt.title("Yearly Comparison of Drinking Water and Sanitation Services", fontsize=14)
plt.ylabel("Percentage (%)", fontsize=12)
plt.xlabel("Year", fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

# 3. Histogram: Distribution of Services
plt.figure(figsize=(12, 6))
sns.histplot(data["Share of the population using safely managed drinking water services"], kde=True, bins=20, color="blue", label="Drinking Water")
sns.histplot(data["Share of the population using safely managed sanitation services"], kde=True, bins=20, color="green", label="Sanitation", alpha=0.7)
plt.title("Distribution of Water and Sanitation Services (Percentage)", fontsize=14)
plt.xlabel("Percentage (%)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.legend()
plt.show()

# 4. Box Plot: Distribution of Services
plt.figure(figsize=(12, 6))
sns.boxplot(data=data[['Share of the population using safely managed drinking water services', 'Share of the population using safely managed sanitation services']])
plt.title("Box Plot of Water and Sanitation Services", fontsize=14)
plt.ylabel("Percentage (%)", fontsize=12)
plt.xlabel("Service Type", fontsize=12)
plt.xticks([0, 1], ['Drinking Water', 'Sanitation'])
plt.show()

# 5. Scatter Plot: Drinking Water vs Sanitation
plt.figure(figsize=(12, 6))
sns.scatterplot(data=data, x="Share of the population using safely managed drinking water services", y="Share of the population using safely managed sanitation services", hue="Year", palette="coolwarm", s=100)
plt.title("Drinking Water vs Sanitation Services", fontsize=14)
plt.xlabel("Drinking Water Services (%)", fontsize=12)
plt.ylabel("Sanitation Services (%)", fontsize=12)
plt.legend(title="Year")
plt.grid(True)
plt.show()