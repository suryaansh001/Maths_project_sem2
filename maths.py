import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_csv('population.csv')
print(df.head(10))
print(df.columns)

desc_stats = df.describe()
print("descriptive statistics:")
print(desc_stats)

pop_cols = ['1980', '2000', '2010', '2021', '2022', '2030', '2050']
existing_cols = []
for col in pop_cols:
    if col in df.columns:
        existing_cols.append(col)

print(existing_cols)

df['total_pop'] = df[existing_cols].sum(axis=1)
corr_cols = existing_cols + ['total_pop']
corr_matrix = df[corr_cols].corr()
print("correlation matrix:")
print(corr_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('correlation matrix')
plt.show()

first_five = df.head(5)
plt.figure(figsize=(12, 6))
sns.boxplot(data=first_five[existing_cols])
plt.xlabel('year')
plt.ylabel('population')
plt.title('box plot of population for the first five countries')
plt.show()

# Line graph to analyze population growth pattern
plt.figure(figsize=(12, 6))
for country in df['country'].unique():
    country_data = df[df['country'] == country]
    plt.plot(existing_cols, country_data[existing_cols].values.flatten(), label=country)

plt.xlabel('year')
plt.ylabel('population')
plt.title('population growth pattern')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True)
plt.show()