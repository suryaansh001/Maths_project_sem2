import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('population.csv')
print(df.head(10))
print(df.columns)

descstats = df.describe()

print(descstats)

popcols = ['1980', '2000', '2010', '2021', '2022', '2030', '2050']#ignoring the growth rate wale columns
existingcols = []
for col in popcols:
    if col in df.columns:
        existingcols.append(col)
    

print(existingcols)

# print('='*50)
# countrydata = df[df['country'] == 'India']
# print(countrydata[existingcols].values)


firstfive = df.head(5)
plt.figure(figsize=(12, 6))
sns.boxplot(data=firstfive[existingcols])
plt.xlabel('year')
plt.ylabel('population')
plt.title('box plot')
plt.show()

plt.figure(figsize=(12, 6))
for country in df['country'].unique():
    countrydata = df[df['country'] == country]
    plt.plot(existingcols, countrydata[existingcols].values.flatten(), label=country)

plt.xlabel('year')
plt.ylabel('population')
plt.title('population growth pattern')
#plt.legend()#show the labels
plt.grid(True)
plt.show()
