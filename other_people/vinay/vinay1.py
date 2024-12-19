import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('final_death.csv')
print(df.head())
print(df.columns)
#remove columns named 'Unnamed: 35','Unnamed: 36', 'total', 'total.1'
df.drop(['Unnamed: 35'], axis=1, inplace=True)
print(df.columns)
df.dropna(inplace=True)
# #calculate the sum  values of the columns corresponding to the same 'Entity' and create a new csv with total deaths of teh country and the country name

#plot the total deaths of the countries
df2=pd.read_csv('final_death.csv')
#remove the row 'G20'
df2.drop(df2[df2['Entity']=='G20'].index, inplace=True)
#find the top5 countriesl with the highest total deaths
df3=df2.nlargest(5,'Total Deaths')
print(df3)  
#now put them in a list
countries=df3['Entity'].tolist()
print(countries)
