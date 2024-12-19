import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('himanshi.csv')
print(df.head())
#find the country with max number of deaths yearwise
df1=df.groupby('Year')['Deaths'].max()
print(df1)
