# import pandas as pd
# import numpy as np
# import pandas as pd
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt

# df=pd.read_csv('rak.csv')

# #descriptive stats
# mean_columns1=df.groupby('Entity')[' Proportion of groundwater bodies with good ambient water quality '].mean()
# print(mean_columns1)
# # mean_columns2=df.groupby('Entity')[' Proportion of surface water bodies with good ambient water quality '].mean()
# # print(mean_columns2)
# # mean_columns3=df.groupby('Entity')[' Proportion of groundwater bodies with poor ambient water quality '].mean()
# # print(mean_columns3)
# #correlation
# df.columns = df.columns.str.replace('\\t', '', regex=False).str.strip() 
# print(df.columns)
# dfCorr=df[[' Proportion of groundwater bodies with good ambient water quality ''	 Proportion of open water bodies with good ambient water quality ']]
# correlation_matrix=dfCorr.corr()

# print(correlation)


#code for linear regression and clustering

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler

# df=pd.read_csv('rak.csv')

# df.columns = df.columns.str.replace('\\t', '', regex=False).str.strip() 
# print(df.columns)
# dfCorr=df[[' Proportion of groundwater bodies with good ambient water quality ''	 Proportion of open water bodies with good ambient water quality ']]

