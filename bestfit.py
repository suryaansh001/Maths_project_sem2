# # import pandas as pd
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from sklearn.preprocessing import PolynomialFeatures
# # from sklearn.linear_model import LinearRegression
# # from scipy.stats import zscore

# # # Load the dataset
# # df = pd.read_csv('merged_disaster_co2_data.csv')
# # df.columns = df.columns.str.strip()  # Strip any leading/trailing spaces from column names
# # df = df.dropna()
# # df = df.drop(columns='Code')

# # # Remove 'All disasters excluding extreme temperature' and 'All disasters excluding earthquakes' from the 'Disaster Type' column
# # df = df[df['Disaster Type'] != 'All disasters excluding extreme temperature']
# # df = df[df['Disaster Type'] != 'All disasters excluding earthquakes']

# # # Remove outliers using IQR and mean
# # def remove_outliers(df):
# #     q1 = df['Total Deaths'].quantile(0.25)
# #     q3 = df['Total Deaths'].quantile(0.75)
# #     iqr = q3 - q1
# #     lower_bound = q1 - 1.5 * iqr
# #     upper_bound = q3 + 1.5 * iqr
# #     df_cleaned = df[(df['Total Deaths'] >= lower_bound) & (df['Total Deaths'] <= upper_bound)]
    
# #     # Plot the graph with outliers removed
# #     plt.boxplot(df_cleaned['Total Deaths'])
# #     plt.title('Box Plot of Total Deaths (Outliers Removed)')
# #     plt.xlabel('Total Deaths')
# #     plt.ylabel('Count')
# #     plt.show()
    
# #     return df_cleaned

# # # Box plot with outliers removed and outliers not removed
# # def boxplot(df):
# #     plt.boxplot(df['Total Deaths'])
# #     plt.title('Box Plot of Total Deaths')
# #     plt.xlabel('Total Deaths')
# #     plt.ylabel('Count')
# #     plt.show()

# # # Plot initial boxplot
# # boxplot(df)

# # # Remove outliers and update the DataFrame
# # df = remove_outliers(df)

# # # Function to predict deaths for a given disaster type using polynomial regression
# # def predict_deaths_for_disaster_poly(disaster_type):
# #     df_disaster = df[df['Disaster Type'] == disaster_type]

# #     # Prepare the data: Select Year and Total Deaths for the regression model
# #     X = df_disaster[['Year']]  # Feature: Year
# #     y = df_disaster['Total Deaths']  # Target: Total Deaths

# #     # Remove outliers using Z-score
# #     df_disaster['z_score'] = zscore(df_disaster['Total Deaths'])
# #     df_disaster_cleaned = df_disaster[df_disaster['z_score'].abs() <= 3]

# #     # Prepare cleaned data
# #     X_cleaned = df_disaster_cleaned[['Year']]
# #     y_cleaned = df_disaster_cleaned['Total Deaths']

# #     # Transform the feature to polynomial features (degree 3)
# #     poly = PolynomialFeatures(degree=3)
# #     X_poly = poly.fit_transform(X_cleaned)

# #     # Fit the model
# #     model = LinearRegression()
# #     model.fit(X_poly, y_cleaned)

# #     # Make predictions
# #     df_disaster_cleaned['Predicted Deaths'] = model.predict(X_poly)

# #     # Plotting the data points and the polynomial line of best fit
# #     plt.figure(figsize=(12, 6))
# #     plt.scatter(df_disaster_cleaned['Year'], df_disaster_cleaned['Total Deaths'], color='blue', label='Actual Deaths')
# #     plt.plot(df_disaster_cleaned['Year'], df_disaster_cleaned['Predicted Deaths'], color='red', label='Polynomial Line of Best Fit')
# #     plt.xlabel('Year')
# #     plt.ylabel('Total Deaths')
# #     plt.title(f'Total Deaths Prediction for {disaster_type} (Polynomial Regression)')
# #     plt.legend()
# #     plt.grid(True)
# #     plt.show()

# #     # Predict deaths for future years (for example, the next 10 years)
# #     future_years = np.arange(2024, 2034).reshape(-1, 1)  # Predicting for 10 years ahead
# #     future_years_poly = poly.transform(future_years)  # Transform future years using the same polynomial features
# #     future_deaths = model.predict(future_years_poly)

# #     # Print the predicted deaths for the next 10 years
# #     for year, deaths in zip(future_years.flatten(), future_deaths):
# #         print(f"Predicted deaths in {year}: {deaths:.0f}")

# # # Function to predict deaths for a given disaster type using linear regression
# # def predict_deaths_for_disaster_linear(disaster_type):
# #     df_disaster = df[df['Disaster Type'] == disaster_type]

# #     # Prepare the data: Select Year and Total Deaths for the regression model
# #     X = df_disaster[['Year']]  # Feature: Year
# #     y = df_disaster['Total Deaths']  # Target: Total Deaths

# #     # Remove outliers using Z-score
# #     df_disaster['z_score'] = zscore(df_disaster['Total Deaths'])
# #     df_disaster_cleaned = df_disaster[df_disaster['z_score'].abs() <= 3]

# #     # Prepare cleaned data
# #     X_cleaned = df_disaster_cleaned[['Year']]
# #     y_cleaned = df_disaster_cleaned['Total Deaths']

# #     # Fit the model
# #     model = LinearRegression()
# #     model.fit(X_cleaned, y_cleaned)

# #     # Make predictions
# #     df_disaster_cleaned['Predicted Deaths'] = model.predict(X_cleaned)

# #     # Plotting the data points and the linear line of best fit
# #     plt.figure(figsize=(12, 6))
# #     plt.scatter(df_disaster_cleaned['Year'], df_disaster_cleaned['Total Deaths'], color='blue', label='Actual Deaths')
# #     plt.plot(df_disaster_cleaned['Year'], df_disaster_cleaned['Predicted Deaths'], color='green', label='Linear Line of Best Fit')
# #     plt.xlabel('Year')
# #     plt.ylabel('Total Deaths')
# #     plt.title(f'Total Deaths Prediction for {disaster_type} (Linear Regression)')
# #     plt.legend()
# #     plt.grid(True)
# #     plt.show()

# #     # Predict deaths for future years (for example, the next 10 years)
# #     future_years = np.arange(2024, 2034).reshape(-1, 1)  # Predicting for 10 years ahead
# #     future_deaths = model.predict(future_years)

# #     # Print the predicted deaths for the next 10 years
# #     for year, deaths in zip(future_years.flatten(), future_deaths):
# #         print(f"Predicted deaths in {year}: {deaths:.0f}")

# # # List of disaster types to predict deaths for
# # disaster_types = df['Disaster Type'].unique()

# # # Run the prediction for each disaster type
# # for disaster_type in disaster_types:
# #     print(f"Predictions for {disaster_type} (Polynomial Regression):")
# #     predict_deaths_for_disaster_poly(disaster_type)
# #     print("\n" + "="*50 + "\n")
# #     print(f"Predictions for {disaster_type} (Linear Regression):")
# #     predict_deaths_for_disaster_linear(disaster_type)
# #     print("\n" + "="*50 + "\n")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.stats import zscore

# Load the dataset
df = pd.read_csv('merged_disaster_co2_data.csv')
df.columns = df.columns.str.strip()  # Strip any leading/trailing spaces from column names
df = df.dropna()
df = df.drop(columns='Code')

# # Remove 'All disasters excluding extreme temperature' and 'All disasters excluding earthquakes' from the 'Disaster Type' column
df = df[df['Disaster Type'] != 'All disasters excluding extreme temperature']
df = df[df['Disaster Type'] != 'All disasters excluding earthquakes']

# # Remove outliers using IQR and mean
# def remove_outliers(df):
#     q1 = df['Total Deaths'].quantile(0.25)
#     q3 = df['Total Deaths'].quantile(0.75)
#     iqr = q3 - q1
#     lower_bound = q1 - 1.5 * iqr
#     upper_bound = q3 + 1.5 * iqr
#     df_cleaned = df[(df['Total Deaths'] >= lower_bound) & (df['Total Deaths'] <= upper_bound)]
    
#     # Plot the graph with outliers removed
#     plt.boxplot(df_cleaned['Total Deaths'])
#     plt.title('Box Plot of Total Deaths (Outliers Removed)')
#     plt.xlabel('Total Deaths')
#     plt.ylabel('Count')
#     plt.show()
    
#     return df_cleaned

# # Box plot with outliers removed and outliers not removed
# def boxplot(df):
#     plt.boxplot(df['Total Deaths'])
#     plt.title('Box Plot of Total Deaths')
#     plt.xlabel('Total Deaths')
#     plt.ylabel('Count')
#     plt.show()

# # Plot initial boxplot
# boxplot(df)

# # Remove outliers and update the DataFrame
# df = remove_outliers(df)

# # Function to predict deaths for a given disaster type using polynomial regression
# def predict_deaths_for_disaster_poly(disaster_type, file):
#     df_disaster = df[df['Disaster Type'] == disaster_type]

#     # Prepare the data: Select Year and Total Deaths for the regression model
#     X = df_disaster[['Year']]  # Feature: Year
#     y = df_disaster['Total Deaths']  # Target: Total Deaths

#     # Remove outliers using Z-score
#     df_disaster['z_score'] = zscore(df_disaster['Total Deaths'])
#     df_disaster_cleaned = df_disaster[df_disaster['z_score'].abs() <= 3]

#     # Prepare cleaned data
#     X_cleaned = df_disaster_cleaned[['Year']]
#     y_cleaned = df_disaster_cleaned['Total Deaths']

#     # Transform the feature to polynomial features (degree 3)
#     poly = PolynomialFeatures(degree=3)
#     X_poly = poly.fit_transform(X_cleaned)

#     # Fit the model
#     model = LinearRegression()
#     model.fit(X_poly, y_cleaned)

#     # Make predictions
#     df_disaster_cleaned['Predicted Deaths (Poly)'] = model.predict(X_poly)

#     # Plotting the data points and the polynomial line of best fit
#     plt.figure(figsize=(12, 6))
#     plt.scatter(df_disaster_cleaned['Year'], df_disaster_cleaned['Total Deaths'], color='blue', label='Actual Deaths')
#     plt.plot(df_disaster_cleaned['Year'], df_disaster_cleaned['Predicted Deaths (Poly)'], color='red', label='Polynomial Line of Best Fit')
#     plt.xlabel('Year')
#     plt.ylabel('Total Deaths')
#     plt.title(f'Total Deaths Prediction for {disaster_type} (Polynomial Regression)')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

#     # Predict deaths for future years (for example, the next 10 years)
#     future_years = np.arange(2024, 2034).reshape(-1, 1)  # Predicting for 10 years ahead
#     future_years_poly = poly.transform(future_years)  # Transform future years using the same polynomial features
#     future_deaths = model.predict(future_years_poly)

#     # Save the predicted deaths for the next 10 years to the file
#     file.write(f"Predictions for the next 10 years (Polynomial Regression) for {disaster_type}:\n")
#     for year, deaths in zip(future_years.flatten(), future_deaths):
#         file.write(f"Predicted deaths in {year}: {deaths:.0f}\n")
#     file.write("\n" + "="*50 + "\n\n")


# # Function to predict deaths for a given disaster type using linear regression
# def predict_deaths_for_disaster_linear(disaster_type, file):
#     df_disaster = df[df['Disaster Type'] == disaster_type]

#     # Prepare the data: Select Year and Total Deaths for the regression model
#     X = df_disaster[['Year']]  # Feature: Year
#     y = df_disaster['Total Deaths']  # Target: Total Deaths

#     # Remove outliers using Z-score
#     df_disaster['z_score'] = zscore(df_disaster['Total Deaths'])
#     df_disaster_cleaned = df_disaster[df_disaster['z_score'].abs() <= 3]

#     # Prepare cleaned data
#     X_cleaned = df_disaster_cleaned[['Year']]
#     y_cleaned = df_disaster_cleaned['Total Deaths']

#     # Fit the model
#     model = LinearRegression()
#     model.fit(X_cleaned, y_cleaned)

#     # Make predictions
#     df_disaster_cleaned['Predicted Deaths (Linear)'] = model.predict(X_cleaned)

#     # Plotting the data points and the linear line of best fit
#     plt.figure(figsize=(12, 6))
#     plt.scatter(df_disaster_cleaned['Year'], df_disaster_cleaned['Total Deaths'], color='blue', label='Actual Deaths')
#     plt.plot(df_disaster_cleaned['Year'], df_disaster_cleaned['Predicted Deaths (Linear)'], color='green', label='Linear Line of Best Fit')
#     plt.xlabel('Year')
#     plt.ylabel('Total Deaths')
#     plt.title(f'Total Deaths Prediction for {disaster_type} (Linear Regression)')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

#     # Predict deaths for future years (for example, the next 10 years)
#     future_years = np.arange(2024, 2034).reshape(-1, 1)  # Predicting for 10 years ahead
#     future_deaths = model.predict(future_years)

#     # Save the predicted deaths for the next 10 years to the file
#     file.write(f"Predictions for the next 10 years (Linear Regression) for {disaster_type}:\n")
#     for year, deaths in zip(future_years.flatten(), future_deaths):
#         file.write(f"Predicted deaths in {year}: {deaths:.0f}\n")
#     file.write("\n" + "="*50 + "\n\n")


# # List of disaster types to predict deaths for
# disaster_types = df['Disaster Type'].unique()

# # Open a file to save the results
# with open('predicted_deaths.txt', 'w') as file:
#     # Run the prediction for each disaster type
#     for disaster_type in disaster_types:
#         file.write(f"Predictions for {disaster_type} (Polynomial Regression):\n")
#         predict_deaths_for_disaster_poly(disaster_type, file)
#         file.write("\n" + "="*50 + "\n")
#         file.write(f"Predictions for {disaster_type} (Linear Regression):\n")
#         predict_deaths_for_disaster_linear(disaster_type, file)
#         file.write("\n" + "="*50 + "\n")

# Dictionary of correlation values between disaster deaths and CO2 emissions
correlations = {
    'All disasters': -0.19732149908717758,
    'Drought': -0.2838068987936502,
    'Earthquake': 0.07269216425625637,
    'Extreme weather': 0.024783784599927125,
    'Flood': -0.17309653929552477,
    'Volcanic activity': -0.18894575966589325,
    'Dry mass movement': -0.22695616334514554,
    'Wet mass movement': -0.010193056953991366,
    'Wildfire': -0.1007740644166613,
    'Extreme temperature': 0.26727035599108917,

}

# Filter disaster types with absolute correlation greater than 0.2 (either positive or negative)
selected_disasters = {disaster: corr for disaster, corr in correlations.items() }

# Function to predict deaths for a given disaster type using Global CO2 emissions
def predict_deaths_for_disaster_by_co2(disaster_type):
    df_disaster = df[df['Disaster Type'] == disaster_type]
# # import pandas as pd
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from sklearn.preprocessing import PolynomialFeatures
# # from sklearn.linear_model import LinearRegression
# # from scipy.stats import zscore

# # # Load the dataset
# # df = pd.read_csv('merged_disaster_co2_data.csv')
# # df.columns = df.columns.str.strip()  # Strip any leading/trailing spaces from column names
# # df = df.dropna()
# # df = df.drop(columns='Code')

# # # Remove 'All disasters excluding extreme temperature' and 'All disasters excluding earthquakes' from the 'Disaster Type' column
# # df = df[df['Disaster Type'] != 'All disasters excluding extreme temperature']
# # df = df[df['Disaster Type'] != 'All disasters excluding earthquakes']

# # # Remove outliers using IQR and mean
# # def remove_outliers(df):
# #     q1 = df['Total Deaths'].quantile(0.25)
# #     q3 = df['Total Deaths'].quantile(0.75)
# #     iqr = q3 - q1
# #     lower_bound = q1 - 1.5 * iqr
# #     upper_bound = q3 + 1.5 * iqr
# #     df_cleaned = df[(df['Total Deaths'] >= lower_bound) & (df['Total Deaths'] <= upper_bound)]
    
# #     # Plot the graph with outliers removed
# #     plt.boxplot(df_cleaned['Total Deaths'])
# #     plt.title('Box Plot of Total Deaths (Outliers Removed)')
# #     plt.xlabel('Total Deaths')
# #     plt.ylabel('Count')
# #     plt.show()
    
# #     return df_cleaned

# # # Box plot with outliers removed and outliers not removed
# # def boxplot(df):
# #     plt.boxplot(df['Total Deaths'])
# #     plt.title('Box Plot of Total Deaths')
# #     plt.xlabel('Total Deaths')
# #     plt.ylabel('Count')
# #     plt.show()

# # # Plot initial boxplot
# # boxplot(df)

# # # Remove outliers and update the DataFrame
# # df = remove_outliers(df)

# # # Function to predict deaths for a given disaster type using polynomial regression
# # def predict_deaths_for_disaster_poly(disaster_type):
# #     df_disaster = df[df['Disaster Type'] == disaster_type]

# #     # Prepare the data: Select Year and Total Deaths for the regression model
# #     X = df_disaster[['Year']]  # Feature: Year
# #     y = df_disaster['Total Deaths']  # Target: Total Deaths

# #     # Remove outliers using Z-score
# #     df_disaster['z_score'] = zscore(df_disaster['Total Deaths'])
# #     df_disaster_cleaned = df_disaster[df_disaster['z_score'].abs() <= 3]

# #     # Prepare cleaned data
# #     X_cleaned = df_disaster_cleaned[['Year']]
# #     y_cleaned = df_disaster_cleaned['Total Deaths']

# #     # Transform the feature to polynomial features (degree 3)
# #     poly = PolynomialFeatures(degree=3)
# #     X_poly = poly.fit_transform(X_cleaned)

# #     # Fit the model
# #     model = LinearRegression()
# #     model.fit(X_poly, y_cleaned)

# #     # Make predictions
# #     df_disaster_cleaned['Predicted Deaths'] = model.predict(X_poly)

# #     # Plotting the data points and the polynomial line of best fit
# #     plt.figure(figsize=(12, 6))
# #     plt.scatter(df_disaster_cleaned['Year'], df_disaster_cleaned['Total Deaths'], color='blue', label='Actual Deaths')
# #     plt.plot(df_disaster_cleaned['Year'], df_disaster_cleaned['Predicted Deaths'], color='red', label='Polynomial Line of Best Fit')
# #     plt.xlabel('Year')
# #     plt.ylabel('Total Deaths')
# #     plt.title(f'Total Deaths Prediction for {disaster_type} (Polynomial Regression)')
# #     plt.legend()
# #     plt.grid(True)
# #     plt.show()

# #     # Predict deaths for future years (for example, the next 10 years)
# #     future_years = np.arange(2024, 2034).reshape(-1, 1)  # Predicting for 10 years ahead
# #     future_years_poly = poly.transform(future_years)  # Transform future years using the same polynomial features
# #     future_deaths = model.predict(future_years_poly)

# #     # Print the predicted deaths for the next 10 years
# #     for year, deaths in zip(future_years.flatten(), future_deaths):
# #         print(f"Predicted deaths in {year}: {deaths:.0f}")

# # # Function to predict deaths for a given disaster type using linear regression
# # def predict_deaths_for_disaster_linear(disaster_type):
# #     df_disaster = df[df['Disaster Type'] == disaster_type]

# #     # Prepare the data: Select Year and Total Deaths for the regression model
# #     X = df_disaster[['Year']]  # Feature: Year
# #     y = df_disaster['Total Deaths']  # Target: Total Deaths

# #     # Remove outliers using Z-score
# #     df_disaster['z_score'] = zscore(df_disaster['Total Deaths'])
# #     df_disaster_cleaned = df_disaster[df_disaster['z_score'].abs() <= 3]

# #     # Prepare cleaned data
# #     X_cleaned = df_disaster_cleaned[['Year']]
# #     y_cleaned = df_disaster_cleaned['Total Deaths']

# #     # Fit the model
# #     model = LinearRegression()
# #     model.fit(X_cleaned, y_cleaned)

# #     # Make predictions
# #     df_disaster_cleaned['Predicted Deaths'] = model.predict(X_cleaned)

# #     # Plotting the data points and the linear line of best fit
# #     plt.figure(figsize=(12, 6))
# #     plt.scatter(df_disaster_cleaned['Year'], df_disaster_cleaned['Total Deaths'], color='blue', label='Actual Deaths')
# #     plt.plot(df_disaster_cleaned['Year'], df_disaster_cleaned['Predicted Deaths'], color='green', label='Linear Line of Best Fit')
# #     plt.xlabel('Year')
# #     plt.ylabel('Total Deaths')
# #     plt.title(f'Total Deaths Prediction for {disaster_type} (Linear Regression)')
# #     plt.legend()
# #     plt.grid(True)
# #     plt.show()

# #     # Predict deaths for future years (for example, the next 10 years)
# #     future_years = np.arange(2024, 2034).reshape(-1, 1)  # Predicting for 10 years ahead
# #     future_deaths = model.predict(future_years)

# #     # Print the predicted deaths for the next 10 years
# #     for year, deaths in zip(future_years.flatten(), future_deaths):
# #         print(f"Predicted deaths in {year}: {deaths:.0f}")

# # # List of disaster types to predict deaths for
# # disaster_types = df['Disaster Type'].unique()

# # # Run the prediction for each disaster type
# # for disaster_type in disaster_types:
# #     print(f"Predictions for {disaster_type} (Polynomial Regression):")
# #     predict_deaths_for_disaster_poly(disaster_type)
# #     print("\n" + "="*50 + "\n")
# #     print(f"Predictions for {disaster_type} (Linear Regression):")
# #     predict_deaths_for_disaster_linear(disaster_type)
# #     print("\n" + "="*50 + "\n")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.stats import zscore

# Load the dataset
df = pd.read_csv('merged_disaster_co2_data.csv')
df.columns = df.columns.str.strip()  # Strip any leading/trailing spaces from column names
df = df.dropna()
df = df.drop(columns='Code')

# # Remove 'All disasters excluding extreme temperature' and 'All disasters excluding earthquakes' from the 'Disaster Type' column
df = df[df['Disaster Type'] != 'All disasters excluding extreme temperature']
df = df[df['Disaster Type'] != 'All disasters excluding earthquakes']

# # Remove outliers using IQR and mean
# def remove_outliers(df):
#     q1 = df['Total Deaths'].quantile(0.25)
#     q3 = df['Total Deaths'].quantile(0.75)
#     iqr = q3 - q1
#     lower_bound = q1 - 1.5 * iqr
#     upper_bound = q3 + 1.5 * iqr
#     df_cleaned = df[(df['Total Deaths'] >= lower_bound) & (df['Total Deaths'] <= upper_bound)]
    
#     # Plot the graph with outliers removed
#     plt.boxplot(df_cleaned['Total Deaths'])
#     plt.title('Box Plot of Total Deaths (Outliers Removed)')
#     plt.xlabel('Total Deaths')
#     plt.ylabel('Count')
#     plt.show()
    
#     return df_cleaned

# # Box plot with outliers removed and outliers not removed
# def boxplot(df):
#     plt.boxplot(df['Total Deaths'])
#     plt.title('Box Plot of Total Deaths')
#     plt.xlabel('Total Deaths')
#     plt.ylabel('Count')
#     plt.show()

# # Plot initial boxplot
# boxplot(df)

# # Remove outliers and update the DataFrame
# df = remove_outliers(df)

# # Function to predict deaths for a given disaster type using polynomial regression
# def predict_deaths_for_disaster_poly(disaster_type, file):
#     df_disaster = df[df['Disaster Type'] == disaster_type]

#     # Prepare the data: Select Year and Total Deaths for the regression model
#     X = df_disaster[['Year']]  # Feature: Year
#     y = df_disaster['Total Deaths']  # Target: Total Deaths

#     # Remove outliers using Z-score
#     df_disaster['z_score'] = zscore(df_disaster['Total Deaths'])
#     df_disaster_cleaned = df_disaster[df_disaster['z_score'].abs() <= 3]

#     # Prepare cleaned data
#     X_cleaned = df_disaster_cleaned[['Year']]
#     y_cleaned = df_disaster_cleaned['Total Deaths']

#     # Transform the feature to polynomial features (degree 3)
#     poly = PolynomialFeatures(degree=3)
#     X_poly = poly.fit_transform(X_cleaned)

#     # Fit the model
#     model = LinearRegression()
#     model.fit(X_poly, y_cleaned)

#     # Make predictions
#     df_disaster_cleaned['Predicted Deaths (Poly)'] = model.predict(X_poly)

#     # Plotting the data points and the polynomial line of best fit
#     plt.figure(figsize=(12, 6))
#     plt.scatter(df_disaster_cleaned['Year'], df_disaster_cleaned['Total Deaths'], color='blue', label='Actual Deaths')
#     plt.plot(df_disaster_cleaned['Year'], df_disaster_cleaned['Predicted Deaths (Poly)'], color='red', label='Polynomial Line of Best Fit')
#     plt.xlabel('Year')
#     plt.ylabel('Total Deaths')
#     plt.title(f'Total Deaths Prediction for {disaster_type} (Polynomial Regression)')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

#     # Predict deaths for future years (for example, the next 10 years)
#     future_years = np.arange(2024, 2034).reshape(-1, 1)  # Predicting for 10 years ahead
#     future_years_poly = poly.transform(future_years)  # Transform future years using the same polynomial features
#     future_deaths = model.predict(future_years_poly)

#     # Save the predicted deaths for the next 10 years to the file
#     file.write(f"Predictions for the next 10 years (Polynomial Regression) for {disaster_type}:\n")
#     for year, deaths in zip(future_years.flatten(), future_deaths):
#         file.write(f"Predicted deaths in {year}: {deaths:.0f}\n")
#     file.write("\n" + "="*50 + "\n\n")


# # Function to predict deaths for a given disaster type using linear regression
# def predict_deaths_for_disaster_linear(disaster_type, file):
#     df_disaster = df[df['Disaster Type'] == disaster_type]

#     # Prepare the data: Select Year and Total Deaths for the regression model
#     X = df_disaster[['Year']]  # Feature: Year
#     y = df_disaster['Total Deaths']  # Target: Total Deaths

#     # Remove outliers using Z-score
#     df_disaster['z_score'] = zscore(df_disaster['Total Deaths'])
#     df_disaster_cleaned = df_disaster[df_disaster['z_score'].abs() <= 3]

#     # Prepare cleaned data
#     X_cleaned = df_disaster_cleaned[['Year']]
#     y_cleaned = df_disaster_cleaned['Total Deaths']

#     # Fit the model
#     model = LinearRegression()
#     model.fit(X_cleaned, y_cleaned)

#     # Make predictions
#     df_disaster_cleaned['Predicted Deaths (Linear)'] = model.predict(X_cleaned)

#     # Plotting the data points and the linear line of best fit
#     plt.figure(figsize=(12, 6))
#     plt.scatter(df_disaster_cleaned['Year'], df_disaster_cleaned['Total Deaths'], color='blue', label='Actual Deaths')
#     plt.plot(df_disaster_cleaned['Year'], df_disaster_cleaned['Predicted Deaths (Linear)'], color='green', label='Linear Line of Best Fit')
#     plt.xlabel('Year')
#     plt.ylabel('Total Deaths')
#     plt.title(f'Total Deaths Prediction for {disaster_type} (Linear Regression)')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

#     # Predict deaths for future years (for example, the next 10 years)
#     future_years = np.arange(2024, 2034).reshape(-1, 1)  # Predicting for 10 years ahead
#     future_deaths = model.predict(future_years)

#     # Save the predicted deaths for the next 10 years to the file
#     file.write(f"Predictions for the next 10 years (Linear Regression) for {disaster_type}:\n")
#     for year, deaths in zip(future_years.flatten(), future_deaths):
#         file.write(f"Predicted deaths in {year}: {deaths:.0f}\n")
#     file.write("\n" + "="*50 + "\n\n")


# # List of disaster types to predict deaths for
# disaster_types = df['Disaster Type'].unique()

# # Open a file to save the results
# with open('predicted_deaths.txt', 'w') as file:
#     # Run the prediction for each disaster type
#     for disaster_type in disaster_types:
#         file.write(f"Predictions for {disaster_type} (Polynomial Regression):\n")
#         predict_deaths_for_disaster_poly(disaster_type, file)
#         file.write("\n" + "="*50 + "\n")
#         file.write(f"Predictions for {disaster_type} (Linear Regression):\n")
#         predict_deaths_for_disaster_linear(disaster_type, file)
#         file.write("\n" + "="*50 + "\n")

# Dictionary of correlation values between disaster deaths and CO2 emissions



correlations = {
    'All disasters': -0.19732149908717758,
    'Drought': -0.2838068987936502,
    'Earthquake': 0.07269216425625637,
    'Extreme weather': 0.024783784599927125,
    'Flood': -0.17309653929552477,
    'Volcanic activity': -0.18894575966589325,
    'Dry mass movement': -0.22695616334514554,
    'Wet mass movement': -0.010193056953991366,
    'Wildfire': -0.1007740644166613,
    'Extreme temperature': 0.26727035599108917,

}

# Filter disaster types with absolute correlation greater than 0.2 (either positive or negative)
selected_disasters = {disaster: corr for disaster, corr in correlations.items() }

# Function to predict deaths for a given disaster type using Global CO2 emissions
def predict_deaths_for_disaster_by_co2(disaster_type):
    df_disaster = df[df['Disaster Type'] == disaster_type]

    # Prepare the data: Select Global CO2 Emissions and Total Deaths for the regression model
    X = df_disaster[['Global CO2 Emissions']]  # Feature: Global CO2 Emissions
    y = df_disaster['Total Deaths']  # Target: Total Deaths

    # Remove outliers using Z-score for Total Deaths and Global CO2 Emissions
    df_disaster['z_score_deaths'] = zscore(df_disaster['Total Deaths'])
    df_disaster['z_score_co2'] = zscore(df_disaster['Global CO2 Emissions'])
    df_disaster_cleaned = df_disaster[(df_disaster['z_score_deaths'].abs() <= 3) & (df_disaster['z_score_co2'].abs() <= 3)]

    # Prepare cleaned data
    X_cleaned = df_disaster_cleaned[['Global CO2 Emissions']]
    y_cleaned = df_disaster_cleaned['Total Deaths']

    # Transform the feature to polynomial features (degree 3)
    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(X_cleaned)

    # Fit the model
    model = LinearRegression()
    model.fit(X_poly, y_cleaned)

    # Make predictions
    df_disaster_cleaned['Predicted Deaths'] = model.predict(X_poly)

    # Plotting the data points and the polynomial line of best fit
    plt.figure(figsize=(12, 6))
    plt.scatter(df_disaster_cleaned['Global CO2 Emissions'], df_disaster_cleaned['Total Deaths'], color='blue', label='Actual Deaths')
    plt.plot(df_disaster_cleaned['Global CO2 Emissions'], df_disaster_cleaned['Predicted Deaths'], color='red', label='Polynomial Line of Best Fit')
    plt.xlabel('Global CO2 Emissions')
    plt.ylabel('Total Deaths')
    plt.title(f'Total Deaths Prediction for {disaster_type} (Polynomial Regression)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Run the prediction for the selected disaster types based on correlation
for disaster_type in selected_disasters.keys():
    print(f"Predictions for {disaster_type}:")
    predict_deaths_for_disaster_by_co2(disaster_type)
    print("\n" + "="*50 + "\n")

    # Prepare the data: Select Global CO2 Emissions and Total Deaths for the regression model
    X = df_disaster[['Global CO2 Emissions']]  # Feature: Global CO2 Emissions
    y = df_disaster['Total Deaths']  # Target: Total Deaths

    # Remove outliers using Z-score for Total Deaths and Global CO2 Emissions
    df_disaster['z_score_deaths'] = zscore(df_disaster['Total Deaths'])
    df_disaster['z_score_co2'] = zscore(df_disaster['Global CO2 Emissions'])
    df_disaster_cleaned = df_disaster[(df_disaster['z_score_deaths'].abs() <= 3) & (df_disaster['z_score_co2'].abs() <= 3)]

    # Prepare cleaned data
    X_cleaned = df_disaster_cleaned[['Global CO2 Emissions']]
    y_cleaned = df_disaster_cleaned['Total Deaths']

    # Transform the feature to polynomial features (degree 3)
    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(X_cleaned)

    # Fit the model
    model = LinearRegression()
    model.fit(X_poly, y_cleaned)

    # Make predictions
    df_disaster_cleaned['Predicted Deaths'] = model.predict(X_poly)

    # Plotting the data points and the polynomial line of best fit
    plt.figure(figsize=(12, 6))
    plt.scatter(df_disaster_cleaned['Global CO2 Emissions'], df_disaster_cleaned['Total Deaths'], color='blue', label='Actual Deaths')
    plt.plot(df_disaster_cleaned['Global CO2 Emissions'], df_disaster_cleaned['Predicted Deaths'], color='red', label='Polynomial Line of Best Fit')
    plt.xlabel('Global CO2 Emissions')
    plt.ylabel('Total Deaths')
    plt.title(f'Total Deaths Prediction for {disaster_type} (Polynomial Regression)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Run the prediction for the selected disaster types based on correlation
for disaster_type in selected_disasters.keys():
    print(f"Predictions for {disaster_type}:")
    predict_deaths_for_disaster_by_co2(disaster_type)
    print("\n" + "="*50 + "\n")
