# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression
# from scipy.stats import zscore

# # Load the dataset

# df = pd.read_csv('merged_disaster_co2_data.csv')
# df.columns = df.columns.str.strip()  # Strip any leading/trailing spaces from column names
# df = df.dropna()
# df = df.drop(columns='Code')
# print(df.columns)

# # Remove 'All disasters excluding extreme temperature' and 'All disasters excluding earthquakes' from the 'Disaster Type' column
# df = df[df['Disaster Type'] != 'All disasters excluding extreme temperature']
# df = df[df['Disaster Type'] != 'All disasters excluding earthquakes']

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

# # # Function to predict deaths for a given disaster type
# # def predict_deaths_for_disaster(disaster_type):
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

# # # List of disaster types to predict deaths for
# # disaster_types = df['Disaster Type'].unique()

# # # Run the prediction for each disaster type
# # for disaster_type in disaster_types:
# #     print(f"Predictions for {disaster_type}:")
# #     predict_deaths_for_disaster(disaster_type)
# #     print("\n" + "="*50 + "\n")
# # Function to predict deaths for a given disaster type using CO2 emissions
# # Function to predict deaths for a given disaster type using Global CO2 emissions
# def predict_deaths_for_disaster_by_co2(disaster_type):
#     df_disaster = df[df['Disaster Type'] == disaster_type]

#     # Prepare the data: Select Global CO2 Emissions and Total Deaths for the regression model
#     X = df_disaster[['Global CO2 Emissions']]  # Feature: Global CO2 Emissions
#     y = df_disaster['Total Deaths']  # Target: Total Deaths

#     # Remove outliers using Z-score for Total Deaths and Global CO2 Emissions
#     df_disaster['z_score_deaths'] = zscore(df_disaster['Total Deaths'])
#     df_disaster['z_score_co2'] = zscore(df_disaster['Global CO2 Emissions'])
#     df_disaster_cleaned = df_disaster[(df_disaster['z_score_deaths'].abs() <= 3) & (df_disaster['z_score_co2'].abs() <= 3)]

#     # Prepare cleaned data
#     X_cleaned = df_disaster_cleaned[['Global CO2 Emissions']]
#     y_cleaned = df_disaster_cleaned['Total Deaths']

#     # Transform the feature to polynomial features (degree 3)
#     poly = PolynomialFeatures(degree=3)
#     X_poly = poly.fit_transform(X_cleaned)

#     # Fit the model
#     model = LinearRegression()
#     model.fit(X_poly, y_cleaned)

#     # Make predictions
#     df_disaster_cleaned['Predicted Deaths'] = model.predict(X_poly)

#     # Plotting the data points and the polynomial line of best fit
#     plt.figure(figsize=(12, 6))
#     plt.scatter(df_disaster_cleaned['Global CO2 Emissions'], df_disaster_cleaned['Total Deaths'], color='blue', label='Actual Deaths')
#     plt.plot(df_disaster_cleaned['Global CO2 Emissions'], df_disaster_cleaned['Predicted Deaths'], color='red', label='Polynomial Line of Best Fit')
#     plt.xlabel('Global CO2 Emissions')
#     plt.ylabel('Total Deaths')
#     plt.title(f'Total Deaths Prediction for {disaster_type} (Polynomial Regression)')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

#     # Predict deaths for future CO2 emissions (for example, a range of future emission values)
#     future_co2 = np.linspace(df['Global CO2 Emissions'].min(), df['Global CO2 Emissions'].max(), 10).reshape(-1, 1)  # Predicting for future emissions
#     future_co2_poly = poly.transform(future_co2)  # Transform future CO2 emissions using the same polynomial features
#     future_deaths = model.predict(future_co2_poly)

#     # Print the predicted deaths for the future CO2 emissions
#     for co2, deaths in zip(future_co2.flatten(), future_deaths):
#         print(f"Predicted deaths for Global CO2 emissions of {co2:.2f}: {deaths:.0f}")

# # List of disaster types to predict deaths for based on Global CO2 emissions
# disaster_types = df['Disaster Type'].unique()

# # Run the prediction for each disaster type
# for disaster_type in disaster_types:
#     print(f"Predictions for {disaster_type}:")
#     predict_deaths_for_disaster_by_co2(disaster_type)
#     print("\n" + "="*50 + "\n")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import zscore

# Load the dataset
df = pd.read_csv('merged_disaster_co2_data.csv')
df.columns = df.columns.str.strip()  # Strip any leading/trailing spaces from column names
df = df.dropna()
df = df.drop(columns='Code')  # Drop 'Code' column if not needed
print(df.columns)

# Remove 'All disasters excluding extreme temperature' and 'All disasters excluding earthquakes' from the 'Disaster Type' column
df = df[df['Disaster Type'] != 'All disasters excluding extreme temperature']
df = df[df['Disaster Type'] != 'All disasters excluding earthquakes']

# Function to remove outliers based on Z-score for both Total Deaths and CO2 emissions
def remove_outliers(df):
    # Z-score based filtering for outliers in both 'Total Deaths' and 'Global CO2 Emissions'
    df['z_score_deaths'] = zscore(df['Total Deaths'])
    df['z_score_co2'] = zscore(df['Global CO2 Emissions'])
    
    # Keep only rows where both Total Deaths and CO2 emissions are within 3 standard deviations
    df_cleaned = df[(df['z_score_deaths'].abs() <= 3) & (df['z_score_co2'].abs() <= 3)]
    return df_cleaned

# Function to perform linear regression using Global CO2 emissions to predict Total Deaths
def predict_deaths_for_disaster_by_co2(disaster_type):
    df_disaster = df[df['Disaster Type'] == disaster_type]

    # Remove outliers using Z-score
    df_disaster_cleaned = remove_outliers(df_disaster)

    # Prepare the data: Select Global CO2 Emissions and Total Deaths for the regression model
    X = df_disaster_cleaned[['Global CO2 Emissions']]  # Feature: Global CO2 Emissions
    y = df_disaster_cleaned['Total Deaths']  # Target: Total Deaths

    # Fit the Linear Regression model
    model = LinearRegression()
    model.fit(X, y)

    # Make predictions
    df_disaster_cleaned['Predicted Deaths'] = model.predict(X)

    # Plotting the data points and the regression line
    plt.figure(figsize=(12, 6))
    plt.scatter(df_disaster_cleaned['Global CO2 Emissions'], df_disaster_cleaned['Total Deaths'], color='blue', label='Actual Deaths')
    plt.plot(df_disaster_cleaned['Global CO2 Emissions'], df_disaster_cleaned['Predicted Deaths'], color='red', label='Linear Regression Line')
    plt.xlabel('Global CO2 Emissions')
    plt.ylabel('Total Deaths')
    plt.title(f'Total Deaths Prediction for {disaster_type} (Linear Regression)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Predict deaths for a range of future CO2 emissions (for example, the next 10 years or a range of values)
    future_co2 = np.linspace(df['Global CO2 Emissions'].min(), df['Global CO2 Emissions'].max(), 10).reshape(-1, 1)
    future_deaths = model.predict(future_co2)

    # Print the predicted deaths for future CO2 emissions
    for co2, deaths in zip(future_co2.flatten(), future_deaths):
        print(f"Predicted deaths for Global CO2 emissions of {co2:.2f}: {deaths:.0f}")

# List of disaster types to predict deaths for based on Global CO2 emissions
disaster_types = df['Disaster Type'].unique()

# Run the prediction for each disaster type
for disaster_type in disaster_types:
    print(f"Predictions for {disaster_type}:")
    predict_deaths_for_disaster_by_co2(disaster_type)
    print("\n" + "="*50 + "\n")
