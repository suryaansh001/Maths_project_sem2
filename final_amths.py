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

# Remove 'All disasters excluding extreme temperature' and 'All disasters excluding earthquakes' from the 'Disaster Type' column
df = df[df['Disaster Type'] != 'All disasters excluding extreme temperature']
df = df[df['Disaster Type'] != 'All disasters excluding earthquakes']

# Function to predict deaths for a given disaster type using polynomial regression
def predict_deaths_for_disaster_poly(disaster_type):
    df_disaster = df[df['Disaster Type'] == disaster_type]

    # Prepare the data: Select Year and Total Deaths for the regression model
    X = df_disaster[['Year']]  # Feature: Year
    y = df_disaster['Total Deaths']  # Target: Total Deaths

    # Remove outliers using Z-score
    df_disaster['z_score'] = zscore(df_disaster['Total Deaths'])
    df_disaster_cleaned = df_disaster[df_disaster['z_score'].abs() <= 3]

    # Prepare cleaned data
    X_cleaned = df_disaster_cleaned[['Year']]
    y_cleaned = df_disaster_cleaned['Total Deaths']

    # Transform the feature to polynomial features (degree 3)
    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(X_cleaned)

    # Fit the model
    model = LinearRegression()
    model.fit(X_poly, y_cleaned)

    # Make predictions
    df_disaster_cleaned['Predicted Deaths (Poly)'] = model.predict(X_poly)

    # Save the results for polynomial regression in a CSV file
    results_poly = df_disaster_cleaned[['Disaster Type', 'Year', 'Total Deaths', 'Predicted Deaths (Poly)']]
    results_poly.to_csv('results_polynomial.csv', mode='a', header=not pd.io.common.file_exists('results_polynomial.csv'), index=False)

    # Plotting the data points and the polynomial line of best fit
    plt.figure(figsize=(12, 6))
    plt.scatter(df_disaster_cleaned['Year'], df_disaster_cleaned['Total Deaths'], color='blue', label='Actual Deaths')
    plt.plot(df_disaster_cleaned['Year'], df_disaster_cleaned['Predicted Deaths (Poly)'], color='red', label='Polynomial Line of Best Fit')
    plt.xlabel('Year')
    plt.ylabel('Total Deaths')
    plt.title(f'Total Deaths Prediction for {disaster_type} (Polynomial Regression)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Function to predict deaths for a given disaster type using linear regression
def predict_deaths_for_disaster_linear(disaster_type):
    df_disaster = df[df['Disaster Type'] == disaster_type]

    # Prepare the data: Select Year and Total Deaths for the regression model
    X = df_disaster[['Year']]  # Feature: Year
    y = df_disaster['Total Deaths']  # Target: Total Deaths

    # Remove outliers using Z-score
    df_disaster['z_score'] = zscore(df_disaster['Total Deaths'])
    df_disaster_cleaned = df_disaster[df_disaster['z_score'].abs() <= 3]

    # Prepare cleaned data
    X_cleaned = df_disaster_cleaned[['Year']]
    y_cleaned = df_disaster_cleaned['Total Deaths']

    # Fit the model
    model = LinearRegression()
    model.fit(X_cleaned, y_cleaned)

    # Make predictions
    df_disaster_cleaned['Predicted Deaths (Linear)'] = model.predict(X_cleaned)

    # Save the results for linear regression in a CSV file
    results_linear = df_disaster_cleaned[['Disaster Type', 'Year', 'Total Deaths', 'Predicted Deaths (Linear)']]
    results_linear.to_csv('results_linear.csv', mode='a', header=not pd.io.common.file_exists('results_linear.csv'), index=False)

    # Plotting the data points and the linear line of best fit
    plt.figure(figsize=(12, 6))
    plt.scatter(df_disaster_cleaned['Year'], df_disaster_cleaned['Total Deaths'], color='blue', label='Actual Deaths')
    plt.plot(df_disaster_cleaned['Year'], df_disaster_cleaned['Predicted Deaths (Linear)'], color='green', label='Linear Line of Best Fit')
    plt.xlabel('Year')
    plt.ylabel('Total Deaths')
    plt.title(f'Total Deaths Prediction for {disaster_type} (Linear Regression)')
    plt.legend()
    plt.grid(True)
    plt.show()

# List of disaster types to predict deaths for
disaster_types = df['Disaster Type'].unique()

# Run the prediction for each disaster type
for disaster_type in disaster_types:
    print(f"Predictions for {disaster_type} (Polynomial Regression):")
    predict_deaths_for_disaster_poly(disaster_type)
    print("\n" + "="*50 + "\n")
    print(f"Predictions for {disaster_type} (Linear Regression):")
    predict_deaths_for_disaster_linear(disaster_type)
    print("\n" + "="*50 + "\n")
