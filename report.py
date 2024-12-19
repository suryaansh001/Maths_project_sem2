import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import zscore, f_oneway

# Load data
file_path = 'merged_disaster_co2_data.csv'
df = pd.read_csv(file_path)

# Clean up the data
df.columns = df.columns.str.strip()
df = df.dropna()  # Remove rows with NaN values
df = df.drop(columns='Code')  # Drop 'Code' column
df = df[df['Disaster Type'] != 'All disasters excluding extreme temperature']
df = df[df['Disaster Type'] != 'All disasters excluding earthquakes']

# Print the summary statistics of the DataFrame
print(df.describe())

# Remove outliers based on z-score
def remove_outliers(df):
    df['z_deaths'] = zscore(df['Total Deaths'])
    df['z_co2'] = zscore(df['Global CO2 Emissions'])
    return df[(df['z_deaths'].abs() <= 3) & (df['z_co2'].abs() <= 3)]

# Perform Linear Regression and predict deaths
def predict_deaths_linear(disaster_type):
    df_disaster = df[df['Disaster Type'] == disaster_type]
    df_disaster_cleaned = remove_outliers(df_disaster)
    
    X = df_disaster_cleaned[['Global CO2 Emissions']]
    y = df_disaster_cleaned['Total Deaths']
    
    model = LinearRegression()
    model.fit(X, y)
    
    df_disaster_cleaned['Predicted Deaths'] = model.predict(X)
    
    plt.figure(figsize=(12, 6))
    plt.scatter(df_disaster_cleaned['Global CO2 Emissions'], df_disaster_cleaned['Total Deaths'], color='blue')
    plt.plot(df_disaster_cleaned['Global CO2 Emissions'], df_disaster_cleaned['Predicted Deaths'], color='red')
    plt.xlabel('Global CO2 Emissions')
    plt.ylabel('Total Deaths')
    plt.title(f'Total Deaths Prediction for {disaster_type} (Linear Regression)')
    plt.legend(['Regression Line', 'Actual Deaths'])
    plt.grid(True)
    plt.show()

    future_co2 = np.linspace(df['Global CO2 Emissions'].min(), df['Global CO2 Emissions'].max(), 10).reshape(-1, 1)
    future_deaths = model.predict(future_co2)
    
    for co2, deaths in zip(future_co2.flatten(), future_deaths):
        print(f"Predicted deaths for {co2:.2f} CO2: {deaths:.0f}")

# Perform Polynomial Regression and predict deaths
def predict_deaths_poly(disaster_type):
    df_disaster = df[df['Disaster Type'] == disaster_type]
    df_disaster_cleaned = remove_outliers(df_disaster)
    
    X = df_disaster_cleaned[['Global CO2 Emissions']]
    y = df_disaster_cleaned['Total Deaths']
    
    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_poly, y)
    
    df_disaster_cleaned['Predicted Deaths'] = model.predict(X_poly)
    
    plt.figure(figsize=(12, 6))
    plt.scatter(df_disaster_cleaned['Global CO2 Emissions'], df_disaster_cleaned['Total Deaths'], color='blue')
    plt.plot(df_disaster_cleaned['Global CO2 Emissions'], df_disaster_cleaned['Predicted Deaths'], color='red')
    plt.xlabel('Global CO2 Emissions')
    plt.ylabel('Total Deaths')
    plt.title(f'Total Deaths Prediction for {disaster_type} (Polynomial Regression)')
    plt.legend(['Regression Line', 'Actual Deaths'])
    plt.grid(True)
    plt.show()

# Perform ANOVA to test for CO2 emission differences across disaster types
def perform_anova():
    eq_co2 = df[df['Disaster Type'] == 'Earthquake']['Global CO2 Emissions']
    dr_co2 = df[df['Disaster Type'] == 'Drought']['Global CO2 Emissions']
    ad_co2 = df[df['Disaster Type'] == 'All disasters']['Global CO2 Emissions']
    aex_eq_co2 = df[df['Disaster Type'] == 'All disasters excluding earthquakes']['Global CO2 Emissions']
    aex_et_co2 = df[df['Disaster Type'] == 'All disasters excluding extreme temperature']['Global CO2 Emissions']

    f_stat, p_val = f_oneway(eq_co2, dr_co2, ad_co2, aex_eq_co2, aex_et_co2)
    
    print(f"F-statistic: {f_stat}\nP-value: {p_val}")
    if p_val < 0.05:
        print("Significant difference in CO2 emissions between disaster types.")
    else:
        print("No significant difference in CO2 emissions between disaster types.")

# Plot yearly total deaths for each disaster type
def plot_yearly_deaths():
    df_clean = df.dropna(subset=['Disaster Type', 'Total Deaths', 'Year'])

    disaster_types = df_clean['Disaster Type'].unique()

    for disaster in disaster_types:
        disaster_data = df_clean[df_clean['Disaster Type'] == disaster]
        yearly_totals = disaster_data.groupby('Year')['Total Deaths'].sum().reset_index()

        plt.figure(figsize=(10, 6))
        plt.plot(yearly_totals['Year'], yearly_totals['Total Deaths'], marker='o', label=disaster)
        plt.title(f'Year-Wise Deaths Due to {disaster}')
        plt.xlabel('Year')
        plt.ylabel('Total Deaths')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# Plot yearly deaths first
plot_yearly_deaths()

# Execute functions for each disaster type with regression models
for disaster_type in df['Disaster Type'].unique():
    print(f"Predictions for {disaster_type} (Linear Regression):")
    predict_deaths_linear(disaster_type)
    print("\n" + "="*50 + "\n")
    print(f"Predictions for {disaster_type} (Polynomial Regression):")
    predict_deaths_poly(disaster_type)
    print("\n" + "="*50 + "\n")

# Perform ANOVA
perform_anova()
