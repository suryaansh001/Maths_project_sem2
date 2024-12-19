import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.stats import zscore


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import zscore

# Load the dataset
file_path = 'merged_disaster_co2_data.csv'
data = pd.read_csv(file_path)

# Check for any missing values in the relevant columns
data = data.dropna(subset=['Disaster Type', 'Total Deaths', 'Year'])

# Display statistical summary
print(data.describe())

# Define correlations for disaster types
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
selected_disasters = {disaster: corr for disaster, corr in correlations.items() if abs(corr) > 0.2}

# Function to predict deaths for a given disaster type using Global CO2 emissions
def predict_deaths_for_disaster_by_co2(disaster_type):
    df_disaster = data[data['Disaster Type'] == disaster_type]

    # Prepare the data: Select Global CO2 Emissions and Total Deaths for the regression model
    X = df_disaster[['Global CO2 Emissions']]  # Feature: Global CO2 Emissions
    y = df_disaster['Total Deaths']  # Target: Total Deaths

    # Remove outliers using Z-score for Total Deaths and Global CO2 Emissions
    df_disaster['z_score_deaths'] = zscore(df_disaster['Total Deaths'])
    df_disaster['z_score_co2'] = zscore(df_disaster['Global CO2 Emissions'])
    
    # Clean the data by removing outliers
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
    
# Plotting yearly totals for each disaster type
disaster_types = data['Disaster Type'].unique()

for disaster in disaster_types:
    # Filter data for the specific disaster type
    disaster_data = data[data['Disaster Type'] == disaster]
    
    # Group by year and sum total deaths
    yearly_totals = disaster_data.groupby('Year')['Total Deaths'].sum().reset_index()

    # Create the line plot
    plt.figure(figsize=(10, 6))
    plt.plot(yearly_totals['Year'], yearly_totals['Total Deaths'], marker='o', label=disaster)
    
    plt.title(f'Year-Wise Deaths Due to {disaster}')
    plt.xlabel('Year')
    plt.ylabel('Total Deaths')
    plt.grid(True)
    
    # Show the plot for this disaster type
    plt.legend()
    plt.tight_layout()
    plt.show()
########################################
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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import zscore

corrs = {
    'All disasters': -0.197, 'Drought': -0.284, 'Earthquake': 0.073, 'Extreme weather': 0.025,
    'Flood': -0.173, 'Volcanic activity': -0.189, 'Dry mass movement': -0.227,
    'Wet mass movement': -0.010, 'Wildfire': -0.101, 'Extreme temperature': 0.267
}

sel_disasters = {d: c for d, c in corrs.items()}

def remove_outliers(df):
    df['z_deaths'] = zscore(df['Total Deaths'])
    df['z_co2'] = zscore(df['Global CO2 Emissions'])
    return df[(df['z_deaths'].abs() <= 3) & (df['z_co2'].abs() <= 3)]

def pred_deaths(d_type):
    df_d = df[df['Disaster Type'] == d_type]
    df_d = remove_outliers(df_d)
    
    X = df_d[['Global CO2 Emissions']]
    y = df_d['Total Deaths']
    
    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_poly, y)
    
    df_d['Predicted Deaths'] = model.predict(X_poly)
    
    plt.figure(figsize=(12, 6))
    plt.scatter(df_d['Global CO2 Emissions'], df_d['Total Deaths'], color='blue')
    plt.plot(df_d['Global CO2 Emissions'], df_d['Predicted Deaths'], color='red')
    plt.xlabel('Global CO2 Emissions')
    plt.ylabel('Total Deaths')
    plt.title(f'Deaths Prediction for {d_type}')
    plt.legend(['Regression Line', 'Actual Deaths'])
    plt.grid(True)
    plt.show()

for d in sel_disasters.keys():
    print(f"Predictions for {d}:")
    pred_deaths(d)
    print("\n" + "="*50 + "\n")

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



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import zscore

df = pd.read_csv('merged_disaster_co2_data.csv')
df.columns = df.columns.str.strip()
df = df.dropna()
df = df.drop(columns='Code')
print(df.columns)

df = df[df['Disaster Type'] != 'All disasters excluding extreme temperature']
df = df[df['Disaster Type'] != 'All disasters excluding earthquakes']

def remove_outliers(df):
    df['z_deaths'] = zscore(df['Total Deaths'])
    df['z_co2'] = zscore(df['Global CO2 Emissions'])
    df_cleaned = df[(df['z_deaths'].abs() <= 3) & (df['z_co2'].abs() <= 3)]
    return df_cleaned

def predict_deaths(disaster_type):
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
    plt.title(f'Total Deaths Prediction for {disaster_type}')
    plt.legend(['Linear Regression Line', 'Actual Deaths'])
    plt.grid(True)
    plt.show()

    future_co2 = np.linspace(df['Global CO2 Emissions'].min(), df['Global CO2 Emissions'].max(), 10).reshape(-1, 1)
    future_deaths = model.predict(future_co2)
    
    for co2, deaths in zip(future_co2.flatten(), future_deaths):
        print(f"Predicted deaths for {co2:.2f} CO2: {deaths:.0f}")

for disaster_type in df['Disaster Type'].unique():
    print(f"Predictions for {disaster_type}:")
    predict_deaths(disaster_type)
    print("\n" + "="*50 + "\n")


import pandas as pd
from scipy.stats import f_oneway

# Load data
df = pd.read_csv('merged_disaster_co2_data.csv')

# Group CO2 emissions by disaster type
eq_co2 = df[df['Disaster Type'] == 'Earthquake']['Global CO2 Emissions']
dr_co2 = df[df['Disaster Type'] == 'Drought']['Global CO2 Emissions']
ad_co2 = df[df['Disaster Type'] == 'All disasters']['Global CO2 Emissions']
aex_eq_co2 = df[df['Disaster Type'] == 'All disasters excluding earthquakes']['Global CO2 Emissions']
aex_et_co2 = df[df['Disaster Type'] == 'All disasters excluding extreme temperature']['Global CO2 Emissions']

# ANOVA
f_stat, p_val = f_oneway(eq_co2, dr_co2, ad_co2, aex_eq_co2, aex_et_co2)

# Result
print(f"F-statistic: {f_stat}\nP-value: {p_val}")
if p_val < 0.05:
    print("Significant difference in CO2 emissions between disaster types.")
else:
    print("No significant difference in CO2 emissions between disaster types.")

