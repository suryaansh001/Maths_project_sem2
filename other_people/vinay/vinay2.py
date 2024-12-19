import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns

# Step 1: Load the data
df = pd.read_csv('final_death.csv')

# Clean the column names (remove extra spaces, newlines, etc.)
df.columns = df.columns.str.replace('\n', ' ').str.strip()

# Remove unnecessary columns
cols_to_drop = ['Unnamed: 35']
df.drop(cols_to_drop, axis=1, inplace=True)

# Remove medical disease columns based on your description


# Step 2: Clean the data (remove missing values)
df.dropna(inplace=True)

# Convert all relevant columns to numeric, forcing errors to NaN
df = df.apply(pd.to_numeric, errors='coerce')

# Convert 'Year' column to integers
df['Year'] = df['Year'].astype(int)

# Step 3: Calculate cumulative deaths for each country
df_cumulative = df.groupby('Entity').sum()

# Step 4: Function to find highest cumulative deaths
def highest_cumulative_deaths(df):
    df_cumulative = df.groupby('Entity').sum()
    
    # Sum across rows (axis=1) to get the total deaths for each entity
    df_cumulative['Total Deaths'] = df_cumulative.sum(axis=1)
    
    # Sort and return the top 5 entities with the highest total deaths
    top_countries = df_cumulative.nlargest(5, 'Total Deaths')
    return top_countries[['Total Deaths']]

# Example usage: Get the top 5 countries with highest cumulative deaths
top_countries = highest_cumulative_deaths(df)
print(top_countries)

# Step 5: Linear regression function to plot and predict future deaths
def plot_linear_regression(df, disease_column):
    # Ensure the disease column exists in the dataframe
    if disease_column not in df.columns:
        print(f"{disease_column} is not a valid column.")
        return
    
    df_disease = df[['Year', disease_column]].dropna()
    
    # Prepare data for linear regression
    X = df_disease[['Year']].values.reshape(-1, 1)
    y = df_disease[disease_column].values
    
    # Fit the linear regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Make future predictions (e.g., predict for the next 5 years)
    future_years = np.array(range(df['Year'].max() + 1, df['Year'].max() + 6)).reshape(-1, 1)
    future_predictions = model.predict(future_years)
    
    # Plot the data and the linear regression line
    plt.scatter(df_disease['Year'], df_disease[disease_column], color='blue', label='Actual Data')
    plt.plot(df_disease['Year'], model.predict(X), color='red', label='Line of Best Fit')
    plt.xlabel('Year')
    plt.ylabel(f'{disease_column} Fatalities')
    plt.title(f'Linear Regression for {disease_column}')
    plt.legend()
    plt.show()
    
    # Return future predictions (next 5 years)
    return future_years.flatten(), future_predictions

# Example usage: Predict future deaths for Meningitis fatalities
future_years, future_predictions = plot_linear_regression(df, 'Meningitis fatalities')
print("Predicted future deaths for Meningitis fatalities:")
for year, prediction in zip(future_years, future_predictions):
    print(f"Year: {year}, Predicted Deaths: {prediction}")

# Step 6: Function to calculate correlation between causes of death
def calculate_correlation(df):
    # Calculate the correlation matrix for all numerical columns
    correlation_matrix = df.corr()
    
    # Plot the heatmap of the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix of Death Causes')
    plt.show()
    
    return correlation_matrix
#function to plot bar grpah of each country for year and diseases
def plot_bar_graph(df, code):
    # Filter the dataframe for the specified country
    df_country = df[df['Code'] == code]
    
    # Check if the country exists in the dataframe
    if df_country.empty:
        print(f"{code} does not exist in the dataframe.")
        return
    
    # Plot a bar graph for each year and disease columns
    df_country.plot(x='Year', kind='bar', figsize=(12, 6), title=f'Disease Fatalities in {country} Over Time')
    plt.xlabel('Year')
    plt.ylabel('Fatalities')
    plt.show()
# Example usage: Plot bar graph of disease fatalities for a specific country



# # Example usage: Calculate and display correlation matrix
# correlation_matrix = calculate_correlation(df)
# print(correlation_matrix)
plot_bar_graph(df, 'AFG')