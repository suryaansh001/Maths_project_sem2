import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Reshape the data
def reshape_data(df):
    """
    Reshape the dataset from wide to long format.
    Args:
        df (DataFrame): Original wide-format DataFrame.
    Returns:
        DataFrame: Reshaped long-format DataFrame with columns ['Entity', 'Year', 'Disease', 'Deaths'].
    """
    # Identify disease columns (excluding 'Entity', 'Code', and 'Year')
    id_vars = ['Entity', 'Code', 'Year']
    value_vars = [col for col in df.columns if col not in id_vars]

    # Melt the data into long format
    reshaped = pd.melt(
        df,
        id_vars=id_vars,
        value_vars=value_vars,
        var_name='Disease',
        value_name='Deaths'
    )

    # Clean disease names (remove trailing \n or other whitespace)
    reshaped['Disease'] = reshaped['Disease'].str.strip()
    reshaped['Deaths'] = pd.to_numeric(reshaped['Deaths'], errors='coerce').fillna(0)
    return reshaped

# Step 2: Get top countries by total deaths
def get_top_countries_by_death(df, top_n=3):
    """
    Get the top `n` countries with the highest total deaths.
    Args:
        df (DataFrame): Long-format DataFrame with ['Entity', 'Year', 'Disease', 'Deaths'].
        top_n (int): Number of top countries to retrieve.
    Returns:
        list: List of top `n` countries with highest deaths.
    """
    country_totals = df.groupby('Entity')['Deaths'].sum().sort_values(ascending=False)
    return country_totals.head(top_n).index.tolist()

# Step 3: Plot year vs. deaths for each disease for a given country
def plot_year_vs_disease(df, country):
    """
    Plot year vs. deaths for each disease in a specific country.
    Args:
        df (DataFrame): Long-format DataFrame with ['Entity', 'Year', 'Disease', 'Deaths'].
        country (str): Country to filter the data for.
    """
    country_data = df[df['Entity'] == country]
    pivot_table = country_data.pivot_table(index='Year', columns='Disease', values='Deaths', aggfunc='sum', fill_value=0)
    pivot_table.plot(kind='bar', figsize=(12, 6))
    plt.title(f'Year vs Deaths by Disease in {country}')
    plt.ylabel('Deaths')
    plt.xlabel('Year')
    plt.legend(title='Disease', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# Step 4: Plot year vs. total deaths across all diseases
def plot_year_vs_total_death(df):
    """
    Plot year vs. total deaths across all countries and diseases.
    Args:
        df (DataFrame): Long-format DataFrame with ['Entity', 'Year', 'Disease', 'Deaths'].
    """
    yearly_totals = df.groupby('Year')['Deaths'].sum()
    yearly_totals.plot(kind='bar', figsize=(12, 6), color='coral')
    plt.title('Year vs Total Deaths Across All Diseases')
    plt.ylabel('Total Deaths')
    plt.xlabel('Year')
    plt.tight_layout()
    plt.show()

# Example usage
# Load your dataset
# data = pd.read_csv('your_dataset.csv')  # Replace with actual file path
# reshaped_data = reshape_data(data)

# Get the top 3 countries
# top_countries = get_top_countries_by_death(reshaped_data, top_n=3)
# print(f'Top 3 countries with highest deaths: {top_countries}')

# Plot for each top country
# for country in top_countries:
#     plot_year_vs_disease(reshaped_data, country)

# Plot total deaths across all years
# plot_year_vs_total_death(reshaped_data)
