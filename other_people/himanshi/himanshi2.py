import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.weightstats import ztest  # Correct import from statsmodels
from scipy.stats import chi2_contingency

# Load dataset
data = pd.read_csv('himanshi.csv')

# Display first few rows and summary of the dataset for analysis
data.head(), data.info()

# Calculate total deaths for each country and identify the top 5 countries
top_countries = data.groupby("Entity")["Deaths"].sum().nlargest(5).index
top_countries_data = data[data["Entity"].isin(top_countries)]

# Calculate mean, median, and mode for the deaths in these top 5 countries
summary_stats = top_countries_data.groupby("Entity")["Deaths"].agg(["mean", "median"])
summary_stats["mode"] = top_countries_data.groupby("Entity")["Deaths"].agg(
    lambda x: x.mode().iloc[0] if not x.mode().empty else None
)

# Display the top 5 countries and their statistics
top_countries, summary_stats

# Set up the plot for boxplot
plt.figure(figsize=(12, 6))

# Plot deaths for the top 5 countries
sns.boxplot(data=top_countries_data, x="Entity", y="Deaths", palette="Set3")
plt.title("Distribution of Deaths in Top 5 Countries", fontsize=16)
plt.xlabel("Country", fontsize=12)
plt.ylabel("Deaths", fontsize=12)

# Show the plot
plt.tight_layout()
plt.show()

# Perform z-tests to compare the mean deaths of Solomon Islands with the other top 4 countries
solomon_deaths = top_countries_data[top_countries_data["Entity"] == "Solomon Islands"]["Deaths"]

z_test_results = {}
for country in top_countries:
    if country != "Solomon Islands":
        country_deaths = top_countries_data[top_countries_data["Entity"] == country]["Deaths"]
        z_stat, p_value = ztest(solomon_deaths, country_deaths)
        z_test_results[country] = {"z_stat": z_stat, "p_value": p_value}

# Print z-test results
for country, results in z_test_results.items():
    print(f"Comparison between Solomon Islands and {country}:")
    print(f"Z-Statistic: {results['z_stat']:.2f}, P-Value: {results['p_value']:.4f}\n")

# Plot moving averages for deaths over years for each country
plt.figure(figsize=(12, 6))
for country in top_countries:
    country_data = top_countries_data[top_countries_data["Entity"] == country]
    country_data = country_data.sort_values("Year")
    rolling_mean = country_data["Deaths"].rolling(window=3).mean()
    
    plt.plot(country_data["Year"], rolling_mean, label=country)

plt.title("3-Year Moving Average of Deaths by Country")
plt.xlabel("Year")
plt.ylabel("Deaths")
plt.legend()
plt.show()

# # Categorize deaths into bins
# top_countries_data["Category"] = pd.qcut(top_countries_data["Deaths"], q=3, labels=["Low", "Medium", "High"])

# # Create a contingency table
# contingency_table = pd.crosstab(top_countries_data["Entity"], top_countries_data["Category"])

# # Perform chi-square test
# chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)
# print(f"Chi-Square Statistic: {chi2_stat:.2f}, P-Value: {p_value:.4f}")

# # Interpretation:
# # - p-value < 0.05: Association exists between country and death category.
# # - p-value ≥ 0.05: No significant association.
from scipy import stats

# Perform one-way ANOVA
f_stat, p_value = stats.f_oneway(*[group["Deaths"].values for name, group in top_countries_data.groupby("Entity")])

# The Z-test results support the conclusion that Solomon Islands has statistically significantly different death counts compared to each of the countries (Afghanistan, Central African Republic, Somalia, and Guinea-Bissau). Since all p-values are below the typical threshold of 0.05, we can reject the null hypothesis for each comparison.

print(f"ANOVA F-Statistic: {f_stat:.2f}, P-Value: {p_value:.4f}")