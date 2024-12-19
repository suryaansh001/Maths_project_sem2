# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.metrics import mean_absolute_error, mean_squared_error

# # Load the dataset
# df = pd.read_csv('merged_disaster_co2_data.csv')
# df.columns = df.columns.str.strip()  # Clean column names
# df = df.dropna()
# df = df[df['Disaster Type'].notnull()]  # Ensure 'Disaster Type' column is not empty

# # Get unique disaster types
# disaster_types = df['Disaster Type'].unique()

# # Function to perform residual analysis
# def residual_analysis(df_disaster, disaster_type):
#     # Prepare the data
#     X = df_disaster[['Global CO2 Emissions']]
#     y = df_disaster['Total Deaths']

#     # Linear Regression
#     linear_model = LinearRegression()
#     linear_model.fit(X, y)
#     y_pred_linear = linear_model.predict(X)
#     residuals_linear = y - y_pred_linear

#     # Polynomial Regression (degree 3)
#     poly = PolynomialFeatures(degree=3)
#     X_poly = poly.fit_transform(X)
#     poly_model = LinearRegression()
#     poly_model.fit(X_poly, y)
#     y_pred_poly = poly_model.predict(X_poly)
#     residuals_poly = y - y_pred_poly

#     # Plot residuals for both models
#     plt.figure(figsize=(14, 6))

#     plt.subplot(1, 2, 1)
#     plt.scatter(y_pred_linear, residuals_linear, alpha=0.7, label="Linear")
#     plt.axhline(y=0, color='r', linestyle='--')
#     plt.title(f"Residuals vs Fitted (Linear) - {disaster_type}")
#     plt.xlabel("Fitted values")
#     plt.ylabel("Residuals")
#     plt.legend()
#     plt.grid()

#     plt.subplot(1, 2, 2)
#     plt.scatter(y_pred_poly, residuals_poly, alpha=0.7, label="Polynomial (Degree 3)", color='orange')
#     plt.axhline(y=0, color='r', linestyle='--')
#     plt.title(f"Residuals vs Fitted (Polynomial) - {disaster_type}")
#     plt.xlabel("Fitted values")
#     plt.ylabel("Residuals")
#     plt.legend()
#     plt.grid()

#     plt.tight_layout()
#     plt.show()

#     # Calculate and display performance metrics
#     mae_linear = mean_absolute_error(y, y_pred_linear)
#     rmse_linear = np.sqrt(mean_squared_error(y, y_pred_linear))

#     mae_poly = mean_absolute_error(y, y_pred_poly)
#     rmse_poly = np.sqrt(mean_squared_error(y, y_pred_poly))

#     print(f"Performance Metrics for {disaster_type}:\n")
#     print(f"Linear Regression: MAE = {mae_linear:.2f}, RMSE = {rmse_linear:.2f}")
#     print(f"Polynomial Regression: MAE = {mae_poly:.2f}, RMSE = {rmse_poly:.2f}")
#     print("=" * 50)

# # Run residual analysis for each disaster type
# for disaster_type in disaster_types:
#     print(f"Residual Analysis for {disaster_type}:")
#     df_disaster = df[df['Disaster Type'] == disaster_type]
#     residual_analysis(df_disaster, disaster_type)
#     print("\n")
import pandas as pd
df=pd.read_csv('merged_disaster_co2_data.csv')
print(df.head(10))
#remoce the row of 'All disasters excluding earthquakes'
df.drop(df[df['Disaster Type']=='All disasters excluding earthquakes'].index, inplace=True)

df.drop(df[df['Disaster Type']=='All disasters excluding extreme temperature'].index, inplace=True)
print(df.head(10))