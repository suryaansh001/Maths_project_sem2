#merge 
import pandas as pd
file1_path = 'merged_disaster_co2_data.csv'
file2_path='final_temp.csv'
# df1 = pd.read_csv(file1_path)
# df2 = pd.read_csv(file2_path)
# Attempting to read the files in raw format to identify issues
with open(file1_path, 'r') as f1, open(file2_path, 'r') as f2:
    file1_raw = f1.readlines()
    file2_raw = f2.readlines()

# Displaying a few lines from each file to spot the problem
file1_raw[:5], file2_raw[:5]
# Cleaning the second file to remove metadata lines
cleaned_file2_path = 'cleaned_final_temp.csv'

with open(cleaned_file2_path, 'w') as cleaned_file2:
    for line in file2_raw:
        if not line.startswith("#"):
            cleaned_file2.write(line)

# Reattempt reading the cleaned second file
file2_cleaned = pd.read_csv(cleaned_file2_path)

# Display the cleaned data to confirm the fix
file2_cleaned.head()
# Ensure the first file is properly loaded as well
file1 = pd.read_csv(file1_path)

# Merging the two datasets year-wise
merged_data = pd.merge(file1, file2_cleaned, on="Year", how="inner")

# Save the merged data to a new CSV file
merged_file_path = 'merged_yearwise_data.csv'
merged_data.to_csv(merged_file_path, index=False)

# Display the first few rows of the merged data to confirm the operation
merged_data.head(), merged_file_path
