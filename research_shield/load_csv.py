import pandas as pd

# Load the CSV file
file_path = r"C:\Users\HIMANSHU\Downloads\people-100.csv"

# Read the file
df = pd.read_csv(file_path)

# Display first 5 rows
print(df.head())
