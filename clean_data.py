import pandas as pd

# Load dataset
df = pd.read_csv("updated_data.csv")

# Remove completely empty column
df = df.drop(columns=["Unnamed: 9"])

# Fill missing values
df["application"] = df["application"].fillna("Information not available")
df["documents"] = df["documents"].fillna("Information not available")
df["tags"] = df["tags"].fillna("")

# Save cleaned dataset
df.to_csv("schemes_cleaned.csv", index=False)

print("Cleaning completed!")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nMissing values:")
print(df.isnull().sum())