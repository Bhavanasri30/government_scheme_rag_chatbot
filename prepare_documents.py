import pandas as pd

# Load cleaned dataset
df = pd.read_csv("schemes_cleaned.csv")

documents = []

for _, row in df.iterrows():
    text = f"""
Scheme Name: {row['scheme_name']}

Details:
{row['details']}

Benefits:
{row['benefits']}

Eligibility:
{row['eligibility']}

Application Process:
{row['application']}

Documents Required:
{row['documents']}

Level:
{row['level']}

Category:
{row['schemeCategory']}

Tags:
{row['tags']}
"""

    documents.append(text.strip())

# Save documents
with open("scheme_documents.txt", "w", encoding="utf-8") as f:
    for i, document in enumerate(documents):
        f.write(f"\n--- SCHEME {i + 1} ---\n")
        f.write(document)
        f.write("\n")

print("Documents created successfully!")
print("Total documents:", len(documents))