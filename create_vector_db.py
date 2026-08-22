import pandas as pd
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# Load cleaned dataset
df = pd.read_csv("schemes_cleaned.csv")

# Convert each scheme into searchable text
documents = []

for _, row in df.iterrows():
    text = f"""
Scheme Name: {row['scheme_name']}
Details: {row['details']}
Benefits: {row['benefits']}
Eligibility: {row['eligibility']}
Application Process: {row['application']}
Documents Required: {row['documents']}
Level: {row['level']}
Category: {row['schemeCategory']}
Tags: {row['tags']}
"""
    documents.append(text.strip())

print("Loading embedding model...")

# Embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Creating embeddings...")

embeddings = model.encode(
    documents,
    show_progress_bar=True
)

# Convert embeddings to FAISS format
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(embeddings.astype("float32"))

# Save vector database
faiss.write_index(index, "scheme_index.faiss")

# Save documents separately
with open("scheme_documents.pkl", "wb") as f:
    pickle.dump(documents, f)

print("\nVector database created successfully!")
print("Total schemes:", len(documents))
print("Embedding dimension:", dimension)