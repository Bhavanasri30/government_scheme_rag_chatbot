import faiss
import pickle
from sentence_transformers import SentenceTransformer

# Load vector database
index = faiss.read_index("scheme_index.faiss")

# Load documents
with open("scheme_documents.pkl", "rb") as f:
    documents = pickle.load(f)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Test question
query = "What government schemes are available for students?"

# Convert question into embedding
query_embedding = model.encode([query]).astype("float32")

# Search top 5 relevant schemes
distances, indices = index.search(query_embedding, 5)

print("\nTop 5 Retrieved Schemes:\n")

for i, idx in enumerate(indices[0]):
    print(f"\n--- Result {i + 1} ---")
    print(documents[idx][:1000])
    print("Distance:", distances[0][i])