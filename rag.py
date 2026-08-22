import os
import pickle

from functools import lru_cache
import faiss
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer


load_dotenv()

# The model name must match the model used in create_vector_db.py
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-3.6-flash"

FAISS_INDEX_PATH = "scheme_index.faiss"
DOCUMENTS_PATH = "scheme_documents.pkl"

@lru_cache(maxsize=1)

def load_resources():
    """Load the embedding model, FAISS index and scheme documents."""

    print("Loading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    print("Loading FAISS index...")
    index = faiss.read_index(FAISS_INDEX_PATH)

    print("Loading scheme documents...")
    with open(DOCUMENTS_PATH, "rb") as file:
        documents = pickle.load(file)

    if index.ntotal != len(documents):
        raise ValueError(
            "FAISS index and scheme document counts do not match."
        )

    return embedding_model, index, documents


def retrieve_schemes(
    question,
    embedding_model,
    index,
    documents,
    top_k=5
):
    """Retrieve the most relevant scheme documents."""

    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    ).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    retrieved_schemes = []

    for rank, document_index in enumerate(indices[0]):
        if document_index == -1:
            continue

        retrieved_schemes.append(
            {
                "rank": rank + 1,
                "distance": float(distances[0][rank]),
                "document": documents[document_index]
            }
        )

    return retrieved_schemes


def generate_answer(question, retrieved_schemes):
    """Generate an answer using only retrieved scheme information."""

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY was not found in .env")

    if not retrieved_schemes:
        return (
            "I could not find any relevant schemes in the dataset. "
            "Please provide more details."
        )

    context_sections = []

    for result in retrieved_schemes:
        context_sections.append(
            f"""
Retrieved Scheme {result["rank"]}:

{result["document"]}
"""
        )

    context = "\n---\n".join(context_sections)

    prompt = f"""
You are SchemeSathi, an Indian government scheme information and
preliminary eligibility assistant.

Answer the user's question using ONLY the retrieved scheme information.

STRICT RULES:

1. Do not use outside knowledge.
2. Do not create or invent any scheme.
3. Do not invent eligibility rules, benefits, documents, amounts,
   deadlines, application links or procedures.
4. If information is missing, clearly say:
   "This information is not available in the retrieved data."
5. Do not declare the user officially eligible.
6. Use one of these preliminary eligibility labels:
   - Likely eligible based on the provided information
   - Likely not eligible based on the provided information
   - More information required
7. Explain which details match and which details are missing.
8. Recommend only schemes relevant to the user's question.
9. Present the answer in simple and clear language.
10. End with a reminder to verify current information through the
    official government portal.

For every relevant scheme, use this structure:

### Scheme Name
- Why it is relevant:
- Preliminary eligibility:
- Benefits:
- Required documents:
- Application process:
- Missing information:

USER QUESTION:

{question}

RETRIEVED SCHEME INFORMATION:

{context}
"""

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    if not response.text:
        return "Gemini did not return a response. Please try again."

    return response.text


def ask_schemesathi(question):
    """Run the complete SchemeSathi RAG pipeline."""

    embedding_model, index, documents = load_resources()

    retrieved_schemes = retrieve_schemes(
        question=question,
        embedding_model=embedding_model,
        index=index,
        documents=documents,
        top_k=5
    )

    print("\nRetrieved schemes:")
    print("=" * 60)

    for result in retrieved_schemes:
        print(f"\nResult {result['rank']}")
        print(result["document"][:300])
        print("-" * 60)

    print("\nGenerating grounded answer with Gemini...")

    answer = generate_answer(
        question=question,
        retrieved_schemes=retrieved_schemes
    )

    return answer


if __name__ == "__main__":
    user_question = input(
        "\nAsk SchemeSathi a government scheme question:\n> "
    )

    final_answer = ask_schemesathi(user_question)

    print("\nSCHEMESATHI RESPONSE")
    print("=" * 60)
    print(final_answer)