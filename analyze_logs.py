# analyze_logs.py
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Load processed dataset and embeddings
# -----------------------------
df = pd.read_csv("processed_logs.csv")
embeddings = np.load("summary_embeddings.npy")

print(f"✅ Loaded {len(df)} records and embeddings.")

# -----------------------------
# Load model (offline)
# -----------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

# -----------------------------
# Function: semantic search
# -----------------------------
def search(query, top_k=5):
    query_vec = model.encode([query])
    scores = cosine_similarity(query_vec, embeddings)[0]
    top_indices = scores.argsort()[-top_k:][::-1]

    results = df.iloc[top_indices].copy()
    results['score'] = scores[top_indices]
    return results[['timestamp', 'platform_type', 'mission_type', 'summary', 'score']]

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    q = input("Enter a query (example: 'Strike on Sector B2'): ")
    results = search(q, top_k=5)
    print("\nTop matching logs:\n")
    print(results.to_string(index=False))
