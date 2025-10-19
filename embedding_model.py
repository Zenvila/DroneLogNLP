# embedding_model.py
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("synthetic_drone_logs.csv")
df['summary'] = df['summary'].astype(str).str.strip()

print(f"✅ Loaded dataset with {len(df)} records.")

# -----------------------------
# Load SentenceTransformer model
# -----------------------------
# First run requires internet to download the model
# After that, it will work offline from cache
model = SentenceTransformer('all-MiniLM-L6-v2')

# -----------------------------
# Generate embeddings
# -----------------------------
print("🔍 Generating embeddings...")
embeddings = model.encode(df['summary'].tolist(), show_progress_bar=True)

# -----------------------------
# Save embeddings locally
# -----------------------------
np.save('summary_embeddings.npy', embeddings)
df.to_csv('processed_logs.csv', index=False)
print("✅ Embeddings saved as 'summary_embeddings.npy'")
