import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# Load dataset and embeddings
# ----------------------------
df = pd.read_csv("synthetic_drone_logs.csv")
embeddings = np.load("summary_embeddings.npy")

# Load embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ----------------------------
# GUI Setup
# ----------------------------
root = tk.Tk()
root.title("Drone Log Analyzer")
root.geometry("800x600")
root.configure(bg="#1e1e2f")  # dark background

# Title label
title = tk.Label(root, text="Drone Log Analyzer", font=("Helvetica", 22, "bold"), bg="#1e1e2f", fg="#FFD700")
title.pack(pady=15)

# Query frame
query_frame = tk.Frame(root, bg="#1e1e2f")
query_frame.pack(pady=10)

query_label = tk.Label(query_frame, text="Enter your query:", font=("Helvetica", 14), bg="#1e1e2f", fg="#FFFFFF")
query_label.pack(side=tk.LEFT, padx=5)

query_entry = tk.Entry(query_frame, font=("Helvetica", 14), width=50)
query_entry.pack(side=tk.LEFT, padx=5)

# Result frame
result_frame = tk.Frame(root, bg="#2e2e3e", bd=2, relief=tk.RIDGE)
result_frame.pack(pady=20, fill=tk.BOTH, expand=True)

result_text = tk.Text(result_frame, font=("Helvetica", 12), bg="#2e2e3e", fg="#FFFFFF", wrap=tk.WORD)
result_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# Load image
img = Image.open("Untitled.jpeg")
img = img.resize((400, 200))
photo = ImageTk.PhotoImage(img)

image_label = tk.Label(root, image=photo, bg="#1e1e2f")
image_label.pack(pady=10)

# ----------------------------
# Search function
# ----------------------------
def search_query():
    query = query_entry.get().strip()
    if not query:
        messagebox.showwarning("Input Required", "Please enter a query!")
        return
    
    query_emb = model.encode([query])
    sims = cosine_similarity(query_emb, embeddings)[0]
    idx = np.argmax(sims)
    
    summary = df.iloc[idx]["summary"]
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, f"Most Relevant Log Summary:\n\n{summary}")

# Search button
search_btn = tk.Button(root, text="Search", font=("Helvetica", 14, "bold"),
                       bg="#FFD700", fg="#1e1e2f", command=search_query)
search_btn.pack(pady=10)

# Start GUI loop
root.mainloop()
