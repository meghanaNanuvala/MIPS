"""
HNSW Evaluation on GloVe Embeddings
-----------------------------------

We evaluate HNSW for Maximum Inner Product Search (Cosine similarity)
on a sampled subset of GloVe vectors.

Ground-truth is computed via exact cosine similarity.
"""

import numpy as np
import hnswlib
import psutil, os, time


# ----------------------------------------
# Memory utility
# ----------------------------------------
def memory_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


# ----------------------------------------
# Load GloVe text file (50d)
# ----------------------------------------
def load_glove(filepath, max_words=50000):
    words = []
    vecs = []

    with open(filepath, "r", encoding="utf8") as f:
        for i, line in enumerate(f):
            if i >= max_words:
                break
            
            parts = line.strip().split()
            word = parts[0]
            vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)

            words.append(word)
            vecs.append(vec)

    vecs = np.vstack(vecs)
    return words, vecs


# ----------------------------------------
# Build MIPS ground truth (cosine)
# ----------------------------------------
def compute_gt(xb, xq, k=10):
    sims = xq @ xb.T                     # cosine sim because already normalized
    idx = np.argpartition(-sims, k, axis=1)[:, :k]

    # sort each row
    sorted_idx = np.argsort(-sims[np.arange(len(xq))[:, None], idx], axis=1)
    return idx[np.arange(len(xq))[:, None], sorted_idx]


# ----------------------------------------
# Load dataset
# ----------------------------------------
FILE = "../datasets/glove.6B/glove.6B.50d.txt"

print("Loading GloVe embeddings...")
words, xb = load_glove(FILE, max_words=20000)   # sample first 20k
dim = xb.shape[1]

# L2-normalize for cosine/MIPS
xb /= np.linalg.norm(xb, axis=1, keepdims=True)

# Create query set by randomly sampling
np.random.seed(0)
q_idx = np.random.choice(len(xb), size=500, replace=False)

xq = xb[q_idx]
base = xb

print("Base vectors:", base.shape)
print("Query vectors:", xq.shape)


# ----------------------------------------
# Compute ground-truth (exact MIPS)
# ----------------------------------------
print("Computing ground-truth MIPS...")
gt = compute_gt(base, xq, k=10)
print("GT shape:", gt.shape)


# ----------------------------------------
# Build HNSW index
# ----------------------------------------
M = 16
ef_construction = 200
ef_search = 100

mem_before = memory_mb()

index = hnswlib.Index(space="ip", dim=dim)
index.init_index(max_elements=len(base), ef_construction=ef_construction, M=M)

print("\nAdding items to HNSW...")
t0 = time.time()
index.add_items(base)
t1 = time.time()

mem_after = memory_mb()
index_mem = mem_after - mem_before


print(f"Index built in {t1 - t0:.4f} sec")
print(f"Index memory: {index_mem:.2f} MB")


# ----------------------------------------
# Search
# ----------------------------------------
index.set_ef(ef_search)

print("Searching...")
t2 = time.time()
labels, dist = index.knn_query(xq, k=10)
t3 = time.time()

search_time = t3 - t2
latency = (search_time / len(xq)) * 1000


# ----------------------------------------
# Compute Recall & Precision
# ----------------------------------------
correct = (labels == gt)
recall = correct.sum() / (len(xq) * 10)
precision = recall   # same for fixed-k


# ----------------------------------------
# Print results
# ----------------------------------------
print("\n============================")
print("       HNSW (GloVe) METRICS")
print("============================")
print(f"Recall@10     = {recall:.4f}")
print(f"Precision@10  = {precision:.4f}")
print(f"Build Time    = {t1 - t0:.4f} sec")
print(f"Search Time   = {search_time:.4f} sec for {len(xq)} queries")
print(f"Latency/query = {latency:.4f} ms")
print(f"Index Memory  = {index_mem:.2f} MB")
print("============================")
