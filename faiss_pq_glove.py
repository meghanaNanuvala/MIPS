"""
OPQ + PQ MIPS for GloVe (Mac-compatible FAISS)
Uses OPQMatrix instead of IndexOPQ.
"""

import numpy as np
import faiss
import time, psutil, os

# --------------------------
# Load GloVe 50d
# --------------------------
print("Loading GloVe...")

def load_glove(path, dim=50, limit=50000):
    words = []
    vecs = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            parts = line.rstrip().split(" ")
            word = parts[0]
            vec = np.array(parts[1:], dtype=float)
            if len(vec) == dim:
                words.append(word)
                vecs.append(vec)
    return words, np.array(vecs).astype("float32")


words, xb = load_glove("../datasets/glove.6B/glove.6B.50d.txt", dim=50, limit=50000)
xq = xb[:500]        # queries = first 500 vectors
xb_base = xb         # database

print("Loaded:", xb_base.shape)


# --------------------------
# Compute MIPS groundtruth
# --------------------------
print("Computing ground truth...")
t0 = time.time()
sims = xq @ xb_base.T
gt = np.argsort(-sims, axis=1)[:, :10]
t1 = time.time()
print(f"GT computed in: {t1 - t0:.4f} sec")


# --------------------------
# Build OPQ + PQ index
# --------------------------
d = xb_base.shape[1]   # 50
M = 10                 # OPQ/PQ blocks
nbits = 8              # PQ bits

print("\n====================================")
print("   Building OPQ + PQ MIPS Index")
print("====================================")


# 1. Create OPQ rotation matrix
opq = faiss.OPQMatrix(d, M)
opq.train(xb_base)

# 2. PQ codebook
pq = faiss.IndexPQ(d, M, nbits, faiss.METRIC_INNER_PRODUCT)

# 3. Embed OPQ → PQ
index = faiss.IndexPreTransform(opq, pq)

# Train PQ
print("Training...")
t2 = time.time()
index.train(xb_base)

# Add base vectors
index.add(xb_base)
t3 = time.time()

pq = index.index     # IndexPQ inside IndexPreTransform
code_bytes = pq.sa_code_size()
index_memory_mb = (code_bytes * xb_base.shape[0]) / (1024*1024)

build_time = t3 - t2


# --------------------------
# Search
# --------------------------
print("Searching...")
t4 = time.time()
D, I = index.search(xq, 10)
t5 = time.time()

search_time = t5 - t4
latency = search_time / len(xq)


# --------------------------
# Recall calculation
# --------------------------
correct = (I == gt)
recall = correct.sum() / (len(xq)*10)


# --------------------------
# Print metrics
# --------------------------
print("\n============================")
print("      OPQ + PQ  METRICS")
print("============================")
print(f"Recall@10        = {recall:.4f}")
print(f"Build Time       = {build_time:.4f} sec")
print(f"Search Time      = {search_time:.4f} sec for {len(xq)} queries")
print(f"Latency/query    = {latency*1000:.4f} ms")
print(f"Index Memory     = {index_memory_mb:.4f} MB")
print("============================")
