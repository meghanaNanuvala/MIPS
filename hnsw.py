"""
PAPER:
    "Quantization-based Fast Inner Product Search"
     (Jégou, Douze, Furon – Facebook Research)

METHOD:
    OPQ + PQ for Maximum Inner Product Search (MIPS)

LIBRARY:
    FAISS

KEY PARAMETERS:
    M = 16
        Number of PQ subquantizers
    nbits = 8
        Bits per subquantizer (256 centroids)
    space: inner-product search (default behavior of OPQ+PQ)
"""

import faiss
import numpy as np
import time
import psutil, os

# ----------------------------------------
# Memory utility
# ----------------------------------------
def memory_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


# ----------------------------------------
# Load your dataset here
# ----------------------------------------
def read_fvecs(fname):
    a = np.fromfile(fname, dtype=np.float32)
    d = a.view(np.int32)[0]
    return a.reshape(-1, d + 1)[:, 1:].astype('float32')

def read_ivecs(fname):
    a = np.fromfile(fname, dtype=np.int32)
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:]

print("Loading SIFT1M dataset...")
path = "../datasets/sift_1M/"

xb = read_fvecs(path + "sift_base.fvecs")     # (1M, 128)
xq = read_fvecs(path + "sift_query.fvecs")    # (10k, 128)

print("Base:", xb.shape)
print("Query:", xq.shape)


# ----------------------------------------
# Normalize (so inner product = cosine similarity)
# ----------------------------------------
faiss.normalize_L2(xb)
faiss.normalize_L2(xq)


# ----------------------------------------
# Compute ground truth for MIPS
# ----------------------------------------
print("Computing MIPS groundtruth (exact)...")
tgt0 = time.time()
sims = xq @ xb.T
gt_mips = np.argsort(-sims, axis=1)[:, :10]
tgt1 = time.time()

print(f"GT computed in {tgt1-tgt0:.2f} sec")


# ----------------------------------------
# Build OPQ + PQ Index
# ----------------------------------------
d = xb.shape[1]          # dimension 128 (SIFT)
M = 16                   # subquantizers
nbits = 8                # 256 centroids
k = 10                   # recall@10

print("\n==============================")
print("   Building OPQ + PQ index")
print("==============================")

# OPQ rotation layer (Improved PQ)
opq = faiss.OPQMatrix(d, M)

# PQ codebook in inner-product domain
pq = faiss.IndexPQ(d, M, nbits, faiss.METRIC_INNER_PRODUCT)

# Chain OPQ → PQ
index = faiss.IndexPreTransform(opq, pq)

# Track memory
mem_before = memory_mb()

# Train and add
t0 = time.time()
index.train(xb)
index.add(xb)
t1 = time.time()

mem_after = memory_mb()
index_mem = mem_after - mem_before


# ----------------------------------------
# Search
# ----------------------------------------
print("Searching...")
t2 = time.time()
D, I = index.search(xq, k)
t3 = time.time()

search_time = t3 - t2

# ----------------------------------------
# Recall
# ----------------------------------------
recall = (I == gt_mips).sum() / (len(xq) * k)
latency = search_time / len(xq)

# ----------------------------------------
# Results
# ----------------------------------------
print("\n============================")
print("        OPQ + PQ RESULTS")
print("============================")
print(f"Recall@10        = {recall:.4f}")
print(f"Build Time       = {t1 - t0:.4f} sec")
print(f"Search Time      = {search_time:.4f} sec for {len(xq)} queries")
print(f"Latency/query    = {latency*1000:.4f} ms")
print(f"Index Memory     = {index_mem:.2f} MB")
print("============================")
