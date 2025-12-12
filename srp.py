"""
===============================================================
               SIGN RANDOM PROJECTION (SRP)
  Implementation of Charikar (2002) - Similarity Estimation 
           Techniques from Rounding Algorithms
===============================================================

This file implements:
  1. Random hyperplane generation
  2. SRP hashing (sign random projection)
  3. Hamming similarity
  4. LSH bucket structure (SRP-LSH)
  5. Cosine vs Hamming comparison
  6. Spotify-style playlist example
  7. Empirical verification of collision probability
  
===============================================================
                       PSEUDOCODE
===============================================================

Given:
  - A d-dimensional vector x
  - k random hyperplanes r1, r2, ..., rk

To compute an SRP hash:

1. Generate k random hyperplanes:
    For i in 1..k:
        ri ← random Gaussian vector of dimension d

2. For each hyperplane ri:
       projection = dot(ri, x)
       if projection >= 0:
           hash[i] = 1
       else:
           hash[i] = 0

3. The k-bit binary vector is the SRP signature of x.

4. Hamming similarity:
       sim = fraction of matching bits 
            = mean(hash_x[i] == hash_y[i])

5. SRP-LSH bucket structure:
       Convert SRP signature to a tuple (hash key)
       Store item in dict[bucket_key]
       Query retrieves all items in same SRP bucket.

===============================================================
"""

import numpy as np
from numpy.linalg import norm
from collections import defaultdict


# ================================================================
# 1. fvecs / ivecs file readers
# ================================================================

def read_fvecs(path):
    """
    Reads a .fvecs file (float vectors) used in SIFT datasets.
    Each vector is stored as:
        [int32 dim] [float32 x1] [float32 x2] ... [float32 xd]
    """
    a = np.fromfile(path, dtype=np.int32)
    if a.size == 0:
        raise ValueError(f"File {path} is empty or not found.")

    d = a[0]                              # dimension
    return a.reshape(-1, d + 1)[:, 1:].astype(np.float32)


def read_ivecs(path):
    """
    Reads a .ivecs file (int vectors) for groundtruth indices.
    """
    a = np.fromfile(path, dtype=np.int32)
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:]    # return int32 matrix


# ================================================================
# 2. SRP Hyperplane Generator
# ================================================================

def generate_hyperplanes(dim, k, seed=42):
    np.random.seed(seed)
    return np.random.randn(k, dim)


# ================================================================
# 3. SRP Hashing (Charikar 2002)
# ================================================================

def srp_hash(x, hyperplanes):
    """
    x: vector (dim,)
    hyperplanes: matrix (k, dim)
    return: k-bit SRP signature (0/1)
    """
    proj = hyperplanes @ x
    return (proj >= 0).astype(np.uint8)


# ================================================================
# 4. SRP-LSH Structure
# ================================================================

class SRPLSH:
    """
    A single-table SRP-LSH structure.
    Stores items in buckets keyed by SRP signature (tuple of bits).
    """
    def __init__(self, dim, k):
        self.k = k
        self.hyperplanes = generate_hyperplanes(dim, k)
        self.buckets = defaultdict(list)

    def add(self, idx, vec):
        sig = tuple(srp_hash(vec, self.hyperplanes))
        self.buckets[sig].append(idx)

    def query(self, vec):
        sig = tuple(srp_hash(vec, self.hyperplanes))
        return self.buckets.get(sig, [])


# ================================================================
# 5. Recall@K Evaluation
# ================================================================

def recall_at_k(results, groundtruth, k):
    """
    results: list of retrieved indices
    groundtruth: true top-100 neighbors from dataset
    returns recall@k
    """
    results_k = set(results[:k])
    gt_k = set(groundtruth[:k])
    return len(results_k & gt_k) / k


# ================================================================
# 6. MAIN SRP + SIFT10K EXPERIMENT
# ================================================================

def run_srp_sift10k_experiment():

    print("\n====================================")
    print("       SRP + ANN_SIFT10K TEST")
    print("====================================")

    # --------------------------
    # Load dataset
    # --------------------------
    print("\nLoading SIFT10K dataset...")
    path = "/Users/mnanuva/Desktop/MIPS/datasets/siftsmall/"

    base = read_fvecs(path+"siftsmall_base.fvecs")          # (10000, 128)
    query = read_fvecs(path+"siftsmall_query.fvecs")        # (100, 128)
    gt = read_ivecs(path+"siftsmall_groundtruth.ivecs")     # (100, 100)

    print("Base vectors:  ", base.shape)
    print("Query vectors: ", query.shape)
    print("Groundtruth:   ", gt.shape)

    dim = base.shape[1]

    # --------------------------
    # Build SRP-LSH Index
    # --------------------------
    k_bits = 32   # small k → more collisions → better recall
    print(f"\nBuilding SRP-LSH index with k={k_bits} bits...")

    lsh = SRPLSH(dim, k_bits)

    for i, v in enumerate(base):
        lsh.add(i, v)

    print("Index built.")

    # --------------------------
    # Compute Recall@K
    # --------------------------
    K = 10
    print(f"\nRunning SRP Recall@{K} evaluation...")

    recalls = []

    for qi in range(len(query)):
        candidates = lsh.query(query[qi])

        if len(candidates) == 0:
            recalls.append(0.0)
        else:
            recalls.append(recall_at_k(candidates, gt[qi], K))

    avg_recall = sum(recalls) / len(recalls)

    print(f"\n====================================")
    print(f"      SRP Recall@{K}: {avg_recall:.4f}")
    print(f"====================================")


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    run_srp_sift10k_experiment()