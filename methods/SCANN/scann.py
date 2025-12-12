'''
Input:
    Xb: 100k SIFT base vectors (128 dims)
    Xq: 100 queries

Procedure:
  1. Build ScaNN index:
        - Partition into sqrt(N) = ~316 leaves
        - Search top 40 leaves (~12%)
        - Use Asymmetric Hashing (block size = 2)
        - Keep top 200 candidates for exact rescoring

  2. Search:
        - Compute approximate scores (fast)
        - Recompute exact scores for 200 best
        - Output top-10 neighbors

  3. Evaluate Recall@10 using groundtruth.

'''


import numpy as np
import tensorflow as tf
import tensorflow_recommenders_addons as tfra
import time

# -----------------------------
# Helpers to read .fvecs
# -----------------------------
def read_fvecs(fname):
    with open(fname, "rb") as f:
        data = np.fromfile(f, dtype=np.int32)
        d = data[0]
        return data.reshape(-1, d + 1)[:, 1:].astype(np.float32)

def read_ivecs(fname):
    iv = np.fromfile(fname, dtype=np.int32)
    d = iv[0]
    return iv.reshape(-1, d + 1)[:, 1:]

# -----------------------------
# Load SIFT100K
# -----------------------------
path = "../datasets/sift_100k/"   # YOUR PATH HERE

print("Loading SIFT100K...")
xb = read_fvecs(path + "sift_base.fvecs")     # (100000,128)
xq = read_fvecs(path + "sift_query.fvecs")    # (100,128)
gt = read_ivecs(path + "sift_groundtruth.ivecs")[:, :10]

print("Base:", xb.shape, "Query:", xq.shape, "GT:", gt.shape)

xb_tf = tf.constant(xb)
xq_tf = tf.constant(xq)

# -----------------------------
# ScaNN configuration
# -----------------------------
num_leaves = 316                   # sqrt(100k)
num_leaves_to_search = 40          # ~12%
dimensions_per_block = 2
reordering_num_neighbors = 200
k = 10

# -----------------------------
# Build ScaNN
# -----------------------------
print("\nBuilding ScaNN index...")

searcher = tfra.layers.ScaNN(
    num_leaves=num_leaves,
    num_leaves_to_search=num_leaves_to_search,
    distance_measure="dot_product",
    training_sample_size=20000,
    dimensions_per_block=dimensions_per_block,
    reordering_num_neighbors=reordering_num_neighbors,
    k=k,
)

searcher = searcher.build(xb_tf)

# -----------------------------
# Searching
# -----------------------------
print("Searching...")
start = time.time()
neighbors, distances = searcher.search(xq_tf)
end = time.time()

neighbors = neighbors.numpy()

# -----------------------------
# Compute Recall@10
# -----------------------------
recall = (neighbors == gt).sum() / (len(xq) * k)

print("\n============================")
print(" ScaNN Recall@10:", recall)
print("============================")
print(f"Search time: {end-start:.4f} sec")
