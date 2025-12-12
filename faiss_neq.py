import numpy as np
import faiss
import time
import psutil, os


# ------------------------------ Utilities ------------------------------
def memory_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

def read_fvecs(fname):
    data = np.fromfile(fname, dtype=np.float32)
    d = data.view(np.int32)[0]
    return data.reshape(-1, d + 1)[:, 1:]

def read_ivecs(fname):
    a = np.fromfile(fname, dtype=np.int32)
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:]


# ------------------------------ Load SIFT1M ------------------------------
print("Loading SIFT1M...")
path = "../datasets/sift_1M/"

xb = read_fvecs(path + "sift_base.fvecs")       # (1,000,000,128)
xq = read_fvecs(path + "sift_query.fvecs")      # (10,000,128)
gt = read_ivecs(path + "sift_groundtruth.ivecs")[:, :10]

xb = xb.astype("float32")
xq = xq.astype("float32")

N, d = xb.shape
k = 10

print("Base:", xb.shape, "Query:", xq.shape)


# ------------------------------ Compute MIPS GT ------------------------------
print("Computing ground truth MIPS...")
t0 = time.time()
sims = xq @ xb.T
gt_mips = np.argsort(-sims, axis=1)[:, :k]
print("GT time:", time.time() - t0, "sec")


# ------------------------------ NEQ: Split into Norm + Unit Direction ------------------------------
norms = np.linalg.norm(xb, axis=1)
xb_unit = xb / norms[:, None]

# ------------------------------ Train PQ on unit vectors ------------------------------
M = 16
nbits = 8

print("\nTraining PQ...")
pq = faiss.IndexPQ(d, M, nbits, faiss.METRIC_INNER_PRODUCT)
pq.train(xb_unit)

print("Encoding unit vectors...")
codes_pq = pq.sa_encode(xb_unit)

# ------------------------------ Train SQ on norms ------------------------------
print("Training scalar quantizer on norms...")
sq = faiss.IndexScalarQuantizer(1, faiss.ScalarQuantizer.QT_8bit_uniform)
sq.train(norms.reshape(-1, 1))

print("Encoding norms...")
codes_norm = sq.sa_encode(norms.reshape(-1, 1))


# ------------------------------ Search Phase ------------------------------
print("\nSearching NEQ...")
t2 = time.time()

# PQ look-up-tables
LUTs = [pq.get_LUT(q) for q in xq]

# decode norms
norms_approx = sq.sa_decode(codes_norm).reshape(-1)

I = np.zeros((xq.shape[0], k), dtype=int)

for qi, q in enumerate(xq):
    lut = LUTs[qi]
    # direction score via PQ LUT scanning
    dir_scores = pq.compute_codes_inner_products(codes_pq, lut)

    # final MIPS score = direction × norm
    scores = dir_scores * norms_approx

    # Top-k
    I[qi] = np.argpartition(-scores, k)[:k]

t3 = time.time()


# ------------------------------ Metrics ------------------------------
correct = (I == gt_mips).sum()
recall = correct / (len(xq) * k)
latency = (t3 - t2) / len(xq)
mem = memory_mb()

print("\n==============================")
print("         NEQ RESULTS (SIFT)")
print("==============================")
print("Recall@10      =", recall)
print("Search Time    =", t3 - t2, "sec for", len(xq), "queries")
print("Latency/query  =", latency * 1000, "ms")
print("Memory Used    =", mem, "MB")
