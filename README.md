# ANN Methods for Maximum Inner Product Search (MIPS)

This directory contains reference implementations of representative **Approximate Nearest Neighbor (ANN)** methods used in our experimental study on **Maximum Inner Product Search (MIPS)** and **Euclidean (L2) similarity search**.  
The code is organized by ANN family and supports experiments on both **GloVe-100-angular** and **SIFT1M** benchmarks.

The implementations are designed to reproduce the results reported in our paper and facilitate systematic comparison across accuracy, latency, build time, and memory usage.

---

## Directory Structure


Each subdirectory corresponds to a distinct ANN family, with dataset-specific scripts.

---

## Implemented Methods

### Annoy (Sign Random Projections)
- **Files:** `annoy_srp.py`, `annoy_srp_glove.py`
- **Family:** Hashing / Random projection
- **Use case:** Fast approximate inner product search using SRP
- **Metric:** Inner Product (MIPS)

---

### FAISS (Product Quantization)
- **Files:** `faiss_pq_glove.py`, `faiss_pq_sift.py`
- **Family:** Quantization-based ANN
- **Indexing:** OPQ + PQ
- **Metrics:**
  - Inner Product (GloVe)
  - L2 distance (SIFT1M)
- **Strength:** Memory efficiency

---

### HNSW (Hierarchical Navigable Small World Graphs)
- **Files:** `hnsw_glove.py`, `hnsw_sift.py`
- **Family:** Graph-based ANN
- **Metric:** Inner Product / L2
- **Strength:** Low query latency via greedy graph traversal

---

### ScaNN
- **Files:** `scann_glove.py`, `scann_sift.py`
- **Family:** Tree + quantization hybrid
- **Metric:** Inner Product / L2
- **Strength:** Strong recall–efficiency trade-off for large-scale MIPS

---

## Datasets

The scripts assume access to the following standard ANN benchmarks:

- **GloVe-100-angular**
  - 1.18M vectors, 100 dimensions
  - Metric: Inner Product (MIPS)
- **SIFT1M**
  - 1M vectors, 128 dimensions
  - Metric: Euclidean (L2)

Exact nearest-neighbor ground truth is required for Recall@k evaluation.

---

## Evaluation Metrics

All methods report the following metrics:

- **Recall@10**
- **Index build time (seconds)**
- **Average query latency (milliseconds)**
- **Total search time (seconds)**
- **Index memory footprint (MB)**

Metrics are computed under identical task definitions and similarity metrics per dataset to ensure fair comparison.

---

## Running Experiments

Each script is self-contained and can be executed directly:

```bash
python scann_glove.py
python hnsw_sift.py
python faiss_pq_glove.py
