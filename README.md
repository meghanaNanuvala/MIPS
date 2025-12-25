# 📊 Survey & Experimental Analysis of Maximum Inner Product Search (MIPS) Methods

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![ScaNN](https://img.shields.io/badge/ScaNN-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://github.com/google-research/google-research/tree/master/scann)
[![FAISS](https://img.shields.io/badge/FAISS-0467DF?style=for-the-badge&logo=meta&logoColor=white)](https://github.com/facebookresearch/faiss)

> **Authors:** Meghana Nanuvala & Vincent Siddons<br>
> **Course:** Data Mining | Indiana University Indianapolis  
> **Date:** December 2025

A comprehensive experimental study comparing state-of-the-art Maximum Inner Product Search (MIPS) algorithms across four major families: hashing, quantization, graph-based, and bandit methods.


## 🎯 What is MIPS?

Maximum Inner Product Search (MIPS) is a fundamental problem in machine learning and information retrieval. Given a query vector **q** and a database **X** of n vectors, MIPS aims to find:

```
x* = arg max ⟨q, x⟩
     x∈X
```

where ⟨q, x⟩ represents the inner product between vectors.

### Why MIPS Matters

MIPS is crucial for modern AI applications:

- 🤖 **Large Language Models (LLMs)** - Retrieval-Augmented Generation (RAG) systems use MIPS to find relevant context from knowledge bases
- 📱 **Recommendation Systems** - Netflix, Spotify, and Amazon use MIPS to match user preferences with item embeddings
- 🔍 **Semantic Search** - Finding similar documents or images based on learned embeddings
- 💬 **Chatbots & Virtual Assistants** - Retrieving relevant responses from large dialog databases

### The Challenge

For large databases (millions of vectors) in high dimensions (100-1000+), computing inner products with all vectors is computationally prohibitive. This motivates the development of **approximate** MIPS methods that trade a small amount of accuracy for massive speedups.


## 🔬 Methods Overview

This survey evaluates four major MIPS algorithm families:

### 1️⃣ Hashing-Based Methods
Transform vectors into compact binary codes using locality-sensitive hashing (LSH). Similar vectors hash to nearby codes, enabling sublinear search by comparing hash signatures instead of full vectors. **Key papers:** Asymmetric LSH (Shrivastava & Li, 2014), Norm-Ranging LSH (Yan et al., 2018).

**Trade-offs:** ✅ Sublinear query time, provable guarantees 
                | ❌ Accuracy degrades in high dimensions, sensitive to hyperparameters

### 2️⃣ Quantization-Based Methods
Compress vectors into compact codes by partitioning space into Voronoi cells with cluster centroids. Product Quantization (PQ) splits dimensions into subspaces for finer compression. **Implementations:** ScaNN (anisotropic quantization + multi-stage search), FAISS OPQ+PQ (optimized for inner products).

**Trade-offs:** ✅ 8-100x memory compression, scales to billions of vectors 
                | ❌ Lower accuracy, quantization error

### 3️⃣ Graph-Based Methods
Build proximity graphs where nodes are vectors and edges connect similar items. Search via greedy graph traversal from entry points. **HNSW** (Hierarchical Navigable Small World) uses multi-layer structure for logarithmic complexity. **Norm-Adjusted Proximity Graph** adjusts edge weights by vector norms.

**Trade-offs:** ✅ Fastest query latency, high recall, handles dynamic updates 
                | ❌ High memory overhead, slow build time

### 4️⃣ Bandit-Based Methods
Frame MIPS as a sequential decision problem using multi-armed bandit algorithms. Adaptively sample promising candidates using upper confidence bounds to balance exploration-exploitation. **Key papers:** Bandit MIPS (Liu et al., 2019), Coordinate Sampling (Tiwari et al., 2024).

**Trade-offs:** ✅ Scales to high dimensions, reduces evaluations 
                | ❌ Less mature, difficult tuning, limited implementations



## 📊 Experimental Results

### Datasets

**GloVe-100-angular**
- 1.18 million 100-dimensional word embeddings
- 10,000 query vectors
- Designed for MIPS evaluation
- Represents semantic similarity in NLP applications

**SIFT1M**
- 1 million 128-dimensional SIFT image descriptors
- Designed for L2 (Euclidean) distance
- Standard benchmark for image retrieval
- Tests method robustness under metric mismatch

### Performance Comparison

#### GloVe-100-angular (MIPS Task)

| Method | Recall@10 | Query Latency (ms) | Build Time (s) | Memory (MB) |
|--------|-----------|-------------------|----------------|-------------|
| **ScaNN** | **90.02%** | 0.156 | 8.28 | 517.32 |
| **HNSW** | 87.29% | **0.056** | 157.79 | 1168.30 |
| **FAISS OPQ+PQ** | 18.53% | 0.385 | 25.67 | **8.41** |

**Key Insights:**
- ScaNN achieves the best accuracy-efficiency balance for MIPS workloads
- HNSW delivers lowest latency but requires 2x more memory
- FAISS trades accuracy for extreme memory efficiency (60x smaller than HNSW)

#### SIFT1M (L2 Distance Task)

| Method | Recall@10 | Query Latency (ms) | Build Time (s) | Memory (MB) |
|--------|-----------|-------------------|----------------|-------------|
| **FAISS** | **49.54%** | 9.126 | 16.77 | **122.20** |
| **HNSW** | 2.39% | **0.077** | 126.16 | 259.27 |
| **ScaNN** | 2.28% | 0.424 | 5.36 | 597.47 |

**Key Insights:**
- FAISS excels on L2 tasks (aligned with its design metric)
- HNSW and ScaNN suffer from metric mismatch (configured for inner product)
- Demonstrates critical importance of metric alignment



## 🔑 Key Findings

### No Universal Winner

Different methods excel at different metrics. The choice depends on application constraints:

- **Accuracy Priority** → ScaNN (90% recall on MIPS)
- **Latency Priority** → HNSW (0.056 ms query time)
- **Memory Priority** → FAISS OPQ+PQ (8.41 MB index)

### Metric Alignment is Critical

ANN methods perform best when the similarity metric used for indexing matches the dataset geometry:
- MIPS methods (ScaNN, HNSW) excel on inner product tasks (GloVe)
- L2 methods (FAISS) excel on Euclidean distance tasks (SIFT)
- Metric mismatch causes 20-40x accuracy degradation

### System-Level Trade-offs

MIPS is fundamentally a systems design problem:
- **Graph methods** prioritize low latency through efficient traversal
- **Quantization methods** balance accuracy and resource usage
- **Hashing methods** offer theoretical guarantees with practical complexity
- **Bandit methods** reduce evaluations through adaptive sampling



## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat-square)

**Libraries:**
- `scann` - Google's ScaNN implementation for MIPS
- `hnswlib` - Fast HNSW graph-based search
- `faiss-cpu` - Facebook AI Similarity Search
- `numpy` - Numerical computing
- `matplotlib` - Result visualization



## 📁 Repository Structure

```
MIPS/
├── methods/
│   ├── Annoy/
│   │   ├── annoy_srp.py              # Annoy with random projection
│   │   └── annoy_srp_glove.py        # Annoy on GloVe dataset
│   ├── FAISS/
│   │   ├── faiss_pq_glove.py         # FAISS Product Quantization (GloVe)
│   │   └── faiss_pq_sift.py          # FAISS Product Quantization (SIFT)
│   ├── HNSW/
│   │   ├── hnsw_glove.py             # HNSW on GloVe dataset
│   │   └── hnsw_sift.py              # HNSW on SIFT dataset
│   └── SCANN/
│       ├── scann_glove.py            # ScaNN on GloVe dataset
│       └── scann_sift.py             # ScaNN on SIFT dataset
├── MIPS_Survey.pdf                    # Full research paper
├── README.md
└── requirements.txt
```



## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- 8GB+ RAM recommended
- CPU-only (no GPU required)

### Installation

```bash
# Clone the repository
git clone https://github.com/meghanaNanuvala/MIPS.git
cd MIPS

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Download Datasets

The scripts automatically download datasets when run for the first time. Alternatively, manually download:

- **GloVe-100-angular**: [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/)
- **SIFT1M**: [http://corpus-texmex.irisa.fr/](http://corpus-texmex.irisa.fr/)

### Run Experiments

```bash
# Run GloVe experiments
python methods/SCANN/scann_glove.py
python methods/HNSW/hnsw_glove.py
python methods/FAISS/faiss_pq_glove.py
python methods/Annoy/annoy_srp_glove.py

# Run SIFT experiments
python methods/SCANN/scann_sift.py
python methods/HNSW/hnsw_sift.py
python methods/FAISS/faiss_pq_sift.py

# Run Annoy with random projection
python methods/Annoy/annoy_srp.py
```



## 📊 Reproduce Results

Each script is self-contained and produces evaluation metrics:

```bash
# Run individual method on specific dataset
cd methods/SCANN
python scann_glove.py

cd ../HNSW
python hnsw_sift.py

# Or from root directory
python methods/FAISS/faiss_pq_glove.py
```

**Each script outputs:**
- **Recall@10** - Fraction of true top-10 neighbors retrieved
- **Query Latency** - Average time per query (milliseconds)
- **Build Time** - Total index construction time (seconds)
- **Memory Footprint** - Index size in memory (megabytes)



## 📚 References

### Landmark Papers

**Hashing Methods:**
- Charikar, M. (2002). "Similarity estimation techniques from rounding algorithms." *STOC*
- Shrivastava, A., & Li, P. (2014). "Asymmetric LSH (ALSH) for sublinear time maximum inner product search (MIPS)." *NeurIPS*
- Yan, X., et al. (2018). "Norm-ranging LSH for maximum inner product search." *NeurIPS*

**Quantization Methods:**
- Guo, R., et al. (2016). "Quantization-based fast inner product search." *AISTATS*
- Dai, X., et al. (2020). "Norm-explicit quantization: Improving vector quantization for maximum inner product search." *AAAI*
- Guo, R., et al. (2020). "Accelerating large-scale inference with anisotropic vector quantization." *ICML*

**Graph-Based Methods:**
- Malkov, Y., & Yashunin, D. (2020). "Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs." *TPAMI*
- Morozov, S., & Babenko, A. (2018). "Non-metric similarity graphs for maximum inner product search." *NeurIPS*
- Tan, M., et al. (2021). "Norm-adjusted proximity graph for fast inner product retrieval." *KDD*

**Bandit Methods:**
- Liu, R., et al. (2019). "A bandit approach to maximum inner product search." *AAAI*
- Yang, S., et al. (2022). "Linear bandit algorithms with sublinear time complexity." *ICML*
- Tiwari, M., et al. (2024). "Faster maximum inner product search in high dimensions." *ICML*

### Datasets

- **GloVe:** Pennington, J., Socher, R., & Manning, C. (2014). "GloVe: Global vectors for word representation." [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/)
- **SIFT:** Lowe, D. (2004). "Distinctive image features from scale-invariant keypoints." [http://corpus-texmex.irisa.fr/](http://corpus-texmex.irisa.fr/)



## 🤝 Contributing

We welcome contributions! Areas for future work:

- [ ] Implement additional bandit-based methods
- [ ] Add GPU-accelerated versions of algorithms
- [ ] Test on larger datasets (10M+ vectors)
- [ ] Implement streaming/online MIPS algorithms
- [ ] Add more visualization tools
- [ ] Create interactive demo dashboard

To contribute:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request



## 📧 Contact

**Meghana Nanuvala**  
📧 meghana2iiit@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/meghana-nanuvala) | [GitHub](https://github.com/meghanaNanuvala) | [Portfolio](https://meghanananuvala.vercel.app)

**Vincent Siddons**  
📧 Contact via GitHub Issues



## 🙏 Acknowledgments

- **Course:** Data Mining, Indiana University Indianapolis
- **Libraries:** Google ScaNN, Facebook FAISS, hnswlib
- **Datasets:** Stanford NLP (GloVe), INRIA/IRISA (SIFT)
- **Inspiration:** ANN-Benchmarks project and related research papers



## 📄 License

This project is released under the MIT License. See [LICENSE](LICENSE) file for details.



**⭐ If you find this work useful, please star the repository!**

**🔗 Related Projects:**
- [ANN-Benchmarks](https://github.com/erikbern/ann-benchmarks) - Comprehensive ANN library comparison
- [FAISS](https://github.com/facebookresearch/faiss) - Facebook AI Similarity Search
- [ScaNN](https://github.com/google-research/google-research/tree/master/scann) - Google's Scalable Nearest Neighbors
