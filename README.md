# Assignment 3: LSH-Based Near-Replica Mining for Molecular Fingerprints

This project implements a scalable PySpark pipeline for performing near-duplicate molecular detection using MinHash Locality-Sensitive Hashing (LSH).
The goal is to efficiently find molecules with high Tanimoto similarity at scale using distributed processing.
The repository also includes optional scripts for chart generation and scaffold analysis.

## 🚀 Features

- Converts 2048-bit fingerprints into sparse vectors
- Fits MinHashLSH models for numHashTables = 16, 32, 64
- Performs approximate similarity joins
- Computes exact Tanimoto using efficient popcount pruning
- Evaluates:
  - Runtime
  - Shuffle I/O
  - Pair counts
  - Precision & recall
- Optional: Generates runtime/recall charts
- Optional: Analyzes Murcko scaffolds of near-duplicate molecules

## 📁 Repository Structure

```
Assignment3/
├── Assignment3_aws.py              # Main PySpark pipeline
├── generate_charts.py              # Optional chart generation
├── scaffold_analysis_simple.py     # Optional scaffold analysis
├── README.md                       # This file
├── charts/                         # Generated charts (optional)
└── scaffold_analysis/              # Scaffold analysis results (optional)
```

## 🧰 Requirements

### Core (for assignment pipeline)
- pyspark
- xxhash

### Optional (local analysis)
- pandas
- matplotlib
- seaborn
- pyarrow
- rdkit

### Install locally if needed:
```bash
pip install pyspark xxhash pandas matplotlib seaborn pyarrow rdkit
```

## ⚙️ Running the Pipeline

### 1. Upload code

```bash
gsutil cp Assignment3_aws.py gs://<bucket>/code/
```

### 2. Submit job (Dataproc)

```bash
gcloud dataproc jobs submit pyspark \
  gs://<bucket>/code/Assignment3_aws.py \
  --cluster=<cluster-name> \
  --region=<region> \
  --properties="spark.executor.memory=12g,\
                spark.driver.memory=12g,\
                spark.sql.shuffle.partitions=300,\
                spark.sql.autoBroadcastJoinThreshold=-1" \
  -- \
  --parquet_path=gs://<bucket>/a2_df.parquet \
  --output_base=gs://<bucket>/a3_output
```

## 🏎️ Spark Optimizations Used

### ✔ Adaptive Query Execution (AQE)
Automatically handles skew and merges tiny partitions.

### ✔ Salting for Skew Reduction
Reduces partitions with heavy keys.

### ✔ Popcount-Based Pruning
Before exact Tanimoto, eliminate pairs where similarity can never reach τ.
This drastically reduces runtime.

### ✔ Checkpoint / DISK_ONLY Storage
Prevents:
- Huge lineages
- Out-of-memory
- Driver failures

### ✔ Pre-Assembled Sparse Parquet
Avoids recomputing fingerprints and improves job startup.

### ✔ Broadcast Disabled
Safer for large datasets.

## 📊 Key Output Metrics (Simplified)

### Precision
- Always 1.0 across all runs
- ✔ No false positives from LSH

### Recall
- τ = 0.80 → ~0.87
- τ = 0.90 → 1.00
- τ = 0.95 → 1.00

### Near-Replica Fraction
Low dataset redundancy:
- τ = 0.80 → ~0.4%
- τ = 0.90 → ~0.05%
- τ = 0.95 → ~0.02%

## 📦 Output Structure (Simplified)

```
a3_output/
├── metrics/
│   ├── lsh_grid_metrics.csv
│   ├── evaluation_metrics.csv
│   └── deduplication_metrics.csv
├── grid_lsh_pairs/
│   ├── nht_16/pairs.parquet
│   ├── nht_32/pairs.parquet
│   └── nht_64/pairs.parquet
├── samples/
│   └── nht_*/tau_*/sample.parquet
└── exact_tanimoto_pairs/
    └── sample_*/pairs.parquet
```

**Note**: Output files are not included in GitHub because they are large and generated during runtime.

## 🖼️ Optional: Generate Charts

```bash
python3 generate_charts.py
```

**Outputs**:
- charts/chart1_runtime_vs_recall.png
- charts/chart2_shuffle_vs_recall.png
- charts/chart3_precision_vs_recall.png
- charts/chart4_deduplication_vs_tau.png

## 🧪 Optional: Scaffold Analysis

```bash
python3 scaffold_analysis_simple.py
```

**Outputs** stored in:
- scaffold_analysis/

## 🛠️ Troubleshooting (Simple)

### Job slow or stuck
→ Reduce shuffle partitions or DATA_FRACTION.

### Out-of-memory
→ Lower sample size:
```bash
--max_eval_rows=20000
```

### Skew in approxSimilarityJoin
→ Already minimized using:
- AQE skew join
- Salting
- Hash repartitioning

---

**Last Updated**: November 16, 2025
