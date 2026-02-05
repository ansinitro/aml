# Advanced Recommender Systems for E-commerce: A Comparative Study

## Research Identity
This work investigates the performance, robustness, and practical trade-offs of classical, hybrid, and neural recommender systems for e-commerce under realistic constraints such as sparsity and cold-start.

## Goal
To move beyond simple rating prediction and understand *why* certain ranking architectures outperform others in specific data regimes (e.g., long-tail items, new users).

## Methodology
We define the task as **Top-K Item Recommendation** using **Implicit Feedback**.
The study compares:
1.  **Classical**: Matrix Factorization (ALS/BPR), SVD++
2.  **Hybrid**: LightFM (handling metadata)
3.  **Neural**: Neural Collaborative Filtering (NCF)

## Project Structure
- `data/`: Raw and processed datasets.
- `notebooks/`: Exploratory analysis and experiments (Python scripts).
- `src/`: Reusable code for data pipelines and models.
- `reports/`: LaTeX source for the final research paper.
- `references/`: Literature and related work.

## Setup
1.  Install dependencies: `pip install -r requirements.txt`
2.  Run data pipeline: `python src/data_pipeline.py`
