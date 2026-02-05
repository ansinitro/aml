#!/bin/bash
set -e

# Activate venv
source venv/bin/activate

echo "======================================"
echo "Running Experiments"
echo "======================================"

echo "[1/4] Running Popularity Baseline..."
python src/experiment.py --model pop > reports/res_pop.txt
cat reports/res_pop.txt

echo "[2/4] Running Item-KNN Baseline..."
python src/experiment.py --model knn --k 20 > reports/res_knn.txt
cat reports/res_knn.txt

echo "[3/4] Running Matrix Factorization (Classical/Latent)..."
python src/experiment.py --model mf --epochs 30 --embedding_dim 32 --lr 0.005 > reports/res_mf.txt
cat reports/res_mf.txt

echo "[4/4] Running Neural Collaborative Filtering (NCF)..."
python src/experiment.py --model ncf --epochs 30 --embedding_dim 32 --lr 0.001 > reports/res_ncf.txt
cat reports/res_ncf.txt

echo "======================================"
echo "All Experiments Completed."
echo "Results saved in reports/"
