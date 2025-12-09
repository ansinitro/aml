# Lab 1: Kazakhstan Basin Water Level Prediction

**Course:** Advanced Machine Learning  
**Task:** Regression Analysis with Hyperparameter Tuning

---

## 📁 Project Structure

```
lab1/
├── src/
│   └── lab1_analysis.py    # Main analysis script
├── docs/
│   ├── lab1_report.md      # Formal academic report
│   └── presentation.html   # HTML slide presentation (← → keys)
├── output/
│   ├── lab1_results.csv    # Model performance results
│   └── best_parameters.csv # Optimal hyperparameters
├── .venv/                  # Python virtual environment
├── requirements.txt        # Dependencies
└── README.md               # This file
```

---

## 🚀 How to Run

### 1. Activate Virtual Environment
```powershell
.venv\Scripts\activate
```

### 2. Install Dependencies (if needed)
```powershell
pip install -r requirements.txt
```

### 3. Run Analysis
```powershell
python src/lab1_analysis.py
```

---

## 📊 Results

| Model | Avg R² | Status |
|-------|--------|--------|
| Decision Tree | 0.9699 | ✅ Best |
| Random Forest | 0.9698 | ✅ |
| KNN (tuned) | 0.9673 | ✅ |
| Hybrid (DT+MLP) | 0.9699 | ✅ |

---

## 📄 Documentation

- **Report:** Open `docs/lab1_report.md` for full methodology and analysis
- **Presentation:** Open `docs/presentation.html` in browser (use ← → arrow keys)

---

## 🔧 Requirements

- Python 3.10+
- pandas, numpy, scikit-learn, kagglehub, matplotlib, tabulate
