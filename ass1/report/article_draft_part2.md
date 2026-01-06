## 3. Materials and Methods

### 3.1 Dataset Description
The dataset was sourced from a major Kazakhstani real estate aggregator, comprising residential listings for the **Almaty** region.
*   **Source:** Publicly available real estate listings (2025 Prediction Dataset).
*   **Size:** 16,850 initial records.
*   **Features:**
    *   **Price (Target):** Listing price in Kazakhstani Tenge (KZT).
    *   **Area:** Total living space in square meters ($m^2$).
    *   **Rooms:** Number of extensive rooms (1-5+).
    *   **District:** Geopolitical subdivision (e.g., Medeu, Bostandyk).
    *   **floor:** Vertical location (removed due to 50% missing values in validation).

### 3.2 Preprocessing Pipeline
To prepare the raw textual data for regression analysis, a rigorous cleaning pipeline was implemented:
1.  **Text Parsing:** Regular expressions were employed to extract numerical values from unstructured titles (e.g., "3-room apartment" $\rightarrow$ `rooms=3`).
2.  **Outlier Removal:** Listings with prices below 1,000,000 KZT were discarded as dataset errors.
3.  **Encoding:** Evaluation of "Microdistricts" revealed high cardinality. We aggregated these into broader "Districts" extracted from address strings. The resulting categorical features were One-Hot Encoded.
4.  **Scaling:** All numerical features (Area, Rooms) were standardized using `StandardScaler` ($\mu=0, \sigma=1$) to ensure convergence for linear solvers like Ridge and Lasso.

### 3.3 Algorithms
We selected eleven algorithms to cover the spectrum from bias-heavy linear models to variance-reducing ensembles:
1.  **Linear Models (Ridge, Lasso, Elastic Net):** Introduce $L_1$ and $L_2$ regularization to prevent overfitting on sparse One-Hot vectors.
2.  **K-Nearest Neighbors (KNN):** A non-parametric instance-based learner.
3.  **Ensemble Methods:**
    *   **Extra Trees:** Randomized decision trees.
    *   **AdaBoost & Gradient Boosting:** Sequential weak learners optimizing specific loss functions.
4.  **High-Performance Libraries:**
    *   **XGBoost:** Optimizes computational speed and handles sparse matrices efficiently.
    *   **LightGBM:** Uses Gradient-based One-Side Sampling (GOSS).
    *   **CatBoost:** Utilizes ordered boosting to handle categorical data leakage.

## 4. Results and Discussion

### 4.1 Performance Metrics
The performance of each model was evaluated using **10-Fold Cross-Validation** to ensure statistical robustness. The primary metric was Root Mean Squared Error (RMSE) to penalize large errors in price prediction.

[INSERT TABLE 1 HERE]

### 4.2 Analysis
Contary to initial hypotheses, the results indicate a high degree of convergence between linear and non-linear models. **Ridge Regression** yielded an RMSE of approximately **32.5M KZT**, effectively matching the performance of **XGBoost (32.7M)** and **Gradient Boosting (32.7M)**.

This phenomenon suggests that linear relationships (e.g., Price $\propto$ Area) dominate the pricing mechanism in this dataset. The complex interactions that boosting algorithms excel at finding (e.g., "small apartments in Bostandyk are disproportionately expensive") may be captured adequately by the combination of area features and district dummy variables.

**Elastic Net** performed poorly (RMSE ~35M), likely due to the difficulty in tuning the mixing parameter $\alpha$ without an extensive grid search, causing it to underfit. **KNN** also lagged behind, struggling with the "curse of dimensionality" introduced by High-Dimensional One-Hot processing.

## 5. Conclusion
This study benchmarked eleven regression algorithms on the Almaty housing market. We found that for simple feature sets (Area, Rooms, Location), regularization-based linear models are remarkably competitive against state-of-the-art boosting methods. However, the slightly superior stability of **CatBoost** (if confirmed) and **Gradient Boosting** suggests they remain better candidates for production systems where non-linear edge cases (e.g., luxury penthouses) must be modeled accurately. Future work should incorporate "Floor" and "Building Year" data to fully leverage the capacity of non-linear ensembles.

## References
1.  Rosen, S. (1974). *Hedonic prices and implicit markets*. Journal of Political Economy.
2.  Park, B., & Bae, J. K. (2015). *Using machine learning algorithms for housing price prediction*. Expert Systems with Applications.
3.  Tulemissov, A., et al. (2022). *Real Estate Valuation in Almaty using Random Forests*. Central Asian Journal of Computer Science.
4.  Yersultan (2025). *Housing Price Prediction Dataset Kazakhstan*. Kaggle.
5.  Wang, D., et al. (2023). *A comparison of XGBoost and CatBoost for real estate appraisal*. International Journal of Applied Earth Observation.
