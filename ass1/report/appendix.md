## Appendix A: Detailed Algorithmic Reference

To ensure reproducibility, we provide formal definitions for the ensemble methods utilized in this study.

**A.1 Support Vector Machines vs. Tree Ensembles**
While SVMs optimize a margin hyperplane ($w^T x + b = 0$), Tree Ensembles partition the feature space into hyper-rectangles. For the Almaty housing dataset, where relationships are often disjoint (e.g., "District A" vs "District B"), tree-based methods naturally capture these segmentations better than kernel-based SVMs.

**A.2 Gradient Boosting Implementation Details**
*   **Loss Function**: We utilized the Squared Error loss function $L(y, \hat{y}) = \frac{1}{2}(y - \hat{y})^2$.
*   **Regularization**:
    *   **XGBoost**: Uses $\gamma$ (min split loss) and $\lambda$ (L2 regularization on weights).
    *   **LightGBM**: Uses `min_data_in_leaf` to prevent overfitting.
    *   **CatBoost**: Uses `depth` and `l2_leaf_reg`.

**A.3 High-Performance Boosting Libraries Comparison**
1.  **XGBoost**: Known for its "System Optimization." It provides parallelization of tree construction using all of your CPU cores during training. It uses cache-aware access patterns which makes it extremely fast.
2.  **LightGBM**: Known for "Leaf-wise Growth." It chooses the leaf with max delta loss to grow. This can lead to deeper trees than level-wise and potentially lower error, but higher risk of overfitting.
3.  **CatBoost**: Known for "Symmetric Trees." It builds trees where the same split is applied at certain levels, making the structure balanced and less prone to overfitting. It handles categorical features by converting them to numbers using statistics on combinations of categorical features.

## Appendix B: Code Implementation Structure
The project repository is structured to facilitate peer review and replication:
*   `src/train.py`: The core engine utilizing `scikit-learn` Pipelines. It ensures that data leakage is prevented by applying `StandardScaler` only within the Cross-Validation fold.
*   `src/preprocess_real_data.py`: A regex-based parser that converts the raw Russian-language listings (e.g., "3-комнатная") into structured integers.
*   `report/figures/`: High-resolution PNG exports (300 DPI) generated via `matplotlib` and `seaborn`.

## Appendix C: Full Feature List
1.  **Price (KZT)**: The dependent variable.
2.  **Area ($m^2$)**: Continuous variable, found to be the strongest predictor.
3.  **Rooms**: Ordinal variable (1-6).
4.  **District**: One-Hot Encoded categorical variables for:
    *   Almaly
    *   Medeu
    *   Bostandyk
    *   Auezov
    *   Jetysu
    *   Turksib
    *   Alatau
    *   Nauryzbay

This structure ensures that the "Linearity Hypothesis" discussed in Section 4.3 is tested against a comprehensive set of spatial and structural variances.
