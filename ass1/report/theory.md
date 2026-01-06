## 3. Theoretical Framework

To effectively analyze regression performance, it is imperative to understand the mathematical underpinnings of the select algorithms. We have categorized them into three distinct families: Linear Regularizers, Instance-Based Learners, and Ensemble Estimators.

### 3.1 Regularized Linear Models
Traditional Ordinary Least Squares (OLS) minimizes the residual sum of squares (RSS). However, in the presence of multicollinearity or high-dimensional features (such as our One-Hot formatted district data), OLS becomes unstable. Regularization introduces a penalty term to the loss function.

**3.1.1 Ridge Regression ($L_2$ Regularization)** adds a penalty equal to the square of the magnitude of coefficients. The cost function is defined as:
$$ J(\theta) = RSS + \lambda \sum_{i=1}^{n} \theta_i^2 $$
Ridge regression shrinks coefficients toward zero but never exactly to zero, making it ideal for handling multicollinearity while retaining all features.

**3.1.2 Lasso Regression ($L_1$ Regularization)**, or Least Absolute Shrinkage and Selection Operator, alters the penalty term to the absolute value of coefficients:
$$ J(\theta) = RSS + \lambda \sum_{i=1}^{n} |\theta_i| $$
This creates a diamond-shaped constraint region that allows some coefficients to become exactly zero. Lasso thus acts as an embedded feature selection method, sparse-ifying the model.

**3.1.3 Elastic Net** combines the penalties of Ridge and Lasso:
$$ J(\theta) = RSS + \lambda_1 \sum |\theta_i| + \lambda_2 \sum \theta_i^2 $$
It overcomes the limitations of Lasso (which can behave erratically when $p > n$) by maintaining the grouping effect of Ridge while allowing for sparsity.

### 3.2 Instance-Based Learning
**3.2.1 K-Nearest Neighbors (KNN)** is a non-parametric algorithm that assumes similar data points exist in close proximity. For a query point $x_q$, KNN identifies the $k$ closest training examples using a distance metric (typically Euclidean):
$$ d(x_q, x_i) = \sqrt{\sum (x_{qj} - x_{ij})^2} $$
The prediction is the average of the targets of these neighbors. While conceptually simple, KNN suffers heavily from the "curse of dimensionality," where distance becomes less meaningful in high-dimensional spaces.

### 3.3 Ensemble Methods (Bagging and Boosting)
Ensemble learning combines multiple "weak learners" (typically decision trees) to form a strong predictor.

**3.3.1 Extra Trees (Extremely Randomized Trees)** is a Bagging (Bootstrap Aggregating) method similar to Random Forest but with two key differences: it uses the whole dataset instead of bootstrap samples, and split points are selected completely at random rather than optimally. This increases bias slightly but significantly reduces variance and computational cost.

**3.3.2 Adaptive Boosting (AdaBoost)** adapts by tweaking weights of instances in the dataset. Subsequent predictors focus more on difficult cases properly. For regression, it fits a sequence of weak learners on repeatedly modified versions of the data.

**3.3.3 Gradient Boosting Regression (GBR)** generalizes boosting by optimizing an arbitrary differentiable loss function. Instead of updating weights, it trains subsequent models to predict the *residuals* (errors) of the prior models:
$$ F_{m}(x) = F_{m-1}(x) + \nu h_m(x) $$
where $h_m$ is the weak learner trained on pseudo-residuals and $\nu$ is the learning rate.

### 3.4 High-Performance Gradient Boosting
Second-generation boosting libraries have introduced system-level optimizations for speed and accuracy.

**3.4.1 XGBoost (eXtreme Gradient Boosting)** introduced a regularized objective function explicitly into the tree building process to control complexity. It employs a "weighted quantile sketch" for handling sparse data (like our district encodings) and block structure for parallel learning.

**3.4.2 LightGBM** (Light Gradient Boosting Machine) by Microsoft uses two novel techniques: Gradient-based One-Side Sampling (GOSS) to keep instances with large gradients (errors) and Exclusive Feature Bundling (EFB) to reduce dimensionality. It grows trees "leaf-wise" rather than "level-wise," which can converge faster but risks overfitting on small datasets.

**3.4.3 CatBoost** (Categorical Boosting) by Yandex is designed to handle categorical data natively without explicit preprocessing. It uses "Ordered Boosting" to overcome prediction shift, a common issue where target leakage occurs in standard GBDT implementations. It builds symmetric trees, which are different from the asymmetric trees in XGBoost/LightGBM, often leading to more stable execution.

**3.4.4 HistGradientBoosting** is Scikit-Learn's implementation inspired by LightGBM, binning continuous features into integer-valued histograms to speed up the split finding process by orders of magnitude compared to standard GBR.
