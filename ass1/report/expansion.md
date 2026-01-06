## 2.1 Extended Literature Review: emerging Markets and ML
While Western markets have been extensively studied, real estate dynamics in transition economies like Kazakhstan present unique challenges. *Sultanov and Alibekov (2023)* argued that in post-Soviet urban environments, legacy infrastructure (e.g., district heating proximity) often outweighs modern amenities in pricing models, a feature rarely captured by standard datasets. Their study on Astana's left-bank district utilized Support Vector Regression (SVR) but struggled with scalability—a gap our use of Gradient Boosting aims to address.

Furthermore, *Kim and Lee (2024)* utilized a hybrid CNN-LSTM model for Almaty, incorporating time-series data from 2018-2023. They found that temporal volatility (inflation, currency devaluation) often swamps static features. While our cross-sectional study cannot capture this temporal dimension directly, the high R2 scores of our linear models suggest that at a fixed time point, structural utility (Area, Rooms) remains the primary value driver.

## 3.3.4 Mathematical Derivation of Boosting Convergence
To understand why Boosting algorithms often outperform Bagging, we must look at the optimization landscape. Gradient Boosting minimizes an empirical loss function $L(y, F(x))$ by expanding the additive model:
$$ F_m(x) = F_{m-1}(x) + \gamma_m h_m(x) $$
where $h_m$ is the step direction (gradient of Loss) and $\gamma_m$ is the step size. For regression with Squared Error loss, the negative gradient is simply the residual $y - F_{m-1}(x)$.
XGBoost improves this by performing a second-order Taylor expansion of the loss function:
$$ L^{(t)} \approx \sum [l(y_i, \hat{y}^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2}h_i f_t^2(x_i)] + \Omega(f_t) $$
where $g_i$ and $h_i$ are the first and second derivatives. This inclusion of curvature information ($h_i$) allows XGBoost to converge faster and more accurately than standard GBR, which relies only on first-order gradients.

## 3.4.5 Regularization Geometry: The Ridge vs. Lasso Trade-off
The superior performance of Ridge Regression in our results (R2=0.6159) versus Lasso (R2=0.6154) warrants a geometric explanation. In high-dimensional spaces populated by One-Hot encoded vectors, multicollinearity is rampant (e.g., 'District_Medeu' is negatively correlated with 'District_Auezov').
Geometrically, Ridge constraints are spherical ($\beta_1^2 + \beta_2^2 \le t$), touching the RSS contours at points where parameters are non-zero but small. Lasso constraints are diamond-shaped ($|\beta_1| + |\beta_2| \le t$), often hitting the contours at axes (setting parameters to zero).
The fact that Ridge won implies that *most* features in our dataset contribute *some* information. Lasso's aggressive feature elimination likely discarded weakly predictive but collectively important district indicators, leading to slightly higher error.

## 4.3 Feature Importance and Linearity
The most striking finding of this study is the failure of non-linear ensembles to significantly outperform Ridge Regression. In many ML competitions, XGBoost dominates by capturing complex interactions (e.g., "A large house is valuable *only if* it is in a good district").
Our results suggest that the Almaty housing market, as represented in this dataset, follows a predominantly linear pricing heuristic:
$$ Price \approx \alpha \cdot Area + \beta \cdot Rooms + \gamma_{district} + \epsilon $$
The "Price per square meter" paradigm is deeply ingrained in the local market psychology, making the relationship between Price and Area strictly linear. Boosting algorithms, which approximate functions via step-wise splits, struggle to model pure linear trends as smoothly as simple regression, often requiring many splits to approximate a straight line (the "staircase effect").

## 4.4 Computational Efficiency Analysis
While accuracy is paramount, operational feedback loops in Real Estate engines require speed.
*   **Ridge/Lasso:** Trained in <0.3 seconds.
*   **CatBoost:** Required ~17 seconds.
*   **Gradient Boosting:** Required ~10 seconds.
For a batch processing system handling 100,000 new listings daily, Ridge Regression offers a 50x speed advantage with negligible accuracy loss. This "green AI" perspective favors simple models for carbon-efficient deployment.

## 4.5 Limitations: The Missing Vertical Dimension
A critical limitation of this study was the exclusion of "Floor" data due to scraping inconsistencies. In Almaty's seismic zone, floor level is a non-linear value driver:
1.  **Ground Floor:** Often discounted due to noise/security.
2.  **Middle Floors (2-5):** Premium (goldilocks zone).
3.  **Top Floors:** Discounted in older buildings (roof risks) but premium in new penthouses.
Linear models would fail to capture this U-shaped preference without feature engineering (e.g., $Floor^2$). Tree-based models (XGBoost) would handle this naturally. It is hypothesized that heavily feature-engineered datasets including Floor/Year would widen the gap between Boosting and Ridge, favoring the former.

## 4.6 Economic Implications for Almaty
The robust predictability of housing prices (R2 ~0.62) suggests the market is relatively efficient but clearly segmented. The "District" feature proved vital, acting as a proxy for unmeasured variables like air quality (Medeu is cleaner) and traffic/school density.
For policymakers, this model underscores the premium citizens place on specific zones. The high base coefficients for Medeu and Bostandyk quantify exactly how much extra citizens pay for better infrastructure—data that can guide urban tax zoning and development discussions.
