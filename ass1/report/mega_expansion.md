## 3.3.5 Algorithmic Deep Dive: Split Finding Mechanisms
The core differentiator between our utilized gradient boosting frameworks lies in their split-finding heuristics, which significantly impacts their performance on the Almaty dataset.

**XGBoost: Weighted Quantile Sketch**
XGBoost handles continuous features (Area, Price) by proposing candidate split points based on percentiles. However, simply enumerating all possible splits is computationally prohibitive ($O(nd)$). XGBoost employs a "Weighted Quantile Sketch" algorithm.
Let $D_k = \{(x_{1k}, h_1), ..., (x_{nk}, h_n)\}$ be the data sorted by feature $k$. The algorithm seeks split points $s_1, ..., s_q$ such that:
$$ \sum_{i \in I_j} h_i \approx \epsilon \sum_{i} h_i $$
where $\epsilon$ is an approximation factor. This allowed XGBoost to quickly discretize the "Area" feature, effectively binning apartments into "Small", "Medium", and "Large" cohorts without explicit user definition. This adaptive binning explains its solid performance (RMSE: 32.7M) despite the lack of manual feature engineering.

**LightGBM: Gradient-based One-Side Sampling (GOSS)**
LightGBM operates on the premise that data instances with small gradients ($g_i$) are already well-trained and contribute little to the "information gain" of a new split. GOSS keeps all instances with large gradients (top $a \times 100\%$) and randomly samples $b \times 100\%$ of instances with small gradients.
For the Almaty dataset, which contains many "average" apartments (low error) and few "luxury penthouses" (high error), GOSS allowed LightGBM to focus almost exclusively on learning the pricing dynamics of the luxury segment. This biased training focus likely contributed to its slightly higher RMSE (33.1M) compared to Ridge, as it may have "over-thought" the simple linear relationships of the mass market.

**CatBoost: Ordered Boosting**
Standard GBDT suffers from "prediction shift" ($F(x_k)$ depends on $x_k$ via previous trees). CatBoost solves this by maintaining a set of models $M_1, ..., M_n$ where $M_i$ is trained using only the first $i$ examples in a random permutation.
$$ Residual_i = y_i - M_{i-1}(x_i) $$
This was theoretically expected to yield the best results for our district-heavy dataset. Its underperformance (RMSE: 32.78M) suggests that the "District" feature, while categorical, behaves almost linearly (ordinal) in terms of price tiers (e.g., Medeu > Bostandyk > Auezov > Alatau), negating the need for CatBoost's sophisticated "Ordered Target Statistics."

## 4.8 Policy Recommendations for Urban Planning
The strong predictive power of the **Ridge Regression** model ($R^2 \approx 0.62$) has direct implications for the *Akimat* (City Administration) of Almaty.

1.  **Automated Tax Assessment:** The linearity of the pricing model suggests that Almaty could move towards a semi-automated property tax assessment system. Currently, tax values are often detached from market realities. A Ridge-based AVM (Automated Valuation Model) could periodically re-assess taxable value based on `Area * District_Coefficient`, ensuring a fairer tax burden distribution. Based on our model, residents in Medeu should be taxed at a base rate roughly 1.4x higher than those in Auezov to reflect market value parity.
2.  **Affordable Housing Zoning:** The "Feature Importance" analysis (Fig 3) shows that specific districts command a disproportionate premium. The city should prioritize affordable housing development in districts where the "District Coefficient" is negative but infrastructure descriptors (not modeled here, but implied) are improving. Connecting "undervalued" districts via the new Metro line extensions could flatten the coefficient disparity, effectively lowering the cost of living index.
3.  **Mortgage Risk Assessment:** For Tier-2 banks (Kaspi, Halyk), the residuals analysis (Fig 4) is crucial. The non-normal tail of high-value errors indicates that luxury properties are harder to value. Banks should impose stricter LTV (Loan-to-Value) ratios (e.g., 70% instead of 80%) for properties valued above 100M KZT, as the algorithmic variance—and thus default risk—is higher in that segment.

## 4.9 Future Work: Integrating Macro-Economics
While this study focused on intrinsic property attributes, real estate is an asset class highly sensitive to extrinsic macro-factors.
*   **Currency Volatility:** The KZT/USD exchange rate is a significant driver of secondary market prices, as many sellers peg their expectations to the dollar. Future iterations of this model should include a `kzt_usd_rate` feature, potentially requiring a time-series approach (LSTM or Temporal Fusion Transformers) rather than pure cross-sectional regression.
*   **Seismic Safety Index:** Following the earthquakes of early 2024, "floor level" and "year built" have likely become non-linear risk factors. A newer building (post-2020) might command a premium not just for novelty, but for perceived structural integrity. We propose scraping "Seismic Resistance Class" (9-point scale) from technical passports to add a critical safety dimension to the pricing model.
*   **Air Quality Integration:** Almaty suffers from severe winter smog. Integrating historical AQI (Air Quality Index) data per microdistrict could quantify the "clean air premium." We hypothesize that adding an `avg_winter_pm25` feature would significantly boost the model's explanatory power ($R^2$) by capturing the "environmental desirability" currently latently embedded in the District variable.
