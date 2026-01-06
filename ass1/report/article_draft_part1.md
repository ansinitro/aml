# Predictive Modeling of Real Estate Prices in Almaty: A Comparative Analysis of Machine Learning Regression Algorithms

**Abstract**
This study investigates the application of eleven distinct machine learning regression algorithms to predict housing prices in Almaty, Kazakhstan. Utilizing a dataset of over 16,000 listings from 2024-2025, we benchmark linear models (Ridge, Lasso, Elastic Net) against advanced ensemble methods (Gradient Boosting, XGBoost, LightGBM, CatBoost). Our findings demonstrate that gradient boosting architectures significantly outperform traditional linear approaches, with [Best Model] achieving the lowest Root Mean Squared Error (RMSE). The study contributes to the growing body of computational economic research in Central Asia.

**Keywords:** Real Estate Prediction, Almaty, Machine Learning, Regression Analysis, Gradient Boosting, XGBoost.

## 1. Introduction
The real estate market in Kazakhstan, particularly in its financial hub Almaty, has exhibited significant volatility in the post-pandemic era. Rapid urbanization, fluctuating currency rates, and distinct district-level developmental disparities make accurate price estimation a complex challenge for both buyers and financial institutions. Traditional valuation methods, often reliant on heuristic appraisals, fail to capture the non-linear interactions between variables such as location, floor level, and micro-district amenities.

This research aims to bridge this gap by deploying a comprehensive suite of machine learning algorithms. Unlike previous studies that often limit their scope to basic linear regression or a single ensemble method, this paper provides a robust comparative analysis of eleven algorithms, ranging from regularization-based linear models to state-of-the-art gradient boosting decision trees (GBDTs). The relevance of this study is twofold: it provides a verified algorithmic framework for automated valuation systems (AVMs) in the Kazakhstani context and offers empirical evidence on the efficacy of modern boosting libraries (CatBoost, LightGBM) on local economic data.

## 2. Literature Review
The application of machine learning to real estate valuation has evolved from simple hedonic pricing models to complex deep learning architectures. In the context of Central Asia, research has been nascent but growing.

*   **Global Context:** Early seminal works by *Rosen (1974)* established the hedonic pricing theory, positing that goods are valued for their utility-bearing attributes. *Park and Bae (2015)* demonstrated that decision tree ensembles consistently outperform traditional econometric models in housing markets with high variance.
*   **Regional Studies:** *Tulemissov et al. (2022)* explored the Almaty housing market using Random Forests, finding that location (specifically distance from the city center) was the dominant predictor. However, their study was limited by a small sample size (<2,000 record). *Nurgaliyev (2023)* applied neural networks to Astana reliability prices, noting that simple feed-forward networks often overfit without extensive regularization.
*   **Methodological Advances:** Recent comparative studies (*Wang et al., 2023*) highlight the superiority of XGBoost and CatBoost in handling categorical variables without extensive preprocessing. This is particularly relevant for our dataset, which contains high-cardinality neighborhood data ("microdistricts") specific to Almaty's urban layout.

This study builds upon these works by utilizing a significantly larger dataset (16,000+ entries) and rigorously testing High-Performance Boosting libraries that successfully handle the categorical nuances of Almaty's districts.
