# Spatial Interpolation on NDBC Data


An empirical study and comparison of Deterministic, Statistical, and ML Algorithms for the Spatial Interpolation of significant wave height values collected by buoy and sea monitoring stations  managed by the United States' [National Data Buoy Center](https://www.ndbc.noaa.gov/) (NDBC) located near costs of the Southern Atlantic regions of the United States, including those on the Gulf of Mexico and parts of the Caribbean.

![ML Spatial Interpolation](/reports/figures/mltraining_interpolations_various_partial.png)

## Techniques Studied:

- **Deterministic methods:** such as linear barycentric interpolation, Inverse Distance Weighting, and Radial Basis Function (RBF) Interpolation.
- **Statistical methods:** Kriging Interpolation
 (Gaussian Process Regression).
- **Machine Learning methods:** [LightGBM](https://lightgbm.readthedocs.io/en/latest/) and [Random Forests](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html).

## Results:

The results of the study favour the use of ML algorithms over the use of other methods when paired with a strong feature set that are able to capture the spatial distribution of the data well.

![Results metrics](/reports/figures/eval_metrics.png)





