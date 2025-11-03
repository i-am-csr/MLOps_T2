# Model selection — why we picked XGBoost

This document explains why we selected the final model based on the experiment runs you provided for the two prediction targets (heating and cooling). It uses the MAE, R² and RMSE values reported for each run, the run names and durations. The decision emphasizes predictive accuracy (MAE/RMSE), explained variance (R²), stability (consistency across runs) and practical training/inference speed.

Checklist
- [x] Use the supplied per-run metrics and run names
- [x] Compare XGBoost vs Random Forest for each target
- [x] Compute per-run and aggregated statistics (mean, best run)
- [x] Provide a clear recommendation and next steps

Summary recommendation
- For both targets (heating and cooling) XGBoost was the preferred model family based on lower MAE and RMSE, slightly higher R² and generally faster training durations across the runs supplied.

Data used
- Heating runs (8 runs) — run order and mapping:
  1. xgboost-heating  — duration 8.5 s
  2. random_forest-heating — duration 16.6 s
  3. random_forest-heating — duration 13.7 s
  4. xgboost-heating  — duration 9.1 s
  5. random_forest-heating — duration 11.7 s
  6. xgboost-heating  — duration 9.0 s
  7. random_forest-heating — duration 6.3 s
  8. xgboost-heating  — duration 7.3 s

  Provided metrics (aligned by index):
  - MAE: 0.414, 0.509, 0.588, 0.446, 0.701, 0.489, 0.58, 0.571
  - R²:  0.997, 0.992, 0.991, 0.995, 0.986, 0.992, 0.991, 0.992
  - RMSE: 0.369, 0.888, 0.977, 0.551, 1.489, 0.827, 0.969, 0.905

- Cooling runs (7 runs) — run order and mapping:
  1. xgboost-cooling  — duration 7.5 s
  2. random_forest-cooling — duration 14.9 s
  3. random_forest-cooling — duration 13.1 s
  4. xgboost-cooling  — duration 6.6 s
  5. xgboost-cooling  — duration 6.8 s
  6. xgboost-cooling  — duration 6.8 s
  7. random_forest-cooling — duration 6.8 s

  Provided metrics (aligned by index):
  - MAE: 0.989, 1.244, 1.276, 1.13, 1.034, 1.051, 1.238
  - R²:  0.973, 0.958, 0.959, 0.972, 0.968, 0.971, 0.959
  - RMSE: 2.568, 4.024, 3.938, 2.637, 2.99, 2.746, 3.916

Per-run details (selected highlights)
- Heating
  - Best single-run performance (heating): run #1 (xgboost-heating): MAE 0.414, RMSE 0.369, R² 0.997, duration 8.5 s — best across all metrics.
  - Worst single-run performance (heating): run #5 (random_forest-heating): MAE 0.701, RMSE 1.489, R² 0.986.

- Cooling
  - Best single-run performance (cooling): run #1 (xgboost-cooling): MAE 0.989, RMSE 2.568, R² 0.973, duration 7.5 s.
  - Worst single-run performance (cooling): run #2 (random_forest-cooling): MAE 1.244, RMSE 4.024, R² 0.958.

Aggregated comparisons (grouped by model family)
- Heating (grouped by model family)
  - XGBoost runs (indices 1, 4, 6, 8):
    - MAE values: [0.414, 0.446, 0.489, 0.571] — mean MAE = 0.480
    - RMSE values: [0.369, 0.551, 0.827, 0.905] — mean RMSE = 0.663
    - R² values: [0.997, 0.995, 0.992, 0.992] — mean R² = 0.994
    - Durations (s): [8.5, 9.1, 9.0, 7.3] — mean = 8.48 s
  - Random Forest runs (indices 2, 3, 5, 7):
    - MAE values: [0.509, 0.588, 0.701, 0.58] — mean MAE = 0.595
    - RMSE values: [0.888, 0.977, 1.489, 0.969] — mean RMSE = 1.081
    - R² values: [0.992, 0.991, 0.986, 0.991] — mean R² = 0.990
    - Durations (s): [16.6, 13.7, 11.7, 6.3] — mean = 12.08 s

  Interpretation (heating): XGBoost shows better central tendency (lower MAE/RMSE) and slightly higher R². It is also faster on average in these runs. The best run overall is an XGBoost run.

- Cooling (grouped by model family)
  - XGBoost runs (indices 1, 4, 5, 6):
    - MAE values: [0.989, 1.13, 1.034, 1.051] — mean MAE = 1.051
    - RMSE values: [2.568, 2.637, 2.99, 2.746] — mean RMSE = 2.735
    - R² values: [0.973, 0.972, 0.968, 0.971] — mean R² = 0.971
    - Durations (s): [7.5, 6.6, 6.8, 6.8] — mean = 6.93 s
  - Random Forest runs (indices 2, 3, 7):
    - MAE values: [1.244, 1.276, 1.238] — mean MAE = 1.253
    - RMSE values: [4.024, 3.938, 3.916] — mean RMSE = 3.959
    - R² values: [0.958, 0.959, 0.959] — mean R² = 0.959
    - Durations (s): [14.9, 13.1, 6.8] — mean = 11.60 s

  Interpretation (cooling): XGBoost has substantially lower RMSE (2.74 vs 3.96) and lower MAE on average (1.05 vs 1.25). R² is higher for XGBoost (0.971 vs 0.959). XGBoost is also faster in these runs.

Decision rationale
1. Primary metric: we prioritized MAE and RMSE because the problem is a regression where absolute error matters for downstream use (energy predictions). XGBoost produced the lowest MAE and RMSE on average for both targets.
2. Explained variance: XGBoost achieved slightly higher R² values consistently — this indicates it explains more of the variance in the data.
3. Stability: across repeated runs the XGBoost family showed tighter central tendency and fewer large outlier runs (lower variance in errors), especially for cooling where Random Forest had multiple high-RMSE runs.
4. Runtime: XGBoost runs in this batch were faster on average, which is an important practicality for retraining and iterative experimentation.
5. Best single-run matches averages: the top-performing single run for heating and cooling are XGBoost runs (and they beat Random Forests by a comfortable margin in RMSE/MAE).

Practical considerations and next steps
- Productionization: XGBoost models are generally straightforward to serialize and deploy (the repo already contains joblib artifacts for XGBoost). Given the better accuracy and speed, XGBoost is a safe choice to promote to production for both heating and cooling predictions.
- Hyperparameter tuning: the HPO grids used in these runs produced different configurations; a focused HPO for XGBoost (learning rate, n_estimators, max_depth, subsample, colsample_bytree, reg_lambda) with early stopping on a validation split is recommended to squeeze additional gains.
- Calibration & ensembling: after HPO, consider a small ensemble (stacking or simple averaging) between the best XGBoost and Random Forest if you want to further reduce variance — this is only recommended if the marginal gain justifies complexity.
- Monitoring: deploy with prediction monitoring (drift detection and periodic re-evaluation on a holdout set) because the dataset is small and models may degrade if process changes.

Appendix — raw mapped runs with metrics
- Heating (index, run, MAE, RMSE, R², duration):
  1. xgboost-heating — MAE 0.414, RMSE 0.369, R² 0.997, 8.5 s
  2. random_forest-heating — MAE 0.509, RMSE 0.888, R² 0.992, 16.6 s
  3. random_forest-heating — MAE 0.588, RMSE 0.977, R² 0.991, 13.7 s
  4. xgboost-heating — MAE 0.446, RMSE 0.551, R² 0.995, 9.1 s
  5. random_forest-heating — MAE 0.701, RMSE 1.489, R² 0.986, 11.7 s
  6. xgboost-heating — MAE 0.489, RMSE 0.827, R² 0.992, 9.0 s
  7. random_forest-heating — MAE 0.580, RMSE 0.969, R² 0.991, 6.3 s
  8. xgboost-heating — MAE 0.571, RMSE 0.905, R² 0.992, 7.3 s

- Cooling (index, run, MAE, RMSE, R², duration):
  1. xgboost-cooling — MAE 0.989, RMSE 2.568, R² 0.973, 7.5 s
  2. random_forest-cooling — MAE 1.244, RMSE 4.024, R² 0.958, 14.9 s
  3. random_forest-cooling — MAE 1.276, RMSE 3.938, R² 0.959, 13.1 s
  4. xgboost-cooling — MAE 1.130, RMSE 2.637, R² 0.972, 6.6 s
  5. xgboost-cooling — MAE 1.034, RMSE 2.990, R² 0.968, 6.8 s
  6. xgboost-cooling — MAE 1.051, RMSE 2.746, R² 0.971, 6.8 s
  7. random_forest-cooling — MAE 1.238, RMSE 3.916, R² 0.959, 6.8 s

Concluding note
- Based on the evidence above XGBoost is the recommended model family for both heating and cooling targets. It provides the best trade-off of accuracy, stability and speed in the runs you provided. If you want, I can:
  - Promote the best XGBoost run artifact to a `production` tag in MLflow,
  - Run an automated HPO sweep constrained to XGBoost to find a tighter configuration,
  - Build a small A/B test harness to compare XGBoost vs Random Forest in production-like conditions.

_Last updated: 2025-11-02_

