# Model notebook outline

- Load: read `data/fishing_trips.csv` and set `random_state=355`; outputs the dataframe and configuration.
- Check: inspect shape, dtypes, missingness, duplicate IDs, base rates, and category counts; outputs validation summaries.
- Split: define the 13 features and excluded columns and split seasons 1-2 from season 3; outputs train/test frames and rates.
- Baselines: calculate majority-class and training-season month-rule predictions; outputs baseline scores.
- Models and comparison: fit logistic regression and gradient boosting pipelines using season 1 validation; outputs a comparison table and chosen model.
- Costs and threshold: set the two planning costs and derive the threshold; outputs the threshold and cost function.
- Test evaluation: evaluate season 3 once, compare dollar costs, sweep thresholds, and calculate segment metrics; outputs test metrics and weakest segment.
- Segment analysis: identify performance by trip type, species, and month; outputs per-segment AUC and accuracy.
- Save artifacts: save `model.joblib` and `model_meta.json`; outputs deployable artifacts.
- Verification: run leakage, split, traceability, and planted-effect checks; outputs `verification_results.json`.
