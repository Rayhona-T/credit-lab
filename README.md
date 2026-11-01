# CREDIT LAB

CREDIT LAB is an interactive machine-learning lab built around the UCI Default
of Credit Card Clients dataset. It explores how classical classifiers predict
whether a card client defaulted on payment the following month.

## Explore

- **Command Center** summarizes the dataset and compares models on a held-out test set.
- **Model Arena** compares K-Nearest Neighbors, a Decision Tree, and Random Forest;
  inspect metrics, confusion matrices, and global tree feature importance.
- **Data Observatory** visualizes observed default rates, repayment status, credit
  limits, bills, and payments.
- **Credit Simulator** runs hypothetical profiles through the selected trained
  model and compares changed inputs in the What-If Lab.

Predictions and probabilities are model estimates, not official credit scores or
lending decisions. The simulator presets are hypothetical profiles; their displayed
predictions are calculated at runtime.

## Dataset and experiment

The dataset is the [UCI Default of Credit Card Clients dataset](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients).
The project includes a local CSV copy. `ID` is excluded from model features.
If the CSV and extracted spreadsheet are unavailable, the data loader can download
the official UCI archive.

Evaluation uses a reproducible stratified 80/20 split (random seed 42), with default
(`1`) as the positive class. KNN uses a `StandardScaler` fitted on training data.
The supplied `credit_model.joblib` Decision Tree is loaded as-is; KNN and Random
Forest artifacts are in `models/`. Accuracy, precision, recall, F1, ROC-AUC, and
confusion matrices are computed on the same held-out test split. F1 is the default
ranking metric; the model ranking can change with the selected metric.

The classes are imbalanced, the data is historical and observational, and demographic
features are present. These results do not establish causality, fairness, or
performance on current applicants. This is an educational experiment, not a
production credit-risk system.

## Install and run

Use Python 3.12 for the pinned scikit-learn version and supplied model artifacts.
The original Decision Tree was serialized with scikit-learn 1.6.1; loading it with
a different version can produce an `InconsistentVersionWarning`, so keep the pinned
version when reproducing the provided evaluation.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

If the `streamlit` command is not on `PATH`, run the same entry point with:

```powershell
python -m streamlit run app.py
```

The comparison models are trained and cached if their artifacts are missing. The
original Decision Tree file is not overwritten.

## Deployment

The app uses `app.py` as its Streamlit entry point and can be deployed to a
Streamlit-compatible host. Include the dependency file and required model artifacts
(`credit_model.joblib`, `models/knn_pipeline.joblib`, and
`models/random_forest.joblib`) in the deployment. Do not publish private credentials.
