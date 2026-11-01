# FraudSignalLab

An independent **AI/ML, Finance, and Risk Management** portfolio project: combine loan applications with past transaction behavior, handle class imbalance, compare fraud classifiers, and model fictional historical case outcomes.

**All customers, transactions, and outcomes are synthetic.** This repository contains newly written implementations, documentation, and generated figures. It contains no copied reference code, private customer records, assignment materials, or external datasets.

## Methods

- Strictly pre-application transaction windows: **30, 90, 180, and 365 days**.
- Counts, total/mean amounts, merchant diversity, and illustrative flagged-category counts.
- Affordability ratios, calendar features, missing-value imputation, 1st/99th percentile clipping, scaling, and one-hot encoding.
- Categorical-aware **SMOTENC** oversampling, a SMOTE-family method.
- **Logistic regression, random forest, LightGBM**, and an unsampled prior baseline.
- A separate multiclass **LightGBM** classifier for synthetic historical case outcomes.
- Customer-disjoint train/validation/test partitions, ROC-AUC, average precision, precision/recall/F1, confusion matrices, and validation permutation importance.

## Run

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -e .
python -m fraud_signal.cli train --customers 1400 --seed 27 --output reports
python -m fraud_signal.cli predict
python -m unittest discover -s tests -v
```

Training generates the actual synthetic applications and transaction context used by the prediction demo, a fitted model bundle, metrics, and charts. Generated CSV/model files are excluded from Git; reproduce them locally. Only load model artifacts you trust.

The saved pipeline uses the **same feature builder for training and inference**, including historical windows for each fictional application rather than replacing all transaction features with zero.

## Results and plots

[Executed results](reports/RESULTS.md) · [Machine-readable metrics](reports/metrics.json) · [Methodology](docs/METHODOLOGY.md) · [Implementation provenance](docs/PROVENANCE.md)

![Synthetic fraud evaluation](reports/fraud_evaluation.png)

![Holdout confusion matrices](reports/confusion_matrices.png)

![Validation feature importance](reports/feature_importance.png)

![Training exploratory analysis](reports/training_eda.png)

## Integrity decisions

Transactions are aggregated directly from each customer's history, avoiding a many-to-many application/transaction merge. The lower time-window boundary is included, and the application timestamp and future transactions are excluded. A transaction at exactly the decision time is unavailable by this policy.

All applications for a customer stay in one split. Imputation, clipping, scaling, encoders, and SMOTENC learn from training data only. Model selection uses validation average precision. A validation-only threshold targets a 10% review budget; ties may exceed that budget and future review rates can differ.

SMOTENC operates on numeric and ordinal categorical inputs; one-hot encoding follows oversampling. This avoids interpolating one-hot dummy values. Resampling is bypassed during prediction. Numeric counts can still become fractional synthetic training values; this is an experimental approximation.

## Interpretation and limits

Fraud scores are **uncalibrated classifier outputs**, not validated real-world fraud probabilities. Oversampling changes class prevalence, so probability calibration would require additional independent data and evaluation. Brier scores are reported to make probability error visible, not to establish calibration.

The secondary outcome task uses a disclosed toy policy with `approved`, `declined`, and `confirmed_fraud` labels. It classifies fictional historical records and does not recommend real approval decisions. Its apparent accuracy is strongly influenced by the simulated labeling rule.

This project does not provide a real-time transaction detector, lending engine, production API, causal fraud explanation, fairness certification, or regulatory-compliance claim. Merchant flags are illustrative and do not identify real fraudulent merchants. Real use requires lawful data, outcome definitions, temporal validation, calibration, fairness assessment, governance, security, and monitoring.
