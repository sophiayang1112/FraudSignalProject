# Data and methodology

## Simulation and schema

The simulator creates 1,400 fictional customers, two applications each, and timestamped transactions. Customer IDs are local integers and are never predictors. There are no names, addresses, protected demographics, bank accounts, or external data sources.

Application inputs: `application_id`, `customer_id`, `submitted_at` (UTC), `income_monthly`, `amount_requested`, `monthly_obligations`, `credit_score`, `term_months`, `purpose`, and `employment_type`. Transaction inputs: unique `transaction_id`, `customer_id`, `occurred_at` (UTC), nonnegative `amount`, and `merchant`.

Missing numerical/categorical application values are allowed; income must be positive when present. IDs and timestamps cannot be missing. Duplicate application or transaction IDs are rejected. All timestamps are normalized to UTC. A window is [submitted_at - days, submitted_at). Transactions at or after application are never used. The demo assumes transaction event time equals availability time; delayed arrival is not modeled.

## Targets

Fraud labels are sampled from a disclosed logistic mechanism using credit score, affordability, prior flagged-category activity, requested-principal ratio, and employment category. This mechanism is a fictional statistical relationship, not an empirical fraud finding. Labels are generated before injecting missingness. Default/nonpayment is not the fraud target.

For a separate toy historical outcome target, simulated fraud becomes confirmed_fraud; otherwise low credit scores or high affordability ratios trigger declined and the rest become approved. Both targets and all IDs are excluded from features. Because the tasks are structurally related by design, multiclass accuracy is not evidence of independent real-world underwriting capability.

## Splits and transformations

GroupShuffleSplit reserves 20% of customers for test and 20% for validation, leaving 60% for training. Both applications for a customer stay together. Group splitting is not stratified; realized prevalence is saved per split, and training aborts if binary classes are missing. This estimates transfer to unseen fictional customers, not performance over time. Random simulated dates do not justify an out-of-time validation claim.

Numeric training predictors pass through median imputation, training-only 1st/99th percentile bounds, and standardization. Categories are imputed and ordinal encoded, SMOTENC resamples training classes with three neighbors, and one-hot encoding follows. Unknown inference categories are handled without refitting. The unsampled baseline preserves training prevalence.

Logistic regression, random forest, and LightGBM use fixed declared settings. Highest validation average precision selects the model. The selected model and 90th-percentile validation-score threshold are fixed before holdout evaluation; there is no test-based refit. Full holdout comparisons are disclosed for transparency. Threshold ties may exceed the target validation budget. A manual-review flag is not a fraud verdict.

## Metrics and interpretation

Report ROC-AUC, average precision, precision, recall, F1, review fraction, uncalibrated Brier error, and confusion matrices. Binary matrices use actual rows [0,1] and predicted columns [0,1]. Multiclass labels are shown in the figure and report. Low prevalence makes average precision and operational review capacity important; no algorithm is assumed to win in advance.

Permutation importance uses validation average precision. Correlation among nested transaction windows can spread importance across variables, and importance is not causal. No protected attributes are generated; this does not establish absence of proxy bias. No economic-loss reduction, production latency, confidence interval, or real-world performance is claimed.
