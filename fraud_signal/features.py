"""Point-in-time features, shared by training and inference."""
import numpy as np
import pandas as pd

WINDOWS = (30, 90, 180, 365)
CATEGORIES = ["purpose", "employment_type"]
BASE_NUMERIC = ["income_monthly", "amount_requested", "monthly_obligations", "credit_score", "term_months"]
NUMERIC = BASE_NUMERIC + ["payment_to_income", "principal_to_annual_income", "application_month", "application_weekday"]
for days in WINDOWS:
    NUMERIC += [f"tx_count_{days}", f"tx_total_{days}", f"tx_mean_{days}",
                f"merchant_diversity_{days}", f"flagged_category_count_{days}"]
FEATURES = NUMERIC + CATEGORIES
FLAGGED_CATEGORIES = {"cash_transfer", "gaming"}


def _require(frame, columns, name):
    missing = set(columns) - set(frame.columns)
    if missing: raise ValueError(f"{name} missing columns: {sorted(missing)}")


def build_features(applications, transactions):
    _require(applications, ["application_id", "customer_id", "submitted_at"] + BASE_NUMERIC + CATEGORIES, "applications")
    _require(transactions, ["transaction_id", "customer_id", "occurred_at", "amount", "merchant"], "transactions")
    if applications.empty or applications.application_id.isna().any() or applications.application_id.duplicated().any():
        raise ValueError("Applications need nonempty unique IDs")
    if applications.customer_id.isna().any() or transactions.customer_id.isna().any():
        raise ValueError("Customer IDs cannot be missing")
    if transactions.transaction_id.isna().any() or transactions.transaction_id.duplicated().any():
        raise ValueError("Transaction IDs must be unique and nonmissing")
    apps, tx = applications.copy(), transactions.copy()
    apps["submitted_at"] = pd.to_datetime(apps.submitted_at, utc=True, errors="raise")
    tx["occurred_at"] = pd.to_datetime(tx.occurred_at, utc=True, errors="raise")
    if apps.submitted_at.isna().any() or tx.occurred_at.isna().any(): raise ValueError("Missing timestamp")
    for col in BASE_NUMERIC:
        apps[col] = pd.to_numeric(apps[col], errors="raise")
        if np.isinf(apps[col].to_numpy(dtype=float)).any(): raise ValueError("Infinite application value")
    tx["amount"] = pd.to_numeric(tx.amount, errors="raise")
    if not np.isfinite(tx.amount.to_numpy(dtype=float)).all() or (tx.amount < 0).any():
        raise ValueError("Transaction amounts must be finite and nonnegative")
    if (apps.income_monthly.dropna() <= 0).any(): raise ValueError("Income must be positive or missing")
    groups = {key: group.sort_values("occurred_at") for key, group in tx.groupby("customer_id")}
    rows = []
    for app in apps.itertuples(index=False):
        row = {col: getattr(app, col) for col in BASE_NUMERIC + CATEGORIES}
        row["payment_to_income"] = app.monthly_obligations / app.income_monthly
        row["principal_to_annual_income"] = app.amount_requested / (12 * app.income_monthly)
        row["application_month"] = app.submitted_at.month
        row["application_weekday"] = app.submitted_at.day_of_week
        history = groups.get(app.customer_id)
        for days in WINDOWS:
            if history is None:
                amounts, merchants = np.array([]), pd.Series([], dtype=object)
            else:
                start = app.submitted_at - pd.Timedelta(days=days)
                window = history.loc[(history.occurred_at >= start) & (history.occurred_at < app.submitted_at)]
                amounts, merchants = window.amount.to_numpy(), window.merchant
            count = len(amounts)
            row.update({f"tx_count_{days}": count, f"tx_total_{days}": float(amounts.sum()),
                f"tx_mean_{days}": float(amounts.mean()) if count else 0.,
                f"merchant_diversity_{days}": int(merchants.nunique()),
                f"flagged_category_count_{days}": int(merchants.isin(FLAGGED_CATEGORIES).sum())})
        rows.append(row)
    return pd.DataFrame(rows, columns=FEATURES, index=applications.index)
