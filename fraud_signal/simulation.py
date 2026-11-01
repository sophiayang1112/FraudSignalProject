import numpy as np
import pandas as pd
from .features import build_features


def simulate(customers=1400, seed=27):
    """Fictional customers, two applications each, and timestamped transactions."""
    if customers < 100: raise ValueError("Use at least 100 customers")
    rng = np.random.default_rng(seed)
    apps, transactions = [], []
    anchor = pd.Timestamp("2025-01-01", tz="UTC")
    for customer in range(customers):
        income = float(np.clip(rng.lognormal(np.log(5200), .5), 1500, 30000))
        score = float(np.clip(rng.normal(680, 65), 400, 850))
        behavior = rng.beta(1, 5)
        for _ in range(rng.poisson(40) + 5):
            merchant = rng.choice(["retail", "utilities", "travel", "cash_transfer", "gaming"],
                p=[.45*(1-behavior), .4*(1-behavior), .15*(1-behavior), .6*behavior, .4*behavior])
            transactions.append({"transaction_id": f"T{len(transactions):07d}",
                "customer_id": customer, "occurred_at": anchor + pd.Timedelta(days=int(rng.integers(-400, 300))),
                "amount": float(rng.lognormal(4.7 + behavior, .9)), "merchant": merchant})
        for round_id in range(2):
            apps.append({"application_id": f"A{customer:05d}_{round_id}", "customer_id": customer,
                "submitted_at": anchor + pd.Timedelta(days=int(rng.integers(0, 180)) + round_id*35),
                "income_monthly": income, "amount_requested": income*float(rng.uniform(2, 35)),
                "monthly_obligations": income*float(rng.uniform(.05, .65)), "credit_score": score,
                "term_months": int(rng.choice([12, 24, 36, 60])),
                "purpose": str(rng.choice(["personal", "consolidation", "renovation"])),
                "employment_type": str(rng.choice(["salaried", "self_employed", "contract"]))})
    applications, tx = pd.DataFrame(apps), pd.DataFrame(transactions)
    features = build_features(applications, tx)
    eta = (-4 + (660-features.credit_score)/90 + 2*features.payment_to_income
        + .035*features.flagged_category_count_90 + .025*features.principal_to_annual_income
        + .35*(features.employment_type == "contract"))
    applications["fraud_label"] = rng.binomial(1, 1/(1+np.exp(-eta)))
    declined = (features.credit_score < 620) | (features.payment_to_income > .53)
    applications["case_outcome"] = np.where(applications.fraud_label == 1, "confirmed_fraud",
        np.where(declined, "declined", "approved"))
    for column in ["credit_score", "monthly_obligations", "employment_type"]:
        applications.loc[rng.random(len(applications)) < .02, column] = np.nan
    return applications, tx
