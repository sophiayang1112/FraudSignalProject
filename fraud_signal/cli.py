import argparse
import json
import platform
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
import imblearn
import lightgbm
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, classification_report, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .simulation import simulate
from .features import build_features
from .models import candidates, outcome_model
from .evaluation import group_partitions, binary_metrics, review_threshold


def train(output="reports", customers=1400, seed=27):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    applications, transactions = simulate(customers, seed)
    features = build_features(applications, transactions)
    parts = group_partitions(applications, seed)
    train_ids, valid_ids, test_ids = [parts[key] for key in ["train", "validation", "test"]]
    y = applications.fraud_label
    for key, ids in parts.items():
        if set(y.iloc[ids]) != {0, 1}: raise ValueError(f"{key} needs both fraud classes; increase customer count")
    models = candidates(seed)
    validation = {}
    for name, model in models.items():
        model.fit(features.iloc[train_ids], y.iloc[train_ids])
        validation[name] = binary_metrics(y.iloc[valid_ids], model.predict_proba(features.iloc[valid_ids])[:, 1])
    selected_name = max([name for name in models if name != "prior_baseline"],
                        key=lambda name: validation[name]["average_precision"])
    selected = models[selected_name]
    threshold = review_threshold(selected.predict_proba(features.iloc[valid_ids])[:, 1])
    # Freeze model/threshold before accessing any holdout outcomes.
    scores = {name: model.predict_proba(features.iloc[test_ids])[:, 1] for name, model in models.items()}
    test = {name: binary_metrics(y.iloc[test_ids], value) for name, value in scores.items()}
    risk_test = binary_metrics(y.iloc[test_ids], scores[selected_name], threshold)
    status = outcome_model(seed)
    status.fit(features.iloc[train_ids], applications.case_outcome.iloc[train_ids])
    status_labels = status.predict(features.iloc[test_ids])
    status_report = classification_report(applications.case_outcome.iloc[test_ids], status_labels,
                                         output_dict=True, zero_division=0)
    customer_sets = {key: set(applications.customer_id.iloc[ids]) for key, ids in parts.items()}
    assert not customer_sets["train"] & customer_sets["validation"]
    assert not customer_sets["train"] & customer_sets["test"]
    assert not customer_sets["validation"] & customer_sets["test"]
    summary = {"data_source": "locally generated fictional applications and transactions", "seed": seed,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__, "imbalanced_learn": imblearn.__version__, "lightgbm": lightgbm.__version__},
        "customers": customers, "applications": len(applications), "transactions": len(transactions),
        "split_sizes": {key: len(ids) for key, ids in parts.items()},
        "split_fraud_rates": {key: float(y.iloc[ids].mean()) for key, ids in parts.items()},
        "customer_overlap": 0, "selection_rule": "highest validation average precision excluding baseline",
        "selected_model": selected_name, "validation_threshold": threshold,
        "target_validation_review_budget": .1,
        "actual_validation_review_rate": float(np.mean(selected.predict_proba(features.iloc[valid_ids])[:, 1] >= threshold)),
        "validation_at_0_5": validation, "holdout_at_0_5": test,
        "selected_holdout_at_review_threshold": risk_test, "synthetic_outcome_classification": status_report}
    joblib.dump({"fraud_pipeline": selected, "outcome_pipeline": status,
        "threshold": threshold, "data_source": "synthetic"}, output/"model.joblib")
    applications.iloc[test_ids[:8]].drop(columns=["fraud_label", "case_outcome"]).to_csv(output/"demo_applications.csv", index=False)
    ids = set(applications.customer_id.iloc[test_ids[:8]])
    transactions.loc[transactions.customer_id.isin(ids)].to_csv(output/"demo_transactions.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    for name, values in scores.items():
        RocCurveDisplay.from_predictions(y.iloc[test_ids], values, name=name, ax=axes[0])
        PrecisionRecallDisplay.from_predictions(y.iloc[test_ids], values, name=name, ax=axes[1])
    fig.suptitle("Synthetic fraud holdout | customers disjoint across partitions")
    fig.tight_layout(); fig.savefig(output/"fraud_evaluation.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ConfusionMatrixDisplay.from_predictions(y.iloc[test_ids], scores[selected_name] >= threshold,
        display_labels=["not fraud", "fraud"], colorbar=False, ax=axes[0])
    axes[0].set_title("Fraud model at validation review threshold")
    ConfusionMatrixDisplay.from_predictions(applications.case_outcome.iloc[test_ids], status_labels,
        colorbar=False, ax=axes[1], xticks_rotation=20)
    axes[1].set_title("Synthetic historical outcome classification")
    fig.tight_layout(); fig.savefig(output/"confusion_matrices.png", dpi=150); plt.close(fig)
    importance = permutation_importance(selected, features.iloc[valid_ids], y.iloc[valid_ids],
        scoring="average_precision", n_repeats=3, random_state=seed)
    ranking = sorted(zip(features.columns, importance.importances_mean), key=lambda pair: pair[1], reverse=True)
    summary["validation_permutation_importance"] = [{"feature": name, "average_precision_drop": float(value)} for name,value in ranking]
    fig, ax = plt.subplots(figsize=(8, 5))
    top = ranking[:12][::-1]
    ax.barh([r[0] for r in top], [r[1] for r in top], color="#357b89")
    ax.set_xlabel("Validation average-precision drop after permutation")
    ax.set_title("Predictive associations on synthetic data")
    fig.tight_layout(); fig.savefig(output/"feature_importance.png", dpi=150); plt.close(fig)
    prepared = selected.named_steps["prepare"].named_transformers_["numeric"]
    clipper = prepared.named_steps["clip"]
    summary["training_eda"] = {"missing_fraction": features.iloc[train_ids].isna().mean().to_dict(),
        "numeric_clip_bounds": clipper.bounds_.tolist()}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    features.iloc[train_ids].groupby(y.iloc[train_ids]).tx_count_30.mean().plot.bar(ax=axes[0], color="#357b89")
    axes[0].set_title("Mean prior 30-day transaction count")
    axes[0].set_xlabel("Synthetic fraud label"); axes[0].set_ylabel("Count")
    features.iloc[train_ids].groupby(y.iloc[train_ids]).payment_to_income.mean().plot.bar(ax=axes[1], color="#c68e43")
    axes[1].set_title("Mean monthly payment / income")
    axes[1].set_xlabel("Synthetic fraud label"); axes[1].set_ylabel("Ratio")
    fig.suptitle("Training-only exploratory analysis")
    fig.tight_layout(); fig.savefig(output/"training_eda.png", dpi=150); plt.close(fig)
    (output/"metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"selected_model": selected_name, "holdout": risk_test, "output": str(output)}, indent=2))
    return summary


def predict(model_path, application_path, transaction_path, output_path):
    bundle = joblib.load(model_path)  # Trusted artifacts only: joblib can execute code.
    apps, tx = pd.read_csv(application_path), pd.read_csv(transaction_path)
    features = build_features(apps, tx)
    scores = bundle["fraud_pipeline"].predict_proba(features)[:, 1]
    result = pd.DataFrame({"application_id": apps.application_id, "uncalibrated_fraud_score": scores,
        "review_flag": scores >= bundle["threshold"],
        "synthetic_outcome_prediction": bundle["outcome_pipeline"].predict(features)})
    result.to_csv(output_path, index=False)
    print(f"Saved {len(result)} fictional application scores to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Synthetic point-in-time loan fraud modeling")
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("train")
    fit.add_argument("--customers", type=int, default=1400)
    fit.add_argument("--seed", type=int, default=27)
    fit.add_argument("--output", default="reports")
    score = sub.add_parser("predict")
    score.add_argument("--model", default="reports/model.joblib")
    score.add_argument("--applications", default="reports/demo_applications.csv")
    score.add_argument("--transactions", default="reports/demo_transactions.csv")
    score.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()
    if args.command == "train":
        if args.customers < 600: parser.error("Use at least 600 customers to retain sufficient minority samples")
        train(args.output, args.customers, args.seed)
    else: predict(args.model, args.applications, args.transactions, args.output)


if __name__ == "__main__": main()
