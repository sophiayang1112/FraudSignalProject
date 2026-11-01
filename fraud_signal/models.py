import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTENC
from lightgbm import LGBMClassifier
from .features import NUMERIC, CATEGORIES


class QuantileClipper(TransformerMixin, BaseEstimator):
    """Learn 1st/99th percentile bounds on training numeric predictors only."""
    def fit(self, x, y=None):
        self.bounds_ = np.quantile(np.asarray(x, dtype=float), [.01, .99], axis=0)
        self.n_features_in_ = np.asarray(x).shape[1]
        return self

    def transform(self, x):
        return np.clip(np.asarray(x, dtype=float), *self.bounds_)


def pipeline(estimator, seed=27, resample=True):
    numeric = SkPipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("clip", QuantileClipper()), ("scale", StandardScaler())])
    category = SkPipeline([("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))])
    prepare = ColumnTransformer([("numeric", numeric, NUMERIC), ("category", category, CATEGORIES)])
    numeric_indices = list(range(len(NUMERIC)))
    category_indices = list(range(len(NUMERIC), len(NUMERIC)+len(CATEGORIES)))
    encode = ColumnTransformer([("numeric", "passthrough", numeric_indices),
        ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), category_indices)])
    steps = [("prepare", prepare)]
    if resample:
        steps.append(("smote", SMOTENC(categorical_features=category_indices, k_neighbors=3, random_state=seed)))
    steps += [("encode", encode), ("estimator", estimator)]
    return Pipeline(steps)


def candidates(seed=27):
    return {
        "prior_baseline": pipeline(DummyClassifier(strategy="prior"), seed, resample=False),
        "logistic": pipeline(LogisticRegression(max_iter=1800, random_state=seed), seed),
        "random_forest": pipeline(RandomForestClassifier(n_estimators=140, max_depth=8,
            min_samples_leaf=8, n_jobs=2, random_state=seed), seed),
        "lightgbm": pipeline(LGBMClassifier(n_estimators=110, max_depth=4, num_leaves=12,
            learning_rate=.045, verbosity=-1, n_jobs=2, random_state=seed), seed)}


def outcome_model(seed=27):
    return pipeline(LGBMClassifier(objective="multiclass", n_estimators=100, max_depth=4,
        num_leaves=12, learning_rate=.05, n_jobs=2, verbosity=-1, random_state=seed), seed)
