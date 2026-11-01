import unittest
import tempfile
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from fraud_signal.features import build_features, FEATURES
from fraud_signal.simulation import simulate
from fraud_signal.models import QuantileClipper, candidates, outcome_model
from fraud_signal.evaluation import group_partitions, review_threshold


def inputs():
    apps=pd.DataFrame([dict(application_id="a",customer_id=1,submitted_at="2025-02-01",
        income_monthly=5000,amount_requested=10000,monthly_obligations=1000,
        credit_score=700,term_months=24,purpose="personal",employment_type="salaried")])
    tx=pd.DataFrame([dict(transaction_id="t1",customer_id=1,occurred_at="2025-01-02",amount=100,merchant="retail"),
        dict(transaction_id="t2",customer_id=1,occurred_at="2025-02-01",amount=999,merchant="gaming"),
        dict(transaction_id="t3",customer_id=1,occurred_at="2025-02-02",amount=999,merchant="gaming")])
    return apps,tx


class IntegrityTests(unittest.TestCase):
    def test_time_boundary_and_future_exclusion(self):
        apps,tx=inputs()
        row=build_features(apps,tx).iloc[0]
        self.assertEqual(row.tx_count_30,1)
        self.assertEqual(row.tx_total_30,100)
        self.assertEqual(row.flagged_category_count_30,0)

    def test_repeated_applications_do_not_duplicate_transactions(self):
        apps,tx=inputs()
        second=apps.copy();second.application_id="b"
        result=build_features(pd.concat([apps,second],ignore_index=True),tx)
        np.testing.assert_array_equal(result.tx_count_30,[1,1])
        np.testing.assert_array_equal(result.tx_total_30,[100,100])

    def test_no_history_and_labels_excluded(self):
        apps,tx=inputs();apps.customer_id=999;apps["fraud_label"]=1;apps["case_outcome"]="confirmed_fraud"
        features=build_features(apps,tx)
        self.assertEqual(features.iloc[0].tx_count_365,0)
        self.assertNotIn("fraud_label",features)
        self.assertNotIn("customer_id",features)
        self.assertEqual(list(features.columns),FEATURES)

    def test_duplicate_transactions_rejected(self):
        apps,tx=inputs()
        with self.assertRaises(ValueError): build_features(apps,pd.concat([tx,tx.iloc[:1]]))

    def test_bad_income_and_timestamps_rejected(self):
        apps,tx=inputs();apps.income_monthly=0
        with self.assertRaises(ValueError): build_features(apps,tx)
        apps,tx=inputs();apps.submitted_at="not-a-date"
        with self.assertRaises(ValueError): build_features(apps,tx)

    def test_customer_groups_disjoint(self):
        apps,_=simulate(120,12)
        parts=group_partitions(apps)
        groups=[set(apps.customer_id.iloc[part]) for part in parts.values()]
        self.assertFalse(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
        self.assertEqual(sum(map(len,parts.values())),len(apps))

    def test_outlier_bounds_fit_only_training(self):
        clipper=QuantileClipper().fit(np.arange(100).reshape(-1,1))
        bounds=clipper.bounds_.copy()
        result=clipper.transform([[10000]])
        self.assertLess(result[0,0],100)
        np.testing.assert_array_equal(bounds,clipper.bounds_)

    def test_models_unknown_category_and_serialization(self):
        apps,tx=simulate(180,8)
        x=build_features(apps,tx)
        test=x.iloc[:4].copy();test["purpose"]="new_purpose"
        for name,model in candidates().items():
            with self.subTest(model=name):
                model.fit(x,apps.fraud_label)
                p=model.predict_proba(test)[:,1]
                self.assertTrue(np.isfinite(p).all())
                if name!="prior_baseline": self.assertEqual(model.named_steps["smote"].n_features_in_,len(FEATURES))
        status=outcome_model().fit(x,apps.case_outcome)
        self.assertEqual(len(status.predict(test)),4)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"model.joblib";joblib.dump(status,path)
            np.testing.assert_array_equal(status.predict(test),joblib.load(path).predict(test))

    def test_review_threshold_has_explicit_tie_behavior(self):
        scores=np.arange(100)/100
        threshold=review_threshold(scores,.1)
        self.assertLessEqual(np.mean(scores>=threshold),.1)
        self.assertEqual(review_threshold(np.ones(10),.1),1)


if __name__=="__main__": unittest.main()
