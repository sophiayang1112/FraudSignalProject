# Implementation provenance and method references

A user-provided example was reviewed to identify general techniques: time-window transaction summaries, financial ratios, imbalanced classification, SMOTE, logistic regression, random forests, gradient boosting, and a multiclass outcome task. The reference archive did not contain an explicit license file.

No reference source files, prose, diagrams, datasets, output figures, or trained models are redistributed. This repository is a new implementation with a new module layout, synthetic data generator, feature schema, validation policy, tests, documentation, and measured outputs. It is not a renamed copy, fork, or reproduction of the reference's reported results. The original author's work is not claimed as this repository's authorship.

Changes to the general method include customer-disjoint partitions, training-only transformations, direct per-customer aggregation, explicit timestamp boundary tests, categorical-aware SMOTENC, and a shared training/inference feature builder.

Public method documentation:

- [SMOTENC and mixed categorical/numeric oversampling](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTENC.html)
- [Original SMOTE paper, Chawla et al.](https://doi.org/10.1613/jair.953)
- [Logistic regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
- [Random forests](https://scikit-learn.org/stable/modules/ensemble.html#forest)
- [LightGBM](https://lightgbm.readthedocs.io/en/stable/)
- [GroupShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html)

These external libraries remain subject to their own licenses. Method names and citations do not imply endorsement or permission to republish any third-party project.
