# Mercedes-Benz baseline evaluation

This follow-up predicts the labeled test-bench-time target using seven categorical configuration columns, `X0` through `X6`. It compares a training-fold mean with ridge regression after training-fold one-hot encoding. The same five shuffled folds (seed 42) are used for both models over all 4,209 labeled rows.

| Model | Mean fold RMSE | Mean fold MAE | Mean fold R² |
|---|---:|---:|---:|
| Mean baseline | 12.6727 | 10.0922 | -0.0030 |
| Categorical ridge | 8.3867 | 5.4427 | 0.5612 |

Run `python evaluate_categorical.py data/train.csv` from the repository root. Recorded with pandas 2.2.3 and scikit-learn 1.8.0. Metrics use the target's source units. The scope is a cross-validated categorical experiment using the labeled training file. The competition test file has no target labels and was not scored.
