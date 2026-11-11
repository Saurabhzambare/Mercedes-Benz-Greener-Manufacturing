"""Compare a mean baseline and ridge on the labeled Mercedes training file."""

import argparse

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

FEATURES = ["X0", "X1", "X2", "X3", "X4", "X5", "X6"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", default="data/train.csv")
    data = pd.read_csv(parser.parse_args().csv)
    folds = KFold(n_splits=5, shuffle=True, random_state=42)
    models = {
        "Mean baseline": DummyRegressor(),
        "Categorical ridge": make_pipeline(
            OneHotEncoder(handle_unknown="ignore"), Ridge(alpha=1.0)
        ),
    }
    for name, model in models.items():
        scores = cross_validate(
            model, data[FEATURES], data["y"], cv=folds,
            scoring=("r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"),
        )
        print(
            f"{name}: R²={scores['test_r2'].mean():.4f}, "
            f"RMSE={-scores['test_neg_root_mean_squared_error'].mean():.4f}, "
            f"MAE={-scores['test_neg_mean_absolute_error'].mean():.4f}"
        )
