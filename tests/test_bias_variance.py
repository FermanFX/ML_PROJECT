import numpy as np
import pandas as pd
import src.utils.bias_variance_helper as helper

class DummyAdaBoost:
    def __init__(self, *args, **kwargs):
        self.majority_class = 0

    def fit(self, X, y):
        values, counts = np.unique(y, return_counts=True)
        self.majority_class = values[np.argmax(counts)]
        return self

    def predict(self, X):
        return np.full(
            X.shape[0],
            self.majority_class,
            dtype=int,
        )

class DummyRandomForest:
    def __init__(self, *args, **kwargs):
        self.threshold = 0.0
    def fit(self, X, y):
        class_0_mean = X[y == 0, 0].mean()
        class_1_mean = X[y == 1, 0].mean()
        self.threshold = (class_0_mean + class_1_mean) / 2
        return self
    def predict(self, X):
        return (X[:, 0] > self.threshold).astype(int)

def test_bias_variance_experiment_end_to_end(monkeypatch):
    monkeypatch.setattr(
        helper,
        "AdaBoostClassifier",
        DummyAdaBoost,
    )

    monkeypatch.setattr(
        helper,
        "RandomForestClassifier",
        DummyRandomForest,
    )

    X_train = np.array(
        [
            [0.0],
            [0.2],
            [0.4],
            [0.6],
            [1.0],
            [1.2],
            [1.4],
            [1.6],
        ],
        dtype=float,
    )

    y_train = np.array(
        [0, 0, 0, 0, 1, 1, 1, 1],
        dtype=int,
    )

    X_test = np.array(
        [
            [0.1],
            [0.5],
            [1.1],
            [1.5],
        ],
        dtype=float,
    )

    y_test = np.array(
        [0, 0, 1, 1],
        dtype=int,
    )

    results = helper.run_bias_variance_experiment(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        n_bootstrap=10,
        n_estimators=5,
        random_state=42,
        rf_n_jobs=1,
    )

    assert isinstance(results, pd.DataFrame)

    assert list(results["model"]) == [
        "AdaBoost",
        "Random Forest",
    ]

    expected_columns = {
        "model",
        "bias_squared",
        "variance",
        "total_error",
        "mean_accuracy",
        "accuracy_std",
    }

    assert set(results.columns) == expected_columns

    numeric_columns = [
        "bias_squared",
        "variance",
        "total_error",
        "mean_accuracy",
        "accuracy_std",
    ]

    assert np.isfinite(
        results[numeric_columns].to_numpy()
    ).all()

    assert (
        results["bias_squared"] >= 0
    ).all()

    assert (
        results["variance"] >= 0
    ).all()

    assert (
        results["total_error"] >= 0
    ).all()

    assert (
        results["mean_accuracy"].between(0, 1)
    ).all()

    assert (
        results["accuracy_std"] >= 0
    ).all()

    decomposition_sum = (
        results["bias_squared"]
        + results["variance"]
    )

    assert np.allclose(
        results["total_error"],
        decomposition_sum,
        atol=1e-12,
    )

    bootstrap_X, bootstrap_y = (
        helper.generate_bootstrap_sample(
            X_train,
            y_train,
            random_state=42,
        )
    )

    assert bootstrap_X.shape == X_train.shape
    assert bootstrap_y.shape == y_train.shape

    repeated_bootstrap_X, repeated_bootstrap_y = (
        helper.generate_bootstrap_sample(
            X_train,
            y_train,
            random_state=42,
        )
    )

    assert np.array_equal(
        bootstrap_X,
        repeated_bootstrap_X,
    )

    assert np.array_equal(
        bootstrap_y,
        repeated_bootstrap_y,
    )

    predictions = np.array(
        [
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [0, 0, 0, 1],
        ]
    )

    decomposition = helper.calculate_bias_variance(
        predictions,
        y_test,
    )

    assert np.isclose(
        decomposition["total_error"],
        decomposition["bias_squared"]
        + decomposition["variance"],
    )

    assert np.isclose(
        decomposition["decomposition_difference"],
        0.0,
        atol=1e-12,
    )