import numpy as np


class PCA:
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components

        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None
        self.n_features_in_ = None

    def fit(self, X: np.ndarray) -> "PCA":
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")

        n_samples, n_features = X.shape

        if n_samples < 2:
            raise ValueError("PCA needs at least 2 samples")

        if self.n_components < 1 or self.n_components > n_features:
            raise ValueError("wrong n_components")

        self.n_features_in_ = n_features
        self.mean_ = X.mean(axis=0)

        X_centered = X - self.mean_

        cov = np.cov(X_centered, rowvar=False)

        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[indices]
        eigenvectors = eigenvectors[:, indices]

        eigenvalues = np.maximum(eigenvalues, 0)

        self.components_ = eigenvectors[:, :self.n_components].T
        self.explained_variance_ = eigenvalues[:self.n_components]

        total_variance = eigenvalues.sum()

        if total_variance == 0:
            self.explained_variance_ratio_ = np.zeros(self.n_components)
        else:
            self.explained_variance_ratio_ = (
                self.explained_variance_ / total_variance
            )

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.components_ is None:
            raise ValueError("fit first")

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")

        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has wrong number of features")

        X_centered = X - self.mean_
        return X_centered @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        if self.components_ is None:
            raise ValueError("fit first")

        X_transformed = np.asarray(X_transformed, dtype=float)

        if X_transformed.ndim != 2:
            raise ValueError("X_transformed must be a 2D array")

        if X_transformed.shape[1] != self.n_components:
            raise ValueError("X_transformed has wrong number of components")

        return X_transformed @ self.components_ + self.mean_
