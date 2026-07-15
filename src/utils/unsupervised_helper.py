from __future__ import annotations
from typing import Any, Iterable
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from src.unsupervised.dbscan import DBSCAN
from src.unsupervised.kmeans import KMeans
from src.unsupervised.pca import PCA

__all__ = [
    "pca_task",
    "kmeans_task",
    "dbscan_task",
    "find_best_dbscan_eps",
    "plot_dbscan_best_eps",
]

RANDOM_STATE = 42

def _prepare_unsupervised_data(
    X: Any,
    y: Any,
    max_points: int,
    *,
    drop_all_nan_columns: bool = True,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:

    if max_points <= 0:
        raise ValueError("max_points müsbət tam ədəd olmalıdır.")

    X_df = pd.DataFrame(X).copy()

    if isinstance(y, pd.DataFrame):
        if y.shape[1] == 0:
            raise ValueError("y DataFrame heç bir sütun saxlamır.")

        y_series = y.iloc[:, 0].copy()
    elif isinstance(y, pd.Series):
        y_series = y.copy()
    else:
        y_series = pd.Series(np.asarray(y).ravel())

    y_series = pd.Series(
        np.asarray(y_series).ravel()
    ).reset_index(drop=True)

    X_df = X_df.reset_index(drop=True)

    if len(X_df) != len(y_series):
        raise ValueError(
            "X və y eyni sayda nümunəyə sahib olmalıdır. "
            f"X: {len(X_df)}, y: {len(y_series)}"
        )

    if len(X_df) == 0:
        raise ValueError("Dataset boşdur.")

    X_df = X_df.apply(
        pd.to_numeric,
        errors="coerce",
    )

    if drop_all_nan_columns:
        X_df = X_df.dropna(
            axis=1,
            how="all",
        )

    if X_df.shape[1] == 0:
        raise ValueError(
            "Numeric çevirmədən sonra istifadə edilə bilən feature qalmadı."
        )

    column_medians = X_df.median(
        numeric_only=True
    )

    X_df = X_df.fillna(column_medians)
    remaining_nan_columns = X_df.columns[
        X_df.isna().any()
    ].tolist()

    if remaining_nan_columns:
        X_df = X_df.drop(
            columns=remaining_nan_columns
        )

    if X_df.shape[1] == 0:
        raise ValueError(
            "NaN təmizlənməsindən sonra heç bir feature qalmadı."
        )

    if len(X_df) > max_points:
        rng = np.random.default_rng(RANDOM_STATE)

        sample_indices = rng.choice(
            len(X_df),
            size=max_points,
            replace=False,
        )

        sample_indices = np.sort(sample_indices)

        X_df = (
            X_df.iloc[sample_indices]
            .reset_index(drop=True)
        )

        y_series = (
            y_series.iloc[sample_indices]
            .reset_index(drop=True)
        )

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X_df
    )

    X_scaled = np.asarray(
        X_scaled,
        dtype=np.float64,
    )

    if not np.all(np.isfinite(X_scaled)):
        raise ValueError(
            "Scaling-dən sonra X daxilində NaN və ya sonsuz qiymət yarandı."
        )

    return X_df, y_series, X_scaled


def _calculate_k_distances(
    X_scaled: np.ndarray,
    min_samples: int,
) -> np.ndarray:
    
    X_scaled = np.asarray(
        X_scaled,
        dtype=np.float64,
    )

    if X_scaled.ndim != 2:
        raise ValueError(
            "X_scaled ikiölçülü array olmalıdır."
        )

    n_samples = X_scaled.shape[0]

    if n_samples == 0:
        raise ValueError(
            "K-distance boş dataset üçün hesablana bilməz."
        )

    if min_samples < 2:
        raise ValueError(
            "min_samples ən azı 2 olmalıdır."
        )

    if min_samples > n_samples:
        raise ValueError(
            "min_samples dataset-dəki nümunə sayından böyük ola bilməz. "
            f"min_samples={min_samples}, samples={n_samples}"
        )

    # Pairwise Euclidean distance matrix.
    squared_distances = np.sum(
        (
            X_scaled[:, np.newaxis, :]
            - X_scaled[np.newaxis, :, :]
        )
        ** 2,
        axis=2,
    )

    # Floating-point səbəbindən çox kiçik mənfi qiymət yarana bilər.
    squared_distances = np.maximum(
        squared_distances,
        0.0,
    )

    distances = np.sqrt(
        squared_distances
    )

    neighbor_index = min_samples - 1

    kth_distances = np.partition(
        distances,
        kth=neighbor_index,
        axis=1,
    )[:, neighbor_index]

    return np.sort(kth_distances)


def _get_dbscan_statistics(
    labels: np.ndarray,
) -> tuple[int, float, dict[int, int]]:
   
    labels = np.asarray(
        labels,
        dtype=int,
    ).ravel()

    if labels.size == 0:
        raise ValueError(
            "DBSCAN boş label array qaytardı."
        )

    unique_labels, counts = np.unique(
        labels,
        return_counts=True,
    )

    n_clusters = int(
        np.sum(unique_labels != -1)
    )

    noise_fraction = float(
        np.mean(labels == -1)
    )

    distribution = {
        int(label): int(count)
        for label, count in zip(
            unique_labels,
            counts,
        )
    }

    return (
        n_clusters,
        noise_fraction,
        distribution,
    )


def _plot_k_distance(
    dataset_name: str,
    kth_distances_sorted: np.ndarray,
    min_samples: int,
    *,
    chosen_eps: float | None = None,
) -> None:
    """
    K-distance qrafikini çəkir.
    """

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        kth_distances_sorted
    )

    if chosen_eps is not None:
        plt.axhline(
            y=chosen_eps,
            linestyle="--",
            label=f"chosen eps = {chosen_eps:.4f}",
        )

    plt.xlabel(
        "Points sorted by distance"
    )

    plt.ylabel(
        f"Distance to {min_samples}-th nearest neighbor"
    )

    plt.title(
        f"{dataset_name} - k-distance Plot"
    )

    plt.grid(True)

    if chosen_eps is not None:
        plt.legend()

    plt.tight_layout()
    plt.show()


def pca_task(
    dataset_name: str,
    X: Any,
    y: Any,
    n_clusters: int,
    dbscan_eps: float,
    dbscan_min_samples: int = 5,
    max_points: int = 5000,
) -> None:

    if n_clusters <= 0:
        raise ValueError(
            "n_clusters müsbət olmalıdır."
        )

    if dbscan_eps <= 0:
        raise ValueError(
            "dbscan_eps müsbət olmalıdır."
        )

    _, y_series, X_scaled = _prepare_unsupervised_data(
        X,
        y,
        max_points=max_points,
        drop_all_nan_columns=True,
    )

    n_components = min(
        X_scaled.shape[0],
        X_scaled.shape[1],
    )

    if n_components < 2:
        raise ValueError(
            "PCA vizuallaşdırması üçün ən azı 2 komponent mümkün olmalıdır."
        )

    pca = PCA(
        n_components=n_components
    )

    X_pca = pca.fit_transform(
        X_scaled
    )

    explained_variance_ratio = np.asarray(
        pca.explained_variance_ratio_,
        dtype=np.float64,
    )

    cumulative_variance = np.cumsum(
        explained_variance_ratio
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        range(
            1,
            len(cumulative_variance) + 1,
        ),
        cumulative_variance,
        marker="o",
    )

    plt.axhline(
        y=0.90,
        linestyle="--",
        label="90% variance",
    )

    plt.xlabel(
        "Number of Principal Components"
    )

    plt.ylabel(
        "Cumulative Explained Variance"
    )

    plt.title(
        f"{dataset_name} - Scree Plot"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    true_labels, _ = pd.factorize(
        y_series
    )

    # Clustering original standardized feature space-də aparılır.
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=RANDOM_STATE,
    )

    kmeans_labels = np.asarray(
        kmeans.fit_predict(X_scaled),
        dtype=int,
    )

    dbscan = DBSCAN(
        eps=float(dbscan_eps),
        min_samples=dbscan_min_samples,
    )

    dbscan_labels = np.asarray(
        dbscan.fit_predict(X_scaled),
        dtype=int,
    )

    # PCA yalnız vizuallaşdırma üçün istifadə olunur.
    X_2d = X_pca[:, :2]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5),
    )

    titles = [
        "True Labels",
        "K-Means Labels",
        "DBSCAN Labels",
    ]

    labels_list = [
        true_labels,
        kmeans_labels,
        dbscan_labels,
    ]

    for ax, title, labels in zip(
        axes,
        titles,
        labels_list,
    ):
        scatter = ax.scatter(
            X_2d[:, 0],
            X_2d[:, 1],
            c=labels,
            cmap="tab10",
            s=15,
            alpha=0.75,
        )

        ax.set_title(
            f"{dataset_name} - {title}"
        )

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(alpha=0.25)

        fig.colorbar(
            scatter,
            ax=ax,
        )

    plt.tight_layout()
    plt.show()

    first_pc_variance = float(
        explained_variance_ratio[0]
    )

    second_pc_variance = float(
        explained_variance_ratio[1]
    )

    first_two_variance = float(
        explained_variance_ratio[:2].sum()
    )

    reaches_90 = np.flatnonzero(
        cumulative_variance >= 0.90
    )

    if reaches_90.size > 0:
        n_90 = int(
            reaches_90[0] + 1
        )
    else:
        n_90 = int(
            len(cumulative_variance)
        )

    print(dataset_name)
    print("PC1 variance:", first_pc_variance)
    print("PC2 variance:", second_pc_variance)
    print("First 2 PCs total:", first_two_variance)
    print("Number of PCs for >=90% variance:", n_90)

    return None

def kmeans_task(
    dataset_name: str,
    X: Any,
    y: Any,
    max_points: int = 5000,
    n_init: int = 10,
) -> pd.DataFrame:
    
    if n_init <= 0:
        raise ValueError(
            "n_init müsbət olmalıdır."
        )

    _, y_series, X_scaled = _prepare_unsupervised_data(
        X,
        y,
        max_points=max_points,
    )

    inertias: list[float] = []
    aris: list[float] = []

    max_k = min(
        10,
        len(X_scaled),
    )

    for k in range(
        1,
        max_k + 1,
    ):
        best_inertia = np.inf
        best_labels: np.ndarray | None = None

        for restart in range(n_init):
            model = KMeans(
                n_clusters=k,
                max_iter=300,
                tol=1e-4,
                random_state=RANDOM_STATE + restart,
            )

            model.fit(
                X_scaled
            )

            if model.inertia_ is None:
                raise RuntimeError(
                    "KMeans inertia_ hesablanmadı."
                )

            current_inertia = float(
                model.inertia_
            )

            if current_inertia < best_inertia:
                best_inertia = current_inertia

                if model.labels_ is None:
                    raise RuntimeError(
                        "KMeans labels_ hesablanmadı."
                    )

                best_labels = np.asarray(
                    model.labels_,
                    dtype=int,
                ).copy()

        if best_labels is None:
            raise RuntimeError(
                f"k={k} üçün KMeans nəticəsi alınmadı."
            )

        ari = float(
            adjusted_rand_score(
                y_series,
                best_labels,
            )
        )

        inertias.append(
            best_inertia
        )

        aris.append(
            ari
        )

    results = pd.DataFrame(
        {
            "k": range(
                1,
                max_k + 1,
            ),
            "inertia": inertias,
            "ARI": aris,
        }
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        results["k"],
        results["inertia"],
        marker="o",
    )

    plt.xlabel(
        "Number of clusters (k)"
    )

    plt.ylabel(
        "Inertia"
    )

    plt.title(
        f"{dataset_name} - Elbow Method"
    )

    plt.xticks(
        range(
            1,
            max_k + 1,
        )
    )

    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(dataset_name)
    display(results)

    best_row = results.loc[
        results["ARI"].idxmax()
    ]

    print("Best ARI:", float(best_row["ARI"]))
    print("Best k by ARI:", int(best_row["k"]))

    return results


def dbscan_task(
    dataset_name: str,
    X: Any,
    y: Any,
    eps: float,
    min_samples: int = 5,
    max_points: int = 3000,
) -> dict[str, Any]:
    """
    Verilmiş eps və min_samples ilə DBSCAN experimenti aparır.
    """

    if eps <= 0:
        raise ValueError(
            "eps müsbət olmalıdır."
        )

    _, y_series, X_scaled = _prepare_unsupervised_data(
        X,
        y,
        max_points=max_points,
    )

    kth_distances_sorted = _calculate_k_distances(
        X_scaled,
        min_samples=min_samples,
    )

    _plot_k_distance(
        dataset_name=dataset_name,
        kth_distances_sorted=kth_distances_sorted,
        min_samples=min_samples,
        chosen_eps=float(eps),
    )

    dbscan = DBSCAN(
        eps=float(eps),
        min_samples=min_samples,
    )

    labels = np.asarray(
        dbscan.fit_predict(X_scaled),
        dtype=int,
    )

    ari = float(
        adjusted_rand_score(
            y_series,
            labels,
        )
    )

    (
        n_clusters,
        noise_fraction,
        cluster_distribution,
    ) = _get_dbscan_statistics(
        labels
    )

    print(dataset_name)
    print("eps:", float(eps))
    print("min_samples:", min_samples)
    print("DBSCAN clusters:", n_clusters)
    print("ARI:", ari)
    print("Noise fraction:", noise_fraction)
    print("Cluster distribution:", cluster_distribution)

    return {
        "dataset": dataset_name,
        "eps": float(eps),
        "min_samples": int(min_samples),
        "ARI": ari,
        "noise_fraction": noise_fraction,
        "n_clusters": n_clusters,
        "cluster_distribution": cluster_distribution,
        "labels": labels,
    }


def find_best_dbscan_eps(
    dataset_name: str,
    X: Any,
    y: Any,
    min_samples: int = 5,
    max_points: int = 3000,
    eps_values: Iterable[float] | np.ndarray | None = None,
    min_clusters: int = 2,
    max_noise_fraction: float = 0.80,
    number_of_dense_values: int = 60,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    DBSCAN üçün müxtəlif eps qiymətlərini yoxlayır.

    eps_values verilmədikdə:
    - k-distance paylanmasının 10-97 percentilləri hesablanır;
    - minimum və maksimum namizədlər arasında əlavə sıx interval yaradılır;
    - beləliklə 2.5 kimi percentillər arasında qalan kritik qiymətlər də yoxlanılır.

    Best result seçim qaydası:
    1. Əvvəl min_clusters və max_noise_fraction şərtlərini ödəyən nəticələr seçilir.
    2. Onların içində maksimum ARI seçilir.
    3. Heç biri şərtləri ödəmirsə, bütün nəticələr içində maksimum ARI seçilir.
    """

    if min_clusters < 1:
        raise ValueError(
            "min_clusters ən azı 1 olmalıdır."
        )

    if not 0.0 <= max_noise_fraction <= 1.0:
        raise ValueError(
            "max_noise_fraction 0 və 1 arasında olmalıdır."
        )

    if number_of_dense_values < 2:
        raise ValueError(
            "number_of_dense_values ən azı 2 olmalıdır."
        )

    _, y_series, X_scaled = _prepare_unsupervised_data(
        X,
        y,
        max_points=max_points,
    )

    kth_distances_sorted = _calculate_k_distances(
        X_scaled,
        min_samples=min_samples,
    )

    _plot_k_distance(
        dataset_name=dataset_name,
        kth_distances_sorted=kth_distances_sorted,
        min_samples=min_samples,
    )

    if eps_values is None:
        percentiles = np.array(
            [
                10,
                15,
                20,
                25,
                30,
                35,
                40,
                45,
                50,
                55,
                60,
                65,
                70,
                75,
                80,
                85,
                90,
                95,
                97,
            ],
            dtype=float,
        )

        percentile_eps_values = np.percentile(
            kth_distances_sorted,
            percentiles,
        )

        percentile_eps_values = percentile_eps_values[
            np.isfinite(percentile_eps_values)
            & (percentile_eps_values > 0)
        ]

        if percentile_eps_values.size == 0:
            raise ValueError(
                f"{dataset_name}: etibarlı eps namizədi yaradıla bilmədi."
            )

        lower_bound = float(
            percentile_eps_values.min()
        )

        upper_bound = float(
            percentile_eps_values.max()
        )

        # Minimum percentildən də bir qədər aşağı qiymətlər yoxlanılır.
        expanded_lower_bound = max(
            lower_bound * 0.70,
            np.finfo(float).eps,
        )

        dense_eps_values = np.linspace(
            expanded_lower_bound,
            upper_bound,
            num=number_of_dense_values,
        )

        eps_array = np.concatenate(
            [
                percentile_eps_values,
                dense_eps_values,
            ]
        )
    else:
        eps_array = np.asarray(
            list(eps_values),
            dtype=float,
        )

    eps_array = eps_array[
        np.isfinite(eps_array)
        & (eps_array > 0)
    ]

    eps_array = np.unique(
        np.round(
            eps_array,
            decimals=4,
        )
    )

    if eps_array.size == 0:
        raise ValueError(
            "Yoxlamaq üçün heç bir etibarlı eps qiyməti verilməyib."
        )

    rows: list[dict[str, Any]] = []
    distribution_by_eps: dict[float, dict[int, int]] = {}

    for eps in eps_array:
        eps_value = float(eps)

        dbscan = DBSCAN(
            eps=eps_value,
            min_samples=min_samples,
        )

        labels = np.asarray(
            dbscan.fit_predict(X_scaled),
            dtype=int,
        )

        (
            n_clusters,
            noise_fraction,
            cluster_distribution,
        ) = _get_dbscan_statistics(labels)

        ari = float(
            adjusted_rand_score(
                y_series,
                labels,
            )
        )

        rows.append(
            {
                "eps": eps_value,
                "clusters": n_clusters,
                "ARI": ari,
                "noise_fraction": noise_fraction,
            }
        )

        distribution_by_eps[eps_value] = cluster_distribution

    results = (
        pd.DataFrame(
            rows,
            columns=[
                "eps",
                "clusters",
                "ARI",
                "noise_fraction",
            ],
        )
        .sort_values(by="eps")
        .reset_index(drop=True)
    )

    display_columns = [
        "eps",
        "clusters",
        "ARI",
        "noise_fraction",
    ]

    display(
        results[display_columns]
    )

    valid_results = results[
        (results["clusters"] >= min_clusters)
        & (
            results["noise_fraction"]
            <= max_noise_fraction
        )
    ]

    if valid_results.empty:
        print(
            "\nWarning: "
            f"ən azı {min_clusters} cluster və "
            f"noise <= {max_noise_fraction:.2f} "
            "şərtinə uyğun nəticə tapılmadı."
        )

        print(
            "Bütün nəticələr arasından maksimum ARI seçilir."
        )

        best_index = results["ARI"].idxmax()
    else:
        best_index = valid_results["ARI"].idxmax()

    best_row = results.loc[
        best_index
    ].copy()

    print("\n" + dataset_name)
    print("Best eps:", float(best_row["eps"]))
    print("Best ARI:", float(best_row["ARI"]))
    print("Clusters:", int(best_row["clusters"]))
    print(
        "Noise fraction:",
        float(best_row["noise_fraction"]),
    )
    best_eps_value = float(best_row["eps"])
    best_cluster_distribution = distribution_by_eps[best_eps_value]

    print(
        "Cluster distribution:",
        best_cluster_distribution,
    )

    return results, best_row


def plot_dbscan_best_eps(
    dataset_name: str,
    X: Any,
    y: Any,
    best_eps: float,
    min_samples: int = 5,
    max_points: int = 3000,
) -> None:
    """
    Best eps ilə DBSCAN-i full standardized feature space-də işlədir,
    nəticəni PCA-nın ilk iki komponentində vizuallaşdırır.
    """

    if best_eps <= 0:
        raise ValueError(
            "best_eps müsbət olmalıdır."
        )

    _, y_series, X_scaled = _prepare_unsupervised_data(
        X,
        y,
        max_points=max_points,
    )

    dbscan = DBSCAN(
        eps=float(best_eps),
        min_samples=min_samples,
    )

    dbscan_labels = np.asarray(
        dbscan.fit_predict(X_scaled),
        dtype=int,
    )

    pca = PCA(
        n_components=2
    )

    X_2d = pca.fit_transform(
        X_scaled
    )

    plt.figure(
        figsize=(7, 5)
    )

    scatter = plt.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=dbscan_labels,
        cmap="tab10",
        s=18,
        alpha=0.75,
    )

    plt.xlabel("PC1")
    plt.ylabel("PC2")

    plt.title(
        f"{dataset_name} - DBSCAN Clusters "
        f"(eps={float(best_eps):.4f})"
    )

    plt.colorbar(
        scatter,
        label="DBSCAN label",
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()
    plt.show()

    ari = float(
        adjusted_rand_score(
            y_series,
            dbscan_labels,
        )
    )

    (
        n_clusters,
        noise_fraction,
        cluster_distribution,
    ) = _get_dbscan_statistics(
        dbscan_labels
    )

    print(dataset_name)
    print("eps:", float(best_eps))
    print("min_samples:", min_samples)
    print("clusters:", n_clusters)
    print("ARI:", ari)
    print("noise fraction:", noise_fraction)
    print("cluster distribution:", cluster_distribution)

    return None
        
