import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler


FEATURES = [0, 1]  # mean radius, mean texture
K = 3
TRAIN_SAMPLES = 50


def manhattan(a, b):
    return int(np.sum(np.abs(a.astype(int) - b.astype(int))))


def euclidean_sq(a, b):
    d = a.astype(int) - b.astype(int)
    return int(np.sum(d * d))


def chebyshev(a, b):
    return int(np.max(np.abs(a.astype(int) - b.astype(int))))


def knn_predict(x_train, y_train, test_point, distance_fn, k=3):
    distances = [(distance_fn(test_point, x), lbl) for x, lbl in zip(x_train, y_train)]
    distances.sort(key=lambda item: item[0])
    votes = [label for _, label in distances[:k]]
    return 1 if votes.count(1) > votes.count(0) else 0


def main():
    data = load_breast_cancer()
    x = data.data[:, FEATURES]
    y = data.target

    scaler = MinMaxScaler(feature_range=(0, 255))
    x_scaled = scaler.fit_transform(x).astype(np.uint8)

    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled, y, test_size=0.3, random_state=42, stratify=y
    )

    x_train50 = x_train[:TRAIN_SAMPLES]
    y_train50 = y_train[:TRAIN_SAMPLES]

    metric_map = [
        ("Manhattan (L1)", manhattan),
        ("Euclidean squared (L2^2)", euclidean_sq),
        ("Chebyshev (L-infinity)", chebyshev),
    ]

    print("=" * 74)
    print("KNN Distance Comparison (same dataset split, 2 features, K=3, 50 training)")
    print("=" * 74)
    print(f"{'Metric':30s} | {'Accuracy on 171 tests':>21s}")
    print("-" * 74)

    results = []
    for name, fn in metric_map:
        preds = [knn_predict(x_train50, y_train50, point, fn, k=K) for point in x_test]
        acc = accuracy_score(y_test, preds) * 100
        results.append((name, acc))
        print(f"{name:30s} | {acc:>20.1f}%")

    best_name, best_acc = max(results, key=lambda item: item[1])
    print("-" * 74)
    print(f"Best metric on this setup: {best_name} ({best_acc:.1f}%)")

    print("\n" + "=" * 74)
    print("sklearn check (same split, full 398-training, 2 features)")
    print("=" * 74)
    print(f"{'Metric':30s} | {'Accuracy on 171 tests':>21s}")
    print("-" * 74)

    for metric_name, kwargs in [
        ("manhattan", {}),
        ("euclidean", {}),
        ("chebyshev", {}),
        ("minkowski (p=3)", {"metric": "minkowski", "p": 3}),
    ]:
        if metric_name == "minkowski (p=3)":
            clf = KNeighborsClassifier(n_neighbors=K, **kwargs)
        else:
            clf = KNeighborsClassifier(n_neighbors=K, metric=metric_name)
        clf.fit(x_train, y_train)
        preds = clf.predict(x_test)
        acc = accuracy_score(y_test, preds) * 100
        print(f"{metric_name:30s} | {acc:>20.1f}%")


if __name__ == "__main__":
    main()
