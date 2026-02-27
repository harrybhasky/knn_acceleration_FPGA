"""
KNN CPU Benchmark + Dataset Preparation for FPGA Comparison
Dataset: Breast Cancer Wisconsin (2 features, binary classification)
"""

import numpy as np
import time
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# ─── 1. LOAD & PREPARE DATASET ───────────────────────────────────────────────
print("=" * 60)
print("  KNN CPU vs FPGA Benchmark — Breast Cancer Dataset")
print("=" * 60)

data = load_breast_cancer()
X, y = data.data, data.target

# Use only 2 most informative features (mean radius, mean texture)
# so the Verilog knn_classifier.v works with NO changes
FEATURES = [0, 1]  # mean radius, mean texture
X2 = X[:, FEATURES]

print(f"\n[Dataset]")
print(f"  Total samples  : {X2.shape[0]}")
print(f"  Features used  : {data.feature_names[FEATURES[0]]}, {data.feature_names[FEATURES[1]]}")
print(f"  Class 0 (malignant): {np.sum(y==0)}")
print(f"  Class 1 (benign)   : {np.sum(y==1)}")

# ─── 2. SCALE TO 8-BIT INTEGER (0–255) for FPGA compatibility ────────────────
scaler = MinMaxScaler(feature_range=(0, 255))
X_scaled = scaler.fit_transform(X2).astype(np.uint8)

# ─── 3. TRAIN/TEST SPLIT ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)
print(f"\n[Split]")
print(f"  Training samples : {len(X_train)}")
print(f"  Test samples     : {len(X_test)}")

# ─── 4. CPU KNN — SCRATCH IMPLEMENTATION (mirrors the Verilog logic) ─────────
def manhattan_distance(a, b):
    return int(np.sum(np.abs(a.astype(int) - b.astype(int))))

def knn_predict_scratch(X_train, y_train, test_point, k=3):
    distances = [(manhattan_distance(test_point, x), lbl)
                 for x, lbl in zip(X_train, y_train)]
    distances.sort(key=lambda x: x[0])
    k_nearest = distances[:k]
    votes = [lbl for _, lbl in k_nearest]
    return 1 if votes.count(1) > votes.count(0) else 0

# ─── 5. BENCHMARK CPU — multiple training sizes ───────────────────────────────
print("\n[CPU Benchmark — KNN from Scratch (mirrors Verilog logic)]")
print(f"  {'Train Size':>12} | {'Time (ms)':>12} | {'Accuracy':>10}")
print("  " + "-" * 42)

results = []
K = 3

for n_train in [10, 20, 30, 40, 50]:
    X_tr = X_train[:n_train]
    y_tr = y_train[:n_train]

    # Time over ALL test samples
    start = time.perf_counter()
    preds = [knn_predict_scratch(X_tr, y_tr, tp, k=K) for tp in X_test]
    elapsed = time.perf_counter() - start

    acc = accuracy_score(y_test, preds) * 100
    ms = elapsed * 1000
    results.append((n_train, ms, acc))
    print(f"  {n_train:>12} | {ms:>12.3f} | {acc:>9.1f}%")

# ─── 6. SKLEARN REFERENCE (for accuracy sanity check) ────────────────────────
clf = KNeighborsClassifier(n_neighbors=K, metric='manhattan')
clf.fit(X_train, y_train)
sk_acc = accuracy_score(y_test, clf.predict(X_test)) * 100
print(f"\n  sklearn KNN accuracy (reference): {sk_acc:.1f}%")

# ─── 7. EXPORT TRAINING DATA FOR VERILOG ─────────────────────────────────────
# Use first 50 training samples
N = 50
X_fpga_train = X_train[:N]
y_fpga_train = y_train[:N]

# Export Verilog initial block snippet
with open("training_data_verilog.txt", "w") as f:
    f.write("// Auto-generated training data — Breast Cancer (2 features, 8-bit)\n")
    f.write("// Feature 0: mean radius (scaled 0-255)\n")
    f.write("// Feature 1: mean texture (scaled 0-255)\n\n")
    for i in range(N):
        f1, f2 = int(X_fpga_train[i][0]), int(X_fpga_train[i][1])
        lbl = int(y_fpga_train[i])
        f.write(f"        training_data1[{i:2d}] = 8'd{f1:3d};  "
                f"training_data2[{i:2d}] = 8'd{f2:3d};  "
                f"training_label[{i:2d}] = {lbl};\n")

# Export test samples for testbench
with open("/home/claude/test_data_for_tb.txt", "w") as f:
    f.write("// Test samples for Verilog testbench\n")
    f.write("// format: test_data1, test_data2, expected_label\n\n")
    for i in range(min(20, len(X_test))):
        f1, f2 = int(X_test[i][0]), int(X_test[i][1])
        lbl = int(y_test[i])
        f.write(f"// Test {i:2d}: ({f1:3d}, {f2:3d}) -> expected class {lbl}\n")
        f.write(f"test_data = 16'h{f1:02X}{f2:02X}; expected = {lbl};\n")

# ─── 8. FPGA CLOCK CYCLE ESTIMATION ──────────────────────────────────────────
print("\n[FPGA Clock Cycle Estimation @ 100 MHz]")
print(f"  {'Train Size':>12} | {'Clock Cycles':>14} | {'FPGA Time (ns)':>16} | {'Speedup':>10}")
print("  " + "-" * 62)

CLOCK_FREQ_HZ = 100e6
CLOCK_PERIOD_NS = 10  # 10 ns per cycle @ 100 MHz

for n_train, cpu_ms, acc in results:
    # Cycle breakdown:
    #  - CALC_DISTANCE: n_train cycles (1 per sample)
    #  - SORT (bubble): ~n_train^2 cycles worst case
    #  - CLASSIFY: k cycles
    dist_cycles   = n_train
    sort_cycles   = n_train * n_train  # bubble sort worst case
    classify_cycles = K
    overhead      = 10  # FSM transitions
    total_cycles  = dist_cycles + sort_cycles + classify_cycles + overhead

    fpga_ns       = total_cycles * CLOCK_PERIOD_NS
    fpga_ms       = fpga_ns / 1e6
    speedup       = cpu_ms / fpga_ms if fpga_ms > 0 else 0

    print(f"  {n_train:>12} | {total_cycles:>14,} | {fpga_ns:>14,.0f} ns | {speedup:>9.0f}x")

print("\n[Generated Files]")
print("  training_data_verilog.txt  — paste into knn_classifier.v initial block")
print("  test_data_for_tb.txt       — use in Verilog testbench")

# ─── 9. SUMMARY ──────────────────────────────────────────────────────────────
print("\n[Summary — 50 Training Samples]")
n50_ms = results[-1][1]
n50_acc = results[-1][2]
n50_cycles = 50 + 50*50 + K + 10  # = 2563
n50_fpga_ms = (n50_cycles * CLOCK_PERIOD_NS) / 1e6
speedup50 = n50_ms / n50_fpga_ms
print(f"  CPU time         : {n50_ms:.3f} ms")
print(f"  FPGA time (est.) : {n50_fpga_ms*1000:.1f} ns ({n50_fpga_ms:.6f} ms)")
print(f"  Speedup          : ~{speedup50:.0f}x")
print(f"  Accuracy (CPU)   : {n50_acc:.1f}%")
print("=" * 60)
