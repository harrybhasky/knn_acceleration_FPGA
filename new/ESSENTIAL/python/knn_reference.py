#!/usr/bin/env python3
"""
KNN Classifier - Software Reference Implementation
Provides baseline for FPGA performance comparison.
"""

import numpy as np
from enum import Enum
import time

class DistanceMetric(Enum):
    MANHATTAN = 0
    EUCLIDEAN = 1

class KNNClassifier:
    """KNN classifier for breast cancer dataset."""
    
    # Breast cancer dataset (50 samples, 2 features)
    TRAINING_DATA = np.array([
        [55, 73],    [50, 169],   [43, 74],    [69, 78],    [52, 28],
        [9, 136],    [57, 42],    [155, 90],   [91, 111],   [66, 26],
        [181, 105],  [100, 134],  [159, 84],   [88, 61],    [69, 31],
        [78, 95],    [97, 87],    [96, 86],    [70, 71],    [44, 46],
        [117, 90],   [61, 32],    [67, 57],    [66, 61],    [91, 36],
        [66, 51],    [33, 159],   [69, 68],    [30, 23],    [69, 170],
        [79, 40],    [59, 74],    [133, 104],  [122, 57],   [51, 31],
        [72, 132],   [38, 44],    [93, 36],    [66, 57],    [75, 89],
        [31, 53],    [78, 74],    [71, 165],   [220, 131],  [92, 51],
        [51, 28],    [62, 71],    [124, 135],  [123, 127],  [89, 86],
    ], dtype=np.uint8)
    
    TRAINING_LABELS = np.array([
        1, 1, 1, 1, 1, 1, 1, 0, 1, 1,  # 0-9
        0, 0, 0, 1, 1, 0, 0, 1, 1, 1,  # 10-19
        0, 1, 1, 1, 1, 0, 1, 1, 1, 1,  # 20-29
        1, 1, 0, 0, 1, 1, 1, 1, 1, 1,  # 30-39
        1, 1, 1, 0, 1, 1, 1, 0, 0, 0,  # 40-49
    ], dtype=np.uint8)
    
    def __init__(self, k=3, metric=DistanceMetric.MANHATTAN):
        """Initialize KNN classifier.
        
        Args:
            k: Number of nearest neighbors
            metric: Distance metric (MANHATTAN or EUCLIDEAN)
        """
        self.k = k
        self.metric = metric
        self.feature_cache = {}
    
    @staticmethod
    def extract_features(image):
        """Extract features from 28x28 image.
        
        Args:
            image: Flattened array of 784 pixels (0-255)
        
        Returns:
            Tuple of (mean, variance)
        """
        image = np.array(image, dtype=np.float32)
        mean = np.mean(image)
        variance = np.var(image)
        
        # Scale to 0-255 range for compatibility
        mean = np.clip(mean, 0, 255).astype(np.uint8)
        variance = np.clip(variance, 0, 255).astype(np.uint8)
        
        return mean, variance
    
    def distance(self, point1, point2):
        """Calculate distance between two points.
        
        Args:
            point1: First feature vector [f0, f1]
            point2: Second feature vector [f0, f1]
        
        Returns:
            Distance value
        """
        diff = np.abs(point1 - point2)
        
        if self.metric == DistanceMetric.MANHATTAN:
            return np.sum(diff)  # L1 distance
        elif self.metric == DistanceMetric.EUCLIDEAN:
            return np.sum(diff ** 2)  # L2 squared distance (no sqrt)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def predict(self, features):
        """Predict class for given features.
        
        Args:
            features: Feature vector [f0, f1]
        
        Returns:
            Predicted class (0 or 1)
        """
        # Calculate distances to all training samples
        distances = []
        for i, train_sample in enumerate(self.TRAINING_DATA):
            dist = self.distance(features, train_sample)
            distances.append((dist, i))
        
        # Sort by distance and get k nearest
        distances.sort()
        k_nearest = distances[:self.k]
        
        # Majority voting
        votes = [self.TRAINING_LABELS[idx] for _, idx in k_nearest]
        return 1 if sum(votes) > len(votes) / 2 else 0
    
    def evaluate(self, test_features):
        """Evaluate classifier on test features.
        
        Args:
            test_features: List of [f0, f1] feature vectors
        
        Returns:
            List of predictions
        """
        predictions = []
        for features in test_features:
            pred = self.predict(features)
            predictions.append(pred)
        return predictions

def generate_test_images():
    """Generate test images and extract features.
    
    Returns:
        List of (image, mean, variance) tuples
    """
    images = []
    
    # Test 1: Gradient image
    image1 = np.zeros(784, dtype=np.uint8)
    for i in range(28):
        for j in range(28):
            val = ((i * 255) // 28) + ((j * 255) // 28)
            image1[i * 28 + j] = min(255, val)
    images.append(("Gradient", image1))
    
    # Test 2: Uniform brightness
    image2 = np.full(784, 128, dtype=np.uint8)
    images.append(("Uniform", image2))
    
    # Test 3: High contrast
    image3 = np.zeros(784, dtype=np.uint8)
    for i in range(784):
        image3[i] = 255 if i % 2 == 0 else 0
    images.append(("HighContrast", image3))
    
    return images

def benchmark_knn(num_runs=100):
    """Benchmark KNN classification performance.
    
    Args:
        num_runs: Number of classification runs
    """
    print("\n" + "="*60)
    print("KNN Classifier Performance Benchmark")
    print("="*60)
    
    # Create classifiers
    clf_manhattan = KNNClassifier(k=3, metric=DistanceMetric.MANHATTAN)
    clf_euclidean = KNNClassifier(k=3, metric=DistanceMetric.EUCLIDEAN)
    
    # Generate test data
    test_images = generate_test_images()
    
    print(f"\nTesting {len(test_images)} images, {num_runs} runs each\n")
    
    for img_name, image in test_images:
        mean, var = KNNClassifier.extract_features(image)
        features = np.array([mean, var], dtype=np.uint8)
        
        print(f"Image: {img_name}")
        print(f"  Extracted Features: mean={mean}, variance={var}")
        
        # Manhattan distance benchmark
        start_time = time.time()
        for _ in range(num_runs):
            pred_m = clf_manhattan.predict(features)
        elapsed_m = time.time() - start_time
        
        # Euclidean distance benchmark
        start_time = time.time()
        for _ in range(num_runs):
            pred_e = clf_euclidean.predict(features)
        elapsed_e = time.time() - start_time
        
        print(f"  Predictions: Manhattan={pred_m}, Euclidean={pred_e}")
        print(f"  Time (Manhattan): {elapsed_m*1000:.2f}ms ({num_runs} runs)")
        print(f"  Time (Euclidean): {elapsed_e*1000:.2f}ms ({num_runs} runs)")
        print(f"  Time per classification:")
        print(f"    Manhattan: {(elapsed_m*1e6)/num_runs:.1f}µs")
        print(f"    Euclidean: {(elapsed_e*1e6)/num_runs:.1f}µs")
        print()
    
    print("="*60)
    print("Note: FPGA timing should be compared in clock cycles at")
    print("its operating frequency to this microsecond baseline.")
    print("="*60)

if __name__ == "__main__":
    # Run benchmark
    benchmark_knn(num_runs=1000)
    
    # Test consistency between metrics
    print("\nConsistency Check:")
    clf_m = KNNClassifier(metric=DistanceMetric.MANHATTAN)
    clf_e = KNNClassifier(metric=DistanceMetric.EUCLIDEAN)
    
    test_images = generate_test_images()
    for img_name, image in test_images:
        mean, var = KNNClassifier.extract_features(image)
        features = np.array([mean, var], dtype=np.uint8)
        
        pred_m = clf_m.predict(features)
        pred_e = clf_e.predict(features)
        
        status = "✓" if pred_m == pred_e else "✗"
        print(f"{status} {img_name}: Manhattan={pred_m}, Euclidean={pred_e}")
