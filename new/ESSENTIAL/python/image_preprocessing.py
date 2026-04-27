#!/usr/bin/env python3
"""
COMPLETE IMAGE PREPROCESSING PIPELINE
Shows how to convert real images → features → KNN classification

This demonstrates:
1. Loading real medical images
2. Preprocessing and normalization
3. Feature extraction
4. KNN classification
"""

import numpy as np
from PIL import Image
from pathlib import Path


class ImagePreprocessor:
    """Converts real images to FPGA-compatible format."""
    
    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        """Load image from file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Grayscale image array (H×W)
        """
        img = Image.open(image_path)
        # Convert to grayscale if needed
        if img.mode != 'L':
            img = img.convert('L')
        return np.array(img, dtype=np.uint8)
    
    @staticmethod
    def resize_image(image: np.ndarray, target_size: tuple = (28, 28)) -> np.ndarray:
        """Resize image to target size (default 28×28).
        
        Args:
            image: Input image array
            target_size: Target dimensions (height, width)
            
        Returns:
            Resized image array
        """
        img = Image.fromarray(image)
        img = img.resize(target_size, Image.LANCZOS)
        return np.array(img, dtype=np.uint8)
    
    @staticmethod
    def normalize_image(image: np.ndarray) -> np.ndarray:
        """Normalize image to 8-bit (0-255).
        
        Args:
            image: Input image array
            
        Returns:
            Normalized 8-bit image
        """
        img_min = image.min()
        img_max = image.max()
        
        if img_max == img_min:
            # Flat image, return zeros
            return np.zeros_like(image, dtype=np.uint8)
        
        # Normalize to 0-255
        normalized = ((image - img_min) / (img_max - img_min)) * 255
        return normalized.astype(np.uint8)
    
    @staticmethod
    def preprocess(image_path: str, 
                   target_size: tuple = (28, 28),
                   normalize: bool = True) -> np.ndarray:
        """Complete preprocessing pipeline.
        
        Args:
            image_path: Path to image
            target_size: Target dimensions
            normalize: Whether to normalize pixel values
            
        Returns:
            28×28 8-bit grayscale image
        """
        # Load
        image = ImagePreprocessor.load_image(image_path)
        
        # Resize
        image = ImagePreprocessor.resize_image(image, target_size)
        
        # Normalize
        if normalize:
            image = ImagePreprocessor.normalize_image(image)
        
        return image
    
    @staticmethod
    def image_to_pixel_stream(image: np.ndarray) -> list:
        """Convert 2D image to 1D pixel stream.
        
        Args:
            image: 28×28 image array
            
        Returns:
            List of 784 pixel values
        """
        return image.flatten().tolist()
    
    @staticmethod
    def verify_image_format(image: np.ndarray) -> bool:
        """Verify image is in correct FPGA format.
        
        Args:
            image: Image to verify
            
        Returns:
            True if correct format
        """
        checks = [
            image.shape == (28, 28),           # Correct size
            image.dtype == np.uint8,           # 8-bit
            image.min() >= 0,                  # Min value >= 0
            image.max() <= 255,                # Max value <= 255
        ]
        return all(checks)


class FeatureExtractor:
    """Extract features from images (matching FPGA feature_extractor.v)."""
    
    @staticmethod
    def extract_mean(image: np.ndarray) -> int:
        """Extract mean pixel intensity.
        
        Args:
            image: 8-bit grayscale image
            
        Returns:
            Mean intensity (0-255)
        """
        mean = np.mean(image)
        return int(np.clip(mean, 0, 255))
    
    @staticmethod
    def extract_variance(image: np.ndarray) -> int:
        """Extract variance (texture metric).
        
        Args:
            image: 8-bit grayscale image
            
        Returns:
            Variance/texture metric (0-255)
        """
        variance = np.var(image)
        # Scale to 0-255 range
        variance_scaled = (variance / 255.0) * 255  # Max var ~ 255
        return int(np.clip(variance_scaled, 0, 255))
    
    @staticmethod
    def extract_features(image: np.ndarray) -> tuple:
        """Extract both features from image.
        
        Args:
            image: 28×28 8-bit grayscale image
            
        Returns:
            Tuple of (feature0, feature1)
        """
        feature0 = FeatureExtractor.extract_mean(image)
        feature1 = FeatureExtractor.extract_variance(image)
        return (feature0, feature1)


class ImageToKNNPipeline:
    """Complete pipeline from image to KNN classification."""
    
    def __init__(self):
        self.preprocessor = ImagePreprocessor()
        self.feature_extractor = FeatureExtractor()
    
    def process_image(self, image_path: str) -> dict:
        """Process image through complete pipeline.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with pipeline results
        """
        results = {}
        
        # STAGE 1: Preprocessing
        print(f"[STAGE 1] Loading and preprocessing image...")
        image = self.preprocessor.preprocess(image_path)
        
        # Verify format
        if not self.preprocessor.verify_image_format(image):
            raise ValueError("Image not in correct format!")
        
        results['preprocessed_image'] = image
        print(f"  ✓ Image: {image.shape}, dtype={image.dtype}, range=[{image.min()}-{image.max()}]")
        
        # STAGE 2: Feature Extraction
        print(f"[STAGE 2] Extracting features...")
        feature0, feature1 = self.feature_extractor.extract_features(image)
        results['feature0'] = feature0  # Mean
        results['feature1'] = feature1  # Variance
        print(f"  ✓ Feature0 (mean intensity): {feature0}")
        print(f"  ✓ Feature1 (variance/texture): {feature1}")
        
        # STAGE 3: Pixel Stream (for FPGA)
        print(f"[STAGE 3] Creating pixel stream for FPGA...")
        pixel_stream = self.preprocessor.image_to_pixel_stream(image)
        results['pixel_stream'] = pixel_stream
        print(f"  ✓ Pixel stream: {len(pixel_stream)} pixels ready for FPGA")
        
        # STAGE 4: KNN Classification (software for demonstration)
        print(f"[STAGE 4] Running KNN classification...")
        from knn_reference import KNNClassifier, DistanceMetric
        
        clf_m = KNNClassifier(k=3, metric=DistanceMetric.MANHATTAN)
        clf_e = KNNClassifier(k=3, metric=DistanceMetric.EUCLIDEAN)
        
        features = np.array([feature0, feature1], dtype=np.uint8)
        pred_m = clf_m.predict(features)
        pred_e = clf_e.predict(features)
        
        results['prediction_manhattan'] = pred_m
        results['prediction_euclidean'] = pred_e
        
        print(f"  ✓ Manhattan prediction: {pred_m} ({'MALIGNANT' if pred_m else 'BENIGN'})")
        print(f"  ✓ Euclidean prediction: {pred_e} ({'MALIGNANT' if pred_e else 'BENIGN'})")
        
        return results
    
    def process_directory(self, image_dir: str) -> list:
        """Process multiple images from a directory.
        
        Args:
            image_dir: Path to directory with images
            
        Returns:
            List of results
        """
        image_dir = Path(image_dir)
        results = []
        
        image_files = list(image_dir.glob("*.jpg")) + \
                      list(image_dir.glob("*.png")) + \
                      list(image_dir.glob("*.jpeg"))
        
        print(f"Found {len(image_files)} images\n")
        
        for img_path in image_files:
            print(f"\n{'='*70}")
            print(f"Processing: {img_path.name}")
            print(f"{'='*70}")
            
            try:
                result = self.process_image(str(img_path))
                results.append(result)
            except Exception as e:
                print(f"  ✗ Error processing {img_path.name}: {e}")
        
        return results


def demo_with_synthetic_images():
    """Demonstrate pipeline with synthetic test images."""
    print("\n" + "="*70)
    print("SYNTHETIC IMAGE DEMONSTRATION")
    print("="*70)
    
    pipeline = ImageToKNNPipeline()
    preprocessor = ImagePreprocessor()
    feature_extractor = FeatureExtractor()
    
    # Test 1: Gradient image (should look like smooth variation)
    print("\n[Test 1] Gradient Image")
    print("-" * 70)
    gradient = np.zeros((28, 28), dtype=np.float32)
    for i in range(28):
        for j in range(28):
            gradient[i, j] = ((i * 255) // 28) + ((j * 255) // 28)
            gradient[i, j] = min(255, gradient[i, j])
    gradient = gradient.astype(np.uint8)
    
    if preprocessor.verify_image_format(gradient):
        f0, f1 = feature_extractor.extract_features(gradient)
        print(f"  Feature0 (mean): {f0}")
        print(f"  Feature1 (variance): {f1}")
        print(f"  ✓ Valid FPGA input")
    
    # Test 2: Uniform image (low variance)
    print("\n[Test 2] Uniform Brightness Image")
    print("-" * 70)
    uniform = np.full((28, 28), 128, dtype=np.uint8)
    
    if preprocessor.verify_image_format(uniform):
        f0, f1 = feature_extractor.extract_features(uniform)
        print(f"  Feature0 (mean): {f0}")
        print(f"  Feature1 (variance): {f1}")
        print(f"  ✓ Valid FPGA input")
    
    # Test 3: High contrast (high variance)
    print("\n[Test 3] High Contrast Image")
    print("-" * 70)
    contrast = np.zeros((28, 28), dtype=np.uint8)
    for i in range(28):
        for j in range(28):
            contrast[i, j] = 255 if (i + j) % 2 == 0 else 0
    
    if preprocessor.verify_image_format(contrast):
        f0, f1 = feature_extractor.extract_features(contrast)
        print(f"  Feature0 (mean): {f0}")
        print(f"  Feature1 (variance): {f1}")
        print(f"  ✓ Valid FPGA input")
    
    # Test 4: Gaussian blob (medical image-like)
    print("\n[Test 4] Gaussian Blob (Medical Image-like)")
    print("-" * 70)
    x = np.linspace(-3, 3, 28)
    y = np.linspace(-3, 3, 28)
    xx, yy = np.meshgrid(x, y)
    gaussian = np.exp(-(xx**2 + yy**2) / 2)
    gaussian = (gaussian * 255).astype(np.uint8)
    
    if preprocessor.verify_image_format(gaussian):
        f0, f1 = feature_extractor.extract_features(gaussian)
        print(f"  Feature0 (mean): {f0}")
        print(f"  Feature1 (variance): {f1}")
        print(f"  ✓ Valid FPGA input")
    
    print("\n" + "="*70)
    print("All synthetic images processed successfully!")
    print("="*70)


def main():
    """Main demonstration."""
    print("\n" + "="*70)
    print("COMPLETE IMAGE-TO-KNN PREPROCESSING PIPELINE")
    print("="*70)
    print("\nThis demonstrates how real images are converted for FPGA KNN")
    print("\nPipeline:")
    print("  1. Load image (from file)")
    print("  2. Preprocess (grayscale, resize 28×28, normalize)")
    print("  3. Extract features (mean, variance)")
    print("  4. Stream to FPGA (or classify in software)")
    print("  5. Get classification result")
    
    # Run synthetic demo
    demo_with_synthetic_images()
    
    # For real medical images:
    print("\n" + "="*70)
    print("TO USE WITH REAL MEDICAL IMAGES:")
    print("="*70)
    print("""
    pipeline = ImageToKNNPipeline()
    
    # Process single image
    result = pipeline.process_image("medical_scan.jpg")
    print(f"Features: {result['feature0']}, {result['feature1']}")
    print(f"Classification: {result['prediction_manhattan']}")
    
    # Process directory of images
    results = pipeline.process_directory("./medical_images/")
    """)
    
    # Show data format
    print("\n" + "="*70)
    print("DATA FORMAT SPECIFICATION:")
    print("="*70)
    print("""
    INPUT:
    • Any image format (JPEG, PNG, etc.)
    • Any size
    • Color or grayscale
    
    PREPROCESSING:
    • Convert to grayscale
    • Resize to 28×28 pixels
    • Normalize to 8-bit (0-255)
    
    OUTPUT FOR FPGA:
    • 784 8-bit pixels
    • Flattened (row-major order)
    • Stream one pixel per clock cycle
    
    FEATURES EXTRACTED:
    • Feature0 (8-bit): Mean intensity [0-255]
    • Feature1 (8-bit): Variance/texture [0-255]
    
    FPGA PROCESSES:
    • Takes 2 features
    • Compares with 50 training samples
    • Returns: 0 (benign) or 1 (malignant)
    """)


if __name__ == "__main__":
    main()
