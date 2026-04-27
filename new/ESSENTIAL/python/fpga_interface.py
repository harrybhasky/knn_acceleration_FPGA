#!/usr/bin/env python3
"""
Zynq PS-Side Interface to FPGA KNN Classifier
Provides memory-mapped access to AXI-lite slave wrapper

Register Layout:
  0x00: CONTROL_REG      (RW) - bit[0]=start, bit[1]=reset, bit[7:4]=metric
  0x04: STATUS_REG       (R)  - bit[0]=busy, bit[1]=pixel_ready
  0x08: RESULT_REG       (R)  - bit[0]=classification (0=benign, 1=malignant)
  0x0C: PIXEL_DATA_REG   (W)  - bit[7:0]=pixel value (auto-increment counter)
  0x10: PIXEL_COUNT_REG  (R)  - bit[9:0]=current pixel index

Usage:
  fpga = MemoryMappedFPGA(base_addr=0x43c00000)  # Vivado default for AXI
  result = fpga.classify_image(image_array, metric='manhattan')
  print(f"Classification: {result['diagnosis']}")
"""

import ctypes
import os
import time
import numpy as np
from typing import Dict, Optional, Tuple


class MemoryMappedFPGA:
    """
    Interface to memory-mapped FPGA registers via /dev/mem
    """
    
    # Register Offsets (in bytes)
    REG_CONTROL     = 0x00
    REG_STATUS      = 0x04
    REG_RESULT      = 0x08
    REG_PIXEL_DATA  = 0x0C
    REG_PIXEL_COUNT = 0x10
    
    # Page size for memory mapping
    PAGE_SIZE = 0x1000  # 4KB
    PAGE_MASK = ~(PAGE_SIZE - 1)
    
    # Status flags
    STATUS_BUSY = 0x01
    STATUS_PIXEL_READY = 0x02
    
    # Control flags
    CTRL_START = 0x01
    CTRL_RESET = 0x02
    CTRL_METRIC_MANHATTAN = 0x0
    CTRL_METRIC_EUCLIDEAN = 0x10
    
    def __init__(self, base_addr: int = 0x43c00000, verbose: bool = False):
        """
        Initialize memory-mapped FPGA interface
        
        Args:
            base_addr: AXI slave base address (default: Zynq AXI offset)
            verbose: Print debug information
        
        Raises:
            OSError: If /dev/mem cannot be accessed (not running as root)
        """
        self.base_addr = base_addr
        self.verbose = verbose
        self.mem_fd = None
        self.mem_ptr = None
        self.page_aligned_addr = None
        
        try:
            # Open /dev/mem for direct memory access
            self.mem_fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
            if self.verbose:
                print(f"[INFO] Opened /dev/mem (fd={self.mem_fd})")
        except PermissionError:
            raise PermissionError(
                "Cannot access /dev/mem. Must run as root: sudo python3 script.py"
            )
        except OSError as e:
            raise OSError(f"Failed to open /dev/mem: {e}")
        
        self._map_memory()
    
    def _map_memory(self):
        """Map FPGA registers into user space"""
        # Align base address to page boundary
        self.page_aligned_addr = self.base_addr & self.PAGE_MASK
        page_offset = self.base_addr - self.page_aligned_addr
        
        try:
            # Map one page of memory
            self.mem_ptr = ctypes.pythonapi.mmap(
                None,
                self.PAGE_SIZE,
                ctypes.c_int(3),  # PROT_READ | PROT_WRITE
                ctypes.c_int(1),  # MAP_SHARED
                self.mem_fd,
                self.page_aligned_addr
            )
            if self.verbose:
                print(f"[INFO] Memory mapped at {self.mem_ptr:#x}")
        except Exception as e:
            raise RuntimeError(f"Failed to map memory: {e}")
    
    def _read_reg(self, offset: int) -> int:
        """Read 32-bit register"""
        addr = ctypes.cast(self.mem_ptr, ctypes.POINTER(ctypes.c_uint32))
        # Offset into mapped region from page-aligned address
        reg_offset = (self.base_addr - self.page_aligned_addr + offset) // 4
        value = addr[reg_offset]
        return value
    
    def _write_reg(self, offset: int, value: int):
        """Write 32-bit register"""
        addr = ctypes.cast(self.mem_ptr, ctypes.POINTER(ctypes.c_uint32))
        reg_offset = (self.base_addr - self.page_aligned_addr + offset) // 4
        addr[reg_offset] = ctypes.c_uint32(value).value
    
    def reset(self):
        """Reset FPGA module"""
        ctrl = self._read_reg(self.REG_CONTROL)
        self._write_reg(self.REG_CONTROL, ctrl | self.CTRL_RESET)
        time.sleep(0.01)  # Wait for reset
        self._write_reg(self.REG_CONTROL, ctrl & ~self.CTRL_RESET)
        if self.verbose:
            print("[INFO] FPGA reset complete")
    
    def get_status(self) -> Dict[str, bool]:
        """Read status register"""
        status = self._read_reg(self.REG_STATUS)
        return {
            'busy': bool(status & self.STATUS_BUSY),
            'pixel_ready': bool(status & self.STATUS_PIXEL_READY),
            'raw': status
        }
    
    def get_pixel_count(self) -> int:
        """Get current pixel counter (0-783)"""
        return self._read_reg(self.REG_PIXEL_COUNT) & 0x3FF
    
    def get_result(self) -> int:
        """Read classification result (0=benign, 1=malignant)"""
        return self._read_reg(self.REG_RESULT) & 0x01
    
    def set_metric(self, metric: str = 'manhattan'):
        """
        Set distance metric
        
        Args:
            metric: 'manhattan' or 'euclidean'
        """
        ctrl = self._read_reg(self.REG_CONTROL)
        
        if metric.lower() == 'manhattan':
            ctrl = (ctrl & 0x0F) | self.CTRL_METRIC_MANHATTAN
        elif metric.lower() == 'euclidean':
            ctrl = (ctrl & 0x0F) | self.CTRL_METRIC_EUCLIDEAN
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        self._write_reg(self.REG_CONTROL, ctrl)
        if self.verbose:
            print(f"[INFO] Distance metric set to: {metric}")
    
    def write_pixels(self, pixels: np.ndarray, timeout_sec: float = 1.0):
        """
        Stream pixel data to FPGA
        
        Args:
            pixels: Flattened numpy array of 784 uint8 values
            timeout_sec: Maximum time to wait for completion
        
        Raises:
            ValueError: If pixel array wrong size
            TimeoutError: If pixel write times out
        """
        if pixels.size != 784:
            raise ValueError(f"Expected 784 pixels, got {pixels.size}")
        
        # Ensure uint8 type
        pixels = np.asarray(pixels, dtype=np.uint8)
        
        start_time = time.time()
        pixels_written = 0
        
        for pixel_idx, pixel_val in enumerate(pixels):
            # Wait for pixel_ready flag
            while True:
                status = self.get_status()
                if status['pixel_ready']:
                    break
                
                elapsed = time.time() - start_time
                if elapsed > timeout_sec:
                    raise TimeoutError(
                        f"Pixel write timeout at pixel {pixel_idx}. "
                        f"Status: {status}"
                    )
                time.sleep(0.0001)  # 100 µs poll interval
            
            # Write pixel
            self._write_reg(self.REG_PIXEL_DATA, int(pixel_val))
            pixels_written += 1
        
        if self.verbose:
            print(f"[INFO] Wrote {pixels_written} pixels")
    
    def classify_image(
        self,
        pixels: np.ndarray,
        metric: str = 'manhattan',
        timeout_sec: float = 1.0
    ) -> Dict[str, any]:
        """
        Classify an image end-to-end
        
        Args:
            pixels: 28×28 or flattened 784-element array (uint8)
            metric: 'manhattan' or 'euclidean'
            timeout_sec: Maximum processing time
        
        Returns:
            {
                'classification': int (0 or 1),
                'diagnosis': str ('benign' or 'malignant'),
                'metric': str,
                'latency_us': float,
                'pixel_count': int
            }
        
        Example:
            >>> from image_preprocessing import ImagePreprocessor
            >>> preprocessor = ImagePreprocessor()
            >>> image_array = preprocessor.preprocess('tumor.jpg')
            >>> fpga = MemoryMappedFPGA()
            >>> result = fpga.classify_image(image_array, metric='euclidean')
            >>> print(result['diagnosis'])  # "benign" or "malignant"
        """
        
        # Reshape if 28x28
        if pixels.shape == (28, 28):
            pixels = pixels.flatten()
        
        if self.verbose:
            print(f"\n[INFO] Starting classification (metric={metric})")
            start_time = time.time()
        
        # Set metric and reset
        self.set_metric(metric)
        self.reset()
        
        # Write control register to start processing
        self._write_reg(
            self.REG_CONTROL,
            (self.CTRL_METRIC_EUCLIDEAN if metric.lower() == 'euclidean' 
             else self.CTRL_METRIC_MANHATTAN) | self.CTRL_START
        )
        
        # Stream pixels
        self.write_pixels(pixels, timeout_sec)
        
        # Poll for completion
        start_poll = time.time()
        while True:
            status = self.get_status()
            if not status['busy']:
                break
            
            elapsed = time.time() - start_poll
            if elapsed > timeout_sec:
                raise TimeoutError(
                    f"Classification timeout after {elapsed:.3f}s. "
                    f"Status: {status}"
                )
            time.sleep(0.001)  # 1 ms poll interval
        
        # Read result
        classification = self.get_result()
        diagnosis = 'benign' if classification == 0 else 'malignant'
        
        total_latency = time.time() - start_time if self.verbose else 0
        
        if self.verbose:
            print(f"[INFO] Classification complete in {total_latency*1e6:.1f} µs")
            print(f"[INFO] Result: {diagnosis} (code={classification})")
        
        return {
            'classification': classification,
            'diagnosis': diagnosis,
            'metric': metric,
            'latency_us': total_latency * 1e6,
            'pixel_count': self.get_pixel_count()
        }
    
    def batch_classify(
        self,
        images: list,
        metric: str = 'manhattan',
        verbose_per_image: bool = False
    ) -> list:
        """
        Classify multiple images
        
        Args:
            images: List of 28×28 or flattened arrays
            metric: Distance metric to use
            verbose_per_image: Print result for each image
        
        Returns:
            List of result dictionaries
        """
        results = []
        for idx, image in enumerate(images):
            verbose_old = self.verbose
            self.verbose = verbose_per_image
            
            result = self.classify_image(image, metric)
            results.append(result)
            
            self.verbose = verbose_old
            if not verbose_per_image:
                print(f"  [{idx+1}/{len(images)}] {result['diagnosis']}", end=' ')
        
        return results
    
    def cleanup(self):
        """Clean up resources"""
        if self.mem_fd is not None:
            os.close(self.mem_fd)
            self.mem_fd = None
            if self.verbose:
                print("[INFO] Memory resources cleaned up")
    
    def __del__(self):
        """Destructor - cleanup on deletion"""
        self.cleanup()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    """
    Demo: Classify medical images using Zynq FPGA
    
    Usage:
        # On Zynq (Linux):
        sudo python3 fpga_interface.py
        
    This demo:
    1. Connects to FPGA via memory-mapped AXI interface
    2. Creates 4 synthetic medical images (different tissue characteristics)
    3. Classifies each with both distance metrics
    4. Shows timing and performance
    """
    
    import sys
    
    # Check if running on Zynq (has /dev/mem access)
    try:
        fpga = MemoryMappedFPGA(verbose=True)
    except (PermissionError, OSError) as e:
        print(f"\n[ERROR] {e}")
        print("\nNote: This script requires root access on Zynq Linux:")
        print("  $ sudo python3 fpga_interface.py")
        sys.exit(1)
    
    with fpga:
        print("\n" + "="*70)
        print("KNN FPGA Classifier - Zynq PS Interface Demo")
        print("="*70)
        
        # Create synthetic test images
        print("\n[1] Creating synthetic medical images...")
        
        # Test 1: Benign (smooth, low variance)
        benign = np.random.normal(120, 15, 784).clip(0, 255).astype(np.uint8)
        
        # Test 2: Malignant (rough, high variance)
        malignant = np.random.normal(140, 40, 784).clip(0, 255).astype(np.uint8)
        
        # Test 3: Border case
        border_case = np.random.normal(130, 25, 784).clip(0, 255).astype(np.uint8)
        
        test_images = [benign, malignant, border_case]
        test_names = ["Benign", "Malignant", "Border Case"]
        
        # Classify with both metrics
        for metric in ['manhattan', 'euclidean']:
            print(f"\n[2] Classifying with {metric.upper()} distance...")
            print(f"     Image Name         | Classification | Latency")
            print(f"     " + "-"*50)
            
            results = fpga.batch_classify(test_images, metric=metric)
            
            for name, result in zip(test_names, results):
                print(f"     {name:18} | {result['diagnosis']:14} | {result['latency_us']:7.1f} µs")
        
        print("\n[3] Zynq FPGA Classification Complete!")
        print("="*70)
