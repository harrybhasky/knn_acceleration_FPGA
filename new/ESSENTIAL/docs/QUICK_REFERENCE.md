# ⚡ QUICK REFERENCE

## 🐍 How to Run (on Zynq)

```bash
# Setup
cd /path/to/ESSENTIAL/python
chmod +x *.py

# Run with default settings (synthetic image)
sudo python3 knn_zynq_demo.py

# Run with real image
sudo python3 knn_zynq_demo.py --image /path/to/tumor.jpg --metric manhattan

# Run with Euclidean distance
sudo python3 knn_zynq_demo.py --image tumor.jpg --metric euclidean

# Test without FPGA (pure Python reference)
python3 knn_reference.py
```

---

## 🏗️ Vivado Project Setup

1. Create new Vivado project
2. Add files from `fpga/` folder:
   - `knn_zynq_top.v` (set as TOP module)
   - `axi_lite_slave.v`
   - `feature_extractor.v`
   - `distance_metric.v`
   - `knn_selector.v`
   - `knn_classifier.v`

3. Create Block Design (optional) or use RTL
4. Set top module: `knn_zynq_top`
5. Synthesize & Generate Bitstream

---

## 📋 AXI-Lite Register Map

```
Base Address: 0x43c00000 (configurable)

Offset  Name              Direction  Bits    Purpose
─────────────────────────────────────────────────────────────
0x00    CONTROL_REG       Write      [7:0]   [0]=start, [1]=reset, [7:4]=metric
0x04    STATUS_REG        Read       [7:0]   [0]=busy, [1]=pixel_ready
0x08    RESULT_REG        Read       [0]     Classification result
0x0C    PIXEL_DATA_REG    Write      [7:0]   Pixel value (auto-increment)
0x10    PIXEL_COUNT_REG   Read       [9:0]   Current pixel index (debug)
```

### Example: Write Pixel
```c
// In C/C++ on Zynq
volatile int *fpga_base = (int*)0x43c00000;
fpga_base[3] = pixel_value;  // PIXEL_DATA_REG @ offset 0x0C
```

### Example: Read Result
```c
volatile int *fpga_base = (int*)0x43c00000;
int result = fpga_base[2] & 0x1;  // RESULT_REG @ offset 0x08
// result: 0 = benign, 1 = malignant
```

---

## 🔌 Python AXI-Lite API

```python
from fpga_interface import MemoryMappedFPGA

# Initialize (must run as root)
fpga = MemoryMappedFPGA(base_addr=0x43c00000, verbose=True)

# Classify image
result = fpga.classify_image(pixels, metric='manhattan')
# pixels: numpy array of 784 uint8 values (28×28 flattened)
# metric: 'manhattan' or 'euclidean'
# returns: 0 (benign) or 1 (malignant)

# Batch classify (multiple images)
results = fpga.batch_classify(images_list, metric='manhattan')

# Get status
status = fpga.get_status()
# status['busy']: True/False
# status['pixel_ready']: True/False
```

---

## 🖼️ Image Pipeline

```python
from image_preprocessing import ImagePreprocessor

# Load and preprocess image
pixels = ImagePreprocessor.preprocess(
    image_path='/path/to/tumor.jpg',
    target_size=(28, 28),
    normalize=True
)
# Returns: numpy array (784,) with values 0-255
```

---

## 📊 Data Formats

### Input Image
```
Raw JPG/PNG file
    ↓ (load_image)
Grayscale array (H×W, uint8)
    ↓ (resize_image)
28×28 array (uint8)
    ↓ (normalize_image)
28×28 array normalized 0-255 (uint8)
    ↓ (flatten)
784-element array (uint8)
    ↓ (send to FPGA)
Ready for classification
```

### Output Result
```
1-bit classification
    0 = BENIGN
    1 = MALIGNANT
```

---

## ⚙️ Verilog Parameters

### Top Level: `knn_zynq_top.v`
```verilog
parameter DATA_WIDTH = 8;           // 8-bit pixel/feature values
parameter IMAGE_SIZE = 28;          // 28×28 pixels
parameter NUM_PIXELS = 784;         // Total pixels
parameter NUM_FEATURES = 2;         // Mean + Texture
parameter NUM_TRAINING = 50;        // Training samples
parameter K_VALUE = 3;              // K neighbors to vote
parameter DISTANCE_METRIC = 0;      // 0=Manhattan, 1=Euclidean
parameter CLK_FREQ_MHZ = 100;       // Clock frequency
```

### Key Modules
| Module | Lines | Function |
|--------|-------|----------|
| feature_extractor.v | 150 | Extract 2 features from 784 pixels |
| distance_metric.v | 200 | Compute all 50 distances in parallel |
| knn_selector.v | 150 | Find K=3 smallest distances |
| knn_classifier.v | 100 | Majority vote → binary output |
| axi_lite_slave.v | 476 | Register interface (PS↔PL) |

---

## 🎯 Performance Breakdown

```
Stage                    Latency    % of Total
─────────────────────────────────────────────
Pixel streaming          7.84 µs    47%
Feature extraction       7.90 µs    47%
Distance calculation     1.06 µs    6%
K-selection             0.04 µs    <1%
Voting                  <0.01 µs   <1%
─────────────────────────────────────────────
TOTAL                   16.8 µs    100%

Speedup vs CPU:         13.1×
Throughput:             59,500 img/sec
Power:                  2 W
Energy/Image:           34 nJ
```

---

## 🔍 Debug Checklist

**Python won't connect to FPGA?**
- Running as root? `sudo python3 ...`
- /dev/mem available? (need PetaLinux on Zynq)
- Correct base address? (default: 0x43c00000)

**RESULT_REG always reads 0?**
- STATUS_REG[0] (busy) = 0? (wait longer)
- 784 pixels sent? (check PIXEL_COUNT_REG)
- Metric selected correctly? (CONTROL_REG[7:4])

**Image preprocessing failing?**
- PIL/Pillow installed? `pip3 install Pillow`
- NumPy installed? `pip3 install NumPy`
- Image file readable? `file /path/to/image.jpg`

**Vivado synthesis errors?**
- All 6 Verilog files added?
- knn_zynq_top.v set as TOP?
- Clock constraint set to 100 MHz?

---

## 📚 Training Data

50 hardcoded samples in `distance_metric.v`:
- **25 Benign tumors** (label = 0)
- **25 Malignant tumors** (label = 1)
- **2 Features per sample:** mean intensity + texture

From Breast Cancer Wisconsin dataset, scaled to 8-bit.

---

## 🚀 Deployment Commands

```bash
# On Zynq, after Linux boots and FPGA programmed:

# Copy Python files
scp -r python/* <zynq_ip>:/home/root/

# SSH into Zynq
ssh root@<zynq_ip>

# Run classification
cd /home/root/
sudo python3 knn_zynq_demo.py --image input.jpg --metric manhattan

# Or batch process
for image in *.jpg; do
    sudo python3 knn_zynq_demo.py --image "$image"
done
```

---

## 📖 For More Details

- **Workflow explanation:** `../IMAGE_CLASSIFICATION_WORKFLOW.md`
- **Deployment guide:** `../ZYNQ_DEPLOYMENT_GUIDE.md`
- **Full LaTeX report:** `../conference_101719.tex`

