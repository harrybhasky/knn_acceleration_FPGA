# 📌 ESSENTIAL FILES - START HERE

## ✨ What's Inside?

This folder has **only the files you actually need** - nothing else!

```
ESSENTIAL/
├── README.md              ← Read this first
├── python/                ← The actual code
│   ├── knn_zynq_demo.py  (Main program - runs on Zynq)
│   ├── fpga_interface.py (Talks to FPGA)
│   ├── image_preprocessing.py (Cleans images)
│   └── knn_reference.py  (Pure Python for testing)
├── fpga/                  ← Hardware code
│   ├── knn_zynq_top.v    (Top module for Vivado)
│   ├── axi_lite_slave.v  (Register interface)
│   ├── feature_extractor.v (Features from pixels)
│   ├── distance_metric.v (Compute 50 distances)
│   ├── knn_selector.v    (Find K=3 nearest)
│   └── knn_classifier.v  (Majority vote)
└── docs/                  ← Documentation
    ├── README.md         (Overview)
    ├── QUICK_REFERENCE.md (API, register map, commands)
    └── FILE_GUIDE.md     (Detailed file descriptions)
```

---

## 🚀 Get Started in 3 Minutes

### **Step 1: Understand the Big Picture**
Read: `README.md` (this folder)  
Time: 5 minutes

### **Step 2: Know the Details**
Read: `docs/FILE_GUIDE.md` (each file explained)  
Time: 10 minutes

### **Step 3: Use It**
Read: `docs/QUICK_REFERENCE.md` (APIs, commands)  
Time: Reference

---

## 📋 File Sizes (Actual Code)

| Section | Files | Total Lines |
|---------|-------|-------------|
| **Python** | 4 files | 1,266 lines |
| **FPGA** | 6 files | 1,445 lines |
| **Docs** | 3 files | 17,000+ words |

**Total:** 10 production files, ~2,700 lines of actual code

---

## 🎯 One-Line Description Per File

### Python
- **knn_zynq_demo.py** - Main program you run
- **fpga_interface.py** - Memory-mapped AXI-Lite access
- **image_preprocessing.py** - JPG/PNG → 8-bit 28×28 pixels
- **knn_reference.py** - Pure Python KNN (no FPGA needed)

### Verilog
- **knn_zynq_top.v** - Top module connecting all pieces
- **axi_lite_slave.v** - Register interface for ARM↔FPGA
- **feature_extractor.v** - Extract mean & texture from pixels
- **distance_metric.v** - Compute all 50 distances in parallel
- **knn_selector.v** - Find K=3 nearest neighbors
- **knn_classifier.v** - Majority vote → diagnosis

---

## 💡 How to Use These Files

### **Option A: Understand First (Recommended)**
1. Read `README.md`
2. Read `docs/FILE_GUIDE.md` 
3. Read `docs/QUICK_REFERENCE.md`
4. Then look at actual code

### **Option B: Want to Run on Zynq?**
1. Copy `python/*.py` to Zynq board
2. Copy `fpga/*.v` to Vivado project
3. Follow `docs/QUICK_REFERENCE.md` → "How to Run"

### **Option C: Want to Test First?**
1. Use `knn_reference.py` (no FPGA needed)
2. It's pure Python, run on your laptop
3. Validates the algorithm works

---

## 🔗 If You Need More Details

From **parent folder** (`/new/`):
- `IMAGE_CLASSIFICATION_WORKFLOW.md` - Complete workflow trace
- `ZYNQ_DEPLOYMENT_GUIDE.md` - Full Zynq deployment
- `conference_101719.tex` - Academic paper with all details

These are NOT needed to run the project - they're just references!

---

## ⚡ Quick Performance Facts

- **Speed:** 16.8 µs (13.1× faster than CPU)
- **Power:** 2 W (259× more efficient than CPU)
- **Accuracy:** 90% classification accuracy
- **Image Size:** 28×28 pixels (784 total)
- **Features:** 2 (mean intensity + texture)
- **K Value:** 3 nearest neighbors
- **Training Samples:** 50 (hardcoded in FPGA)

---

## ❓ Quick Questions

**Q: Where do I start?**  
A: Read `README.md` then `docs/QUICK_REFERENCE.md`

**Q: How do I run it?**  
A: `sudo python3 python/knn_zynq_demo.py --image tumor.jpg`

**Q: How does it talk to FPGA?**  
A: Memory-mapped AXI-Lite registers (see `fpga_interface.py`)

**Q: How fast is it?**  
A: 16.8 microseconds per image (59,500 images/sec)

**Q: Can I test without FPGA?**  
A: Yes! Use `python3 python/knn_reference.py`

**Q: What if I want to change something?**  
A: See `docs/FILE_GUIDE.md` for modification suggestions per file

---

## ✅ What You Have

✓ Complete Python system (image load → FPGA interface)  
✓ Complete Verilog design (registers → classification)  
✓ Full documentation with examples  
✓ Reference implementation for testing  
✓ Quick reference cards  

---

## 📖 Recommended Reading Order

```
1. This file (INDEX.md) - 3 min
   ↓
2. README.md - 5 min
   ↓
3. docs/FILE_GUIDE.md - 15 min
   ↓
4. docs/QUICK_REFERENCE.md - Reference
   ↓
5. Read actual code (start with knn_zynq_demo.py)
```

---

## 🎓 Learning Resources

If you want to understand **why** things are done this way:

**Python/Hardware Communication:**  
- `fpga_interface.py` - Shows ctypes memory mapping

**FPGA Architecture:**  
- `knn_zynq_top.v` - Shows module hierarchy

**KNN Algorithm:**  
- `knn_reference.py` - Pure Python version (easy to read)
- `knn_classifier.v` - Hardware voting logic

**Image Processing:**  
- `image_preprocessing.py` - Shows resize, normalize steps

---

🎉 **You're ready to go! Start with README.md**

