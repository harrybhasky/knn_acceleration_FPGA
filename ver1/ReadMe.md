# FPGA kNN Classifier – Version 1

## Project Overview
This project implements a basic k-Nearest Neighbors (kNN) classifier in Verilog,
targeted for FPGA-based acceleration. The goal is to demonstrate how a traditionally
software-based machine learning algorithm can be mapped into cycle-accurate hardware.

Version 1 focuses on:
- Distance computation
- Sequential comparison across training samples
- Selection of the nearest neighbor

## Current Status (25% Completion)
✔ Training data initialized in hardware  
✔ Absolute distance calculation module  
✔ Sequential minimum-distance search  
✔ Simulation verified using testbench  

Planned next stages include:
- Multi-dimensional feature vectors (histograms)
- k > 1 neighbor selection
- Hardware sorting / voting
- FPGA resource optimization

## Design Description

### Training Memory
The classifier stores fixed training samples and their labels in internal memory.
Each index corresponds to one training sample and its associated class label.

### Distance Calculation
Distance is computed as the absolute difference between the test input and each
training sample:

|a - b|

This is
