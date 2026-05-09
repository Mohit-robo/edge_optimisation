# Barren Land Segmentation & Edge Optimization

This repository contains a high-performance semantic segmentation pipeline for detecting barren land and rangeland in satellite imagery. The model is specifically optimized for real-time deployment on **Raspberry Pi 4** using **INT8 Quantization**.

## 🚀 Key Features
- **Architecture**: UNet with an EfficientNet-B0 encoder for superior feature extraction.
- **Advanced Loss**: Hybrid implementation of Weighted Cross-Entropy, Focal Loss, and Tversky Loss to handle extreme class imbalance.
- **Edge Optimized**: Full INT8 Post-Training Quantization (PTQ) via the Google AI Edge toolchain.
- **Real-time Ready**: Achieves stable ~780ms inference on Raspberry Pi 4 CPU.

## 📂 Core Scripts
- `train_advanced.py`: The main training script featuring **Online Hard Example Mining (OHEM)** and advanced satellite-specific data augmentations.
- `quantize.py`: A robust quantization pipeline that converts PyTorch FP32 models to LiteRT (TFLite) INT8 format with minimal accuracy drop (<0.5%).

## 📊 Performance
| Model | mIoU | Size | Latency (Pi 4) |
| :--- | :--- | :--- | :--- |
| PyTorch FP32 | 0.6856 | ~26 MB | ~1.5s |
| **LiteRT INT8** | **0.6867** | **~6.5 MB** | **~780ms** |

## 🔗 Dataset
The model was trained on the — [DeepGlobe Land Cover Classification Dataset](https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset).

## 🛠️ Quick Start
1. **Train**: `python3 train_advanced.py`
2. **Quantize**: `python3 quantize.py`
3. **Deploy**: Use `ai-edge-litert` on your Raspberry Pi to run the generated `.tflite` model.
