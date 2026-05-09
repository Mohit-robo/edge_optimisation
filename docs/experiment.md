# Barren Land Detection: Experimentation & Optimization Report

## 1. Project Objective
Deploy a high-fidelity semantic segmentation model on Raspberry Pi 4 to detect four terrain classes: Background, Agriculture, Rangeland, and Barren Land. The primary challenge was the extreme class imbalance (Rangeland and Barren being minority classes).

## 2. Experiment Progression

### Experiment 1: Baseline Pipeline
*   **Architecture**: UNet with MobileNetV2 Encoder
*   **Input Size**: 384 x 384
*   **Loss**: Standard Cross-Entropy
*   **Result**: mIoU ~0.62. Failed to capture Rangeland boundaries accurately.

### Experiment 2: Advanced Optimization (Current Best)
*   **Architecture**: UNet with **EfficientNet-B0** Encoder (Superior feature extraction).
*   **Input Size**: 384 x 384 (Increased spatial resolution).
*   **Loss Strategy**: Hybrid (Weighted Cross-Entropy + Focal + Tversky Loss).
*   **Mining**: **OHEM** (Online Hard Example Mining) focusing on the top 70% hardest pixels.
*   **Sampling**: **WeightedRandomSampler** (10x priority for Rangeland/Barren images).
*   **Post-processing**: Morphological Close/Open operations to remove salt-and-pepper noise.

#### Best Validation Metrics (Epoch 46)
| Metric | Value |
| :--- | :--- |
| **mIoU** | **0.6856** |
| **Pixel Accuracy** | **86.47%** |
| **Training Loss** | **0.3411** |

#### Per-Class IoU Breakdown
| Class | IoU |
| :--- | :--- |
| Background | 0.756 |
| Agriculture | 0.854 |
| **Rangeland** | **0.400** (Significant improvement from 0.32) |
| **Barren** | **0.733** (Significant improvement from 0.56) |

## 3. Post-Training Quantization (PTQ)
To achieve real-time performance on the Raspberry Pi 4 (ARM Cortex-A72), the PyTorch model was converted to **LiteRT (TFLite) INT8** format.

| Model Format | Precision | Size | Latency (Pi 4) |
| :--- | :--- | :--- | :--- |
| PyTorch (.pth) | FP32 | ~26 MB | ~1.5s |
| **LiteRT (.tflite)** | **INT8** | **~6.5 MB** | **~780ms** |

**Quantization Impact**: Through static range calibration, we achieved a ~4x reduction in model size with negligible (<1%) loss in mIoU (masked Intersection Over Union), enabling stable inference on edge hardware.

## 4. Hardware Actuation Logic
The pipeline is integrated with a hardware feedback loop:
1.  **Detection**: Model segments the frame in real-time.
2.  **Targeting**: If **Rangeland (Class 2)** or **Barren (Class 3)** is detected, the system calculates the geometric center of the specific class mask.
3.  **Actuation**: A PWM signal is sent to a Servo motor on **GPIO 17** to point the camera/sensor toward the detected target.

## 5. Conclusion
The transition to an EfficientNet backbone combined with specialized class-weighted loss functions successfully addressed the class imbalance problem. The resulting model provides a robust balance between accuracy (0.68 mIoU) and edge latency (~780ms) required for autonomous barren land monitoring.
