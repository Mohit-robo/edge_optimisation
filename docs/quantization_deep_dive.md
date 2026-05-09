# Deep Dive: Post-Training Quantization for Edge AI Segmentation

This guide provides a comprehensive technical breakdown of our journey from a heavy PyTorch model to an ultra-lean, 8-bit optimized model running on a Raspberry Pi 4.

---

## 1. The Foundation: The PyTorch Experiment
Before we dive into quantization, it is important to understand the model we are optimizing. Our primary experiment involved a **UNet architecture** with an **EfficientNet-B0** encoder. 

*   **The Problem:** Severe class imbalance (satellite imagery often has 90% background/agriculture and <10% barren/rangeland).
*   **The Strategy:** We used a hybrid loss function combining **Weighted Cross-Entropy** and **Tversky Loss**, alongside **Online Hard Example Mining (OHEM)** to force the model to learn difficult pixel boundaries.
*   **The Tversky Advantage:** Unlike standard Dice loss which treats False Positives (FP) and False Negatives (FN) equally, Tversky Loss allows for specialized weighting:
    $$T(\alpha, \beta) = \frac{TP}{TP + \alpha FP + \beta FN}$$
    In our case, we used **$\alpha=0.3, \beta=0.7$**. By setting $\beta$ higher, we penalized False Negatives more aggressively. This ensured that the model prioritized "finding" the rare Rangeland patches even at the cost of slight over-segmentation.

*  **Best Validation Metrics** 

    | Metric | Value |
    | :--- | :--- |
    | **mIoU** | **0.6856** |
    | **Pixel Accuracy** | **86.47%** |
    | **Training Loss** | **0.3411** |

*  **Per-Class IoU Breakdown**

    | Class | IoU |
    | :--- | :--- |
    | Background | 0.756 |
    | Agriculture | 0.854 |
    | **Rangeland** | **0.400** |
    | **Barren** | **0.733**  |


---

## 2. Theory: What is Quantization?
At its core, quantization is the process of mapping a large set of values (32-bit floating point numbers) to a smaller, discrete set (8-bit integers). 

### The Mathematical Mapping
Standard neural networks represent weights and activations as 32-bit floats. Quantization transforms these values using the linear formula:
$$r = S(q - Z)$$
Where:
*   **$r$**: The original real-valued float.
*   **$q$**: The quantized 8-bit integer.
*   **$S$ (Scale)**: A positive float representing the step size.
*   **$Z$ (Zero-point)**: An integer ensuring that the real value "0.0" maps exactly to an integer.

### Why do this?
1.  **Memory Compression:** 8-bit integers take 4x less space than 32-bit floats.
2.  **Hardware Acceleration:** CPUs like the Raspberry Pi’s ARM Cortex-A72 can process integer arithmetic (SIMD) significantly faster and with less power consumption than floating-point math.

---

## 3. Post-Training Quantization (PTQ) Workflow
PTQ is the most efficient optimization method because it doesn't require retraining the model. We used the **Google AI Edge** ecosystem (`litert_torch` and `ai_edge_quantizer`) to execute this.

### Phase 1: Model Conversion (The Bridge)
**Theory:** PyTorch and TFLite use different execution graphs. We must first "lower" the PyTorch ATen operators into a format TFLite understands.
**Implementation:**
```python
import litert_torch
# We provide a sample input so the converter can trace the graph
sample_input = torch.randn(1, 3, 384, 384)
edge_model = litert_torch.convert(model, (sample_input,))
edge_model.export("unet_fp32.tflite")
```

### Phase 2: Calibration (Finding the Range)
**Theory:** Unlike weights (which are fixed), activations change based on the input image. To find the optimal $S$ and $Z$ for activations, we must pass "representative" data through the model to observe the typical range of values.
**Implementation:**
```python
# Pass 250 real images to collect activation statistics
calib_data = []
for p in calibration_images:
    data = preprocess(p)
    # TFLite signatures require dictionaries mapping name to tensor
    calib_data.append({"args_0": torch.from_numpy(data)})

qtiz = Quantizer("unet_fp32.tflite")
calib_result = qtiz.calibrate({"serving_default": calib_data})
```

### Phase 3: The Recipe (The Strategy)
**Theory:** We applied the `static_wi8_ai8` recipe. This stands for **Static Weights INT8, Activations INT8**. This is the "gold standard" for edge deployment because it quantizes every layer, including the inputs and outputs, ensuring the entire execution stays in the integer domain.
**Implementation:**
```python
from ai_edge_quantizer import recipe
# Apply the Static INT8 strategy
qtiz.load_quantization_recipe(recipe.static_wi8_ai8())
quant_result = qtiz.quantize(calib_result)
quant_result.export_model("unet_int8.tflite", overwrite=True)
```

---

## 4. Technical Hurdles & Resolutions
Our journey wasn't without blockers. As documented in our `lessons.md`:
1.  **Signature Mismatch:** TFLite Signature Runners expect input keys like `args_0`. We had to implement dynamic signature detection to make the pipeline robust.
2.  **Input Type Handling:** Because we chose an aggressive INT8 recipe, the model's entry point shifted from `float32` to `int8`. We had to manually apply the scale/zero-point logic in our inference script to feed the model correctly.
3.  **Environment Stability:** We bypassed NumPy 2.0/JAX conflicts using a custom attribute monkeypatch, ensuring the toolchain could run on stable Linux LTS versions.

---

## 5. The Results: Surprising Gains
Most engineers expect a slight accuracy drop when quantizing. Our results showed a rare but welcome phenomenon:

| Metric | PyTorch (FP32) | LiteRT (INT8) | Change |
| :--- | :--- | :--- | :--- |
| **mIoU** | **0.6856** | **0.6867** | **+0.11% (Gain!)** |
| **Model Size** | ~26 MB | ~6.5 MB | ~4x Reduction |
| **Latency (Pi 4)** | ~1500ms | ~780ms | ~2x Speedup |

**Note:** The 0.11% gain in mIoU is likely due to the quantization process acting as a form of "noise regularization," helping the model generalize slightly better on our specific validation set.

---

## 6. Deployment: The Raspberry Pi 4
We deployed the final `unet_efficientnet_int8.tflite` to a **Raspberry Pi 4 (ARM Cortex-A72)**. Using the `ai-edge-litert` runtime with 2-thread XNNPACK delegation, we achieved stable inference under 800ms, making it viable for periodic monitoring tasks.

## 7. Looking Forward: QAT
While our PTQ results were nearly perfect, there is always room to push. Our next post will explore **Quantization Aware Training (QAT)**. Instead of quantizing *after* training, QAT introduces "fake quantization" nodes *during* training, allowing the model to adapt its weights to the rounding errors before it ever hits the hardware.
