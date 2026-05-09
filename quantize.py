"""
quantize.py
───────────
Quantization pipeline using Google AI Edge Torch (litert_torch):

  PyTorch FP32  →  LiteRT (TFLite) INT8

Steps:
  1. Load trained FP32 UNet (MobileNetV2)
  2. Validate FP32 mIoU baseline
  3. Convert to TFLite using litert_torch
  4. Apply INT8 quantization using ai_edge_quantizer (PTQ)
  5. Validate TFLite INT8 mIoU
  6. Report accuracy drop
"""

import os
import glob
import logging
import numpy as np
import cv2
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

# ─────────────────────────── Fix JAX/NumPy 2.0 Compatibility ─────────────────
# JAX 0.4.30+ requires np.dtypes.StringDType which is only in NumPy 2.0+
if not hasattr(np, "dtypes"):
    class MockDtypes: pass
    np.dtypes = MockDtypes()
if not hasattr(np.dtypes, "StringDType"):
    np.dtypes.StringDType = lambda: np.dtype("S")
# ─────────────────────────────────────────────────────────────────────────────

import litert_torch
from ai_edge_quantizer import Quantizer
from ai_edge_quantizer import qtyping
from ai_edge_quantizer import recipe

# ─────────────────────────── Config ───────────────────────────────────────────
MODEL_PATH  = "/home/preetham-divate/barren_land_detection/D-FINE-seg/outputs/unet_advanced_0507_2127/unet_advanced_best.pth"
DATA_ROOT   = "/home/preetham-divate/barren_land_detection/D-FINE-seg/unet_data/archive"
OUT_DIR     = "/home/preetham-divate/barren_land_detection/D-FINE-seg/outputs/unet_advanced_0507_2127"
TFLITE_PATH = os.path.join(OUT_DIR, "unet_efficientnet_int8.tflite")

IMG_SIZE    = 384
N_CLASSES   = 4
DEVICE      = torch.device("cpu")
N_CALIB     = 250
MIOU_DROP_THRESHOLD = 0.02

CLASS_NAMES = ["background", "agriculture", "rangeland", "barren"]
COLOR2IDX = {(255, 255, 0): 1, (255, 0, 255): 2, (255, 255, 255): 3}

os.makedirs(OUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(os.path.join(OUT_DIR, "quantize.log")), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ─────────────────────── Helpers ──────────────────────────────────────────────
def mask_rgb_to_class(mask):
    out = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int64)
    # Agriculture (Yellow: 255, 255, 0)
    out[(mask[:,:,0] > 200) & (mask[:,:,1] > 200) & (mask[:,:,2] < 50)] = 1
    # Rangeland (Magenta: 255, 0, 255)
    out[(mask[:,:,0] > 200) & (mask[:,:,1] < 50) & (mask[:,:,2] > 200)] = 2
    # Barren (Cyan: 0, 255, 255)
    out[(mask[:,:,0] < 50) & (mask[:,:,1] > 200) & (mask[:,:,2] > 200)] = 3
    return out

# Morphological post-processing for cleanup
_morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

def morph_clean(pred_np: np.ndarray) -> np.ndarray:
    out = np.zeros_like(pred_np)
    for cls in range(N_CLASSES):
        m = (pred_np == cls).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _morph_kernel)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  _morph_kernel)
        out[m == 1] = cls
    return out


def preprocess_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img  = (img - mean) / std
    return img.transpose(2, 0, 1).astype(np.float32)

def compute_miou_pytorch(model, img_paths, mask_paths):
    model.eval()
    total_inter, total_union = np.zeros(N_CLASSES), np.zeros(N_CLASSES)
    with torch.no_grad():
        for img_p, mask_p in zip(img_paths, mask_paths):
            x = torch.from_numpy(preprocess_image(img_p)).unsqueeze(0).to(DEVICE).float()
            p = model(x).argmax(1).squeeze(0).cpu().numpy()
            
            # Apply morphological cleanup
            p = morph_clean(p)
            
            mask = cv2.imread(mask_p)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
            y = mask_rgb_to_class(mask)
            for cls in range(N_CLASSES):
                total_inter[cls] += ((p == cls) & (y == cls)).sum()
                total_union[cls] += ((p == cls) | (y == cls)).sum()

    ious = total_inter / (total_union + 1e-6)
    return ious, ious.mean()

def compute_miou_tflite(interpreter, img_paths, mask_paths):
    in_details = interpreter.get_input_details()[0]
    out_details = interpreter.get_output_details()[0]
    
    # Get quantization parameters for input
    in_quant = in_details.get("quantization_parameters", {})
    in_scale = in_quant.get("scales", [1.0])[0] if in_quant else 1.0
    in_zp = in_quant.get("zero_points", [0])[0] if in_quant else 0
    in_dtype = in_details["dtype"]

    # Get quantization parameters for output
    out_quant = out_details.get("quantization_parameters", {})
    out_scale = out_quant.get("scales", [1.0])[0] if out_quant else 1.0
    out_zp = out_quant.get("zero_points", [0])[0] if out_quant else 0
    out_dtype = out_details["dtype"]

    total_inter, total_union = np.zeros(N_CLASSES), np.zeros(N_CLASSES)
    
    for img_p, mask_p in zip(img_paths, mask_paths):
        img = preprocess_image(img_p)
        img_input = img[np.newaxis, ...]
        
        # Manually quantize if input is INT8
        if in_dtype == np.int8:
            img_input = (img_input / in_scale + in_zp).round().clip(-128, 127).astype(np.int8)
        
        interpreter.set_tensor(in_details["index"], img_input)
        interpreter.invoke()
        out = interpreter.get_tensor(out_details["index"])
        
        # Manually dequantize if output is INT8
        if out_dtype == np.int8:
            out = (out.astype(np.float32) - out_zp) * out_scale
            
        p = np.argmax(out[0], axis=0) # out is (1, N_CLASSES, H, W)
        
        # Apply morphological cleanup
        p = morph_clean(p)
        
        mask = cv2.imread(mask_p)

        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        y = mask_rgb_to_class(mask)
        for cls in range(N_CLASSES):
            total_inter[cls] += ((p == cls) & (y == cls)).sum()
            total_union[cls] += ((p == cls) | (y == cls)).sum()
            
    ious = total_inter / (total_union + 1e-6)
    return ious, ious.mean()


# ─────────────────────────── Main Pipeline ────────────────────────────────────
if __name__ == "__main__":
    # 1. Load Data
    val_imgs_raw = sorted(glob.glob(os.path.join(DATA_ROOT, "valid", "*_sat.jpg")))
    all_val_imgs, all_val_masks = [], []
    for p in val_imgs_raw:
        m = p.replace("_sat.jpg", "_mask.png")
        if os.path.exists(m):
            all_val_imgs.append(p)
            all_val_masks.append(m)
    if not all_val_imgs:
        all_imgs_raw = sorted(glob.glob(os.path.join(DATA_ROOT, "train", "*_sat.jpg")))
        split = int(0.8 * len(all_imgs_raw))
        all_val_imgs, all_val_masks = all_imgs_raw[split:], [p.replace("_sat.jpg", "_mask.png") for p in all_imgs_raw[split:]]

    # 2. Load Model
    model = smp.Unet(encoder_name="efficientnet-b0", encoder_weights=None, classes=N_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    model.eval()
    logger.info(f"Loaded FP32 model from {MODEL_PATH}")

    # 3. Validate FP32
    ious, fp32_miou = compute_miou_pytorch(model, all_val_imgs, all_val_masks)
    logger.info(f"FP32 mIoU: {fp32_miou:.4f}")

    # 4. Convert to LiteRT (TFLite)
    logger.info("── Converting to LiteRT via litert_torch ──")
    sample_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    edge_model = litert_torch.convert(model, (sample_input,))
    
    # Export the FP32 model first (Quantizer will load this file)
    edge_model.export(TFLITE_PATH)
    logger.info(f"FP32 TFLite exported to {TFLITE_PATH}")

    # 5. Quantize (INT8 PTQ)
    logger.info("── Applying INT8 Quantization ──")
    
    # Get the signature input name from the TFLite model
    from ai_edge_litert import interpreter as litert
    tmp_interp = litert.Interpreter(model_path=TFLITE_PATH)
    sig_list = tmp_interp.get_signature_list()
    # Usually {'serving_default': {'inputs': ['args_0'], ...}}
    input_name = sig_list["serving_default"]["inputs"][0]
    logger.info(f"TFLite signature input name detected: {input_name}")

    # Initialize quantizer with the file path
    qtiz = Quantizer(TFLITE_PATH)
    
    # Load the static INT8 recipe (PTQ)
    qtiz.load_quantization_recipe(recipe.static_wi8_ai8())
    
    # Prepare calibration data (list of dicts mapping name to tensor)
    calib_data = []
    step = max(1, len(all_val_imgs) // N_CALIB)
    for p in all_val_imgs[::step][:N_CALIB]:
        data = preprocess_image(p)[np.newaxis, ...]
        calib_data.append({input_name: torch.from_numpy(data)})


    
    # Run calibration
    logger.info(f"Running calibration on {len(calib_data)} images...")
    calib_result = qtiz.calibrate({"serving_default": calib_data})
    
    # Apply quantization
    quant_result = qtiz.quantize(calib_result)
    
    # Export the final INT8 model (overwrite the FP32 one)
    quant_result.export_model(TFLITE_PATH, overwrite=True)
    logger.info(f"Quantized TFLite saved to {TFLITE_PATH}")



    # 6. Validate TFLite
    from ai_edge_litert import interpreter as litert
    interpreter = litert.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()
    int8_ious, int8_miou = compute_miou_tflite(interpreter, all_val_imgs, all_val_masks)
    logger.info(f"TFLite INT8 mIoU: {int8_miou:.4f}")
    
    # 7. Final Report
    drop = fp32_miou - int8_miou
    logger.info(f"mIoU Drop: {drop:.4f} ({drop*100:.2f}%)")
    if drop > MIOU_DROP_THRESHOLD:
        logger.warning("Accuracy drop exceeds 2%! Consider QAT.")
    else:
        logger.info("Quantization successful! Model ready for deployment.")
