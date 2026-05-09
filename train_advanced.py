import os
import glob
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import segmentation_models_pytorch as smp
from tqdm import tqdm
import logging
import datetime
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ─────────────────────────── Config ───────────────────────────────────────────
EPOCHS     = 100        

LR         = 3e-4
BATCH_SIZE = 8
IMG_SIZE   = 384
N_CLASSES  = 4
WD         = 1e-4

DATA_ROOT = "/home/preetham-divate/barren_land_detection/D-FINE-seg/unet_data/archive"
OUT_DIR   = f"/home/preetham-divate/barren_land_detection/D-FINE-seg/outputs/unet_advanced_{datetime.datetime.now().strftime('%m%d_%H%M')}"
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["background", "agriculture", "rangeland", "barren"]

# ─────────────────────────── Logger ───────────────────────────────────────────
def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

logger = setup_logger(OUT_DIR)

# ─────────────────────────── Dataset ──────────────────────────────────────────
def mask_rgb_to_class(mask):
    out = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int64)
    # Agriculture (Yellow: 255, 255, 0)
    out[(mask[:,:,0] > 200) & (mask[:,:,1] > 200) & (mask[:,:,2] < 50)] = 1
    # Rangeland (Magenta: 255, 0, 255)
    out[(mask[:,:,0] > 200) & (mask[:,:,1] < 50) & (mask[:,:,2] > 200)] = 2
    # Barren (Cyan: 0, 255, 255)
    out[(mask[:,:,0] < 50) & (mask[:,:,1] > 200) & (mask[:,:,2] > 200)] = 3
    return out

class SegDataset(Dataset):
    def __init__(self, img_paths, mask_paths, transform=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        
        mask = mask_rgb_to_class(mask)
        
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
            
        return img, mask.long()

# ────────────────────────── Augmentations ─────────────────────────────────────
train_tfms = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
    # Advanced: Color and Texture
    A.OneOf([
        A.RandomBrightnessContrast(p=1),
        A.HueSaturationValue(p=1),
        A.RandomGamma(p=1),
    ], p=0.4),
    A.Sharpen(p=0.2),
    A.OneOf([
        A.GaussNoise(p=1),
        A.GaussianBlur(p=1),
    ], p=0.2),
    # Satellite specific: simulate different resolutions/grid artifacts
    A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(8, 32), hole_width_range=(8, 32), p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_tfms = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# ────────────────────────── Loss & Sampler ────────────────────────────────────
class OHEMLoss(nn.Module):
    def __init__(self, base_loss, ratio=0.7):
        super().__init__()
        self.base_loss = base_loss
        self.ratio = ratio

    def forward(self, pred, target):
        # Calculate per-pixel loss
        loss = self.base_loss(pred, target) # Should be (B, H, W)
        loss = loss.view(-1)
        
        # Keep only the top 'ratio' hardest pixels
        num_pixels = loss.numel()
        num_hard = int(num_pixels * self.ratio)
        
        if num_hard > 0:
            loss, _ = torch.topk(loss, num_hard)
        return loss.mean()

# Weights based on inverse pixel frequency with even higher boost for Rangeland
class_weights = torch.tensor([1.0, 0.7, 10.0, 5.0]).to(DEVICE)

# Wrap CE Loss with OHEM (focus on top 70% hardest pixels)
ce_loss_base = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
ce_loss      = OHEMLoss(ce_loss_base, ratio=0.7)

focal_loss   = smp.losses.FocalLoss(mode="multiclass", gamma=2.0)
tversky_loss = smp.losses.TverskyLoss(mode="multiclass", alpha=0.3, beta=0.7)


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


def loss_fn(pred, target):
    return 0.4 * ce_loss(pred, target) + 0.3 * focal_loss(pred, target) + 0.3 * tversky_loss(pred, target)

def get_weighted_sampler(mask_paths):
    logger.info("Scanning dataset for class presence to build WeightedRandomSampler...")
    weights = []
    for p in tqdm(mask_paths):
        mask = cv2.imread(p)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        cls_mask = mask_rgb_to_class(mask)
        
        # Calculate image weight based on class presence
        # Class 2 (Rangeland) is the rarest/hardest - give it extreme priority
        if 2 in cls_mask:
            w = 10.0
        elif 3 in cls_mask:
            w = 4.0
        else:
            w = 1.0
        weights.append(w)

    return WeightedRandomSampler(weights, len(weights))

# ────────────────────────── Training Logic ────────────────────────────────────
def validate(model, loader):
    model.eval()
    total_inter = np.zeros(N_CLASSES)
    total_union = np.zeros(N_CLASSES)
    total_correct = 0
    total_pixels  = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            p = model(x).argmax(1)
            
            # Apply morphological cleanup before IoU calculation
            p_np = p.cpu().numpy()
            for b in range(p_np.shape[0]):
                p_np[b] = morph_clean(p_np[b])
            p = torch.from_numpy(p_np).to(DEVICE)
            
            total_correct += (p == y).sum().item()
            total_pixels  += y.numel()
            for cls in range(N_CLASSES):
                total_inter[cls] += ((p == cls) & (y == cls)).sum().item()
                total_union[cls] += ((p == cls) | (y == cls)).sum().item()

    ious = total_inter / (total_union + 1e-6)
    return ious, ious.mean(), total_correct / (total_pixels + 1e-6)

if __name__ == "__main__":
    logger.info(f"Advanced Training Initialization | Device: {DEVICE}")
    
    all_imgs  = sorted(glob.glob(os.path.join(DATA_ROOT, "train", "*_sat.jpg")))
    all_masks = [p.replace("_sat.jpg", "_mask.png") for p in all_imgs]
    split = int(0.8 * len(all_imgs))
    tr_imgs, val_imgs = all_imgs[:split], all_imgs[split:]
    tr_masks, val_masks = all_masks[:split], all_masks[split:]

    sampler = get_weighted_sampler(tr_masks)
    train_loader = DataLoader(SegDataset(tr_imgs, tr_masks, train_tfms), 
                              batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(SegDataset(val_imgs, val_masks, val_tfms), 
                            batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = smp.Unet(
        encoder_name="efficientnet-b0", 
        encoder_weights="imagenet", 
        classes=N_CLASSES
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    # Use OneCycleLR for faster, more stable convergence
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR*5, steps_per_epoch=len(train_loader), epochs=EPOCHS)

    best_miou = 0.0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for x, y in pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            p = model(x)
            loss = loss_fn(p, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        ious, miou, acc = validate(model, val_loader)
        logger.info(f"Epoch {epoch+1} | Loss: {train_loss/len(train_loader):.4f} | mIoU: {miou:.4f} | Acc: {acc:.4f}")
        logger.info(f"  IoUs: bg:{ious[0]:.3f} ag:{ious[1]:.3f} range:{ious[2]:.3f} barren:{ious[3]:.3f}")
        
        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "unet_advanced_best.pth"))
            logger.info(f"  ⭐ New Best: {best_miou:.4f}")

    logger.info(f"Training Complete. Best Val mIoU: {best_miou:.4f}")
