import os
import logging
import torch
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import jaccard_score

from unet import UNet
from utils.data_loading import BasicDataset
from predict import predict_img


def load_mask(path: str):
    """Load manual segmentation as numpy array (binary mask)."""
    mask = Image.open(path).convert("L")  # grayscale
    mask = np.array(mask)
    #mask = mask[:-70]  # Schneide Bauchbinde ab (70px)
    return mask


# Dice Funktion

def dice_coeff(input: torch.Tensor, target: torch.Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def compute_metric(pred: np.ndarray, target: np.ndarray):
    """Compute IoU (sklearn) und Dice (torch)."""
    pred_flat = pred.flatten()
    target_flat = target.flatten()

    # Dice
    pred_tensor = torch.from_numpy(pred.astype(np.float32)).unsqueeze(0)  # (1,H,W)
    target_tensor = torch.from_numpy(target.astype(np.float32)).unsqueeze(0)  # (1,H,W)
    dice = dice_coeff(pred_tensor, target_tensor, reduce_batch_first=True).item()

    return dice

def compute_metrics(pred, target, eps=1e-6):
    target = target/255
    overlap = np.sum(pred * target)          # Schnittmenge
    total = np.sum(pred) + np.sum(target)    # Summe der beiden Masken
    return (2.0 * overlap + eps) / (total + eps)


def evaluate_models(models_dir, input_image, manual_mask, output_csv,
                    scale=1.0, threshold=0.5, classes=2, bilinear=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device {device}")

    img = Image.open(input_image)
    manual = load_mask(manual_mask)

    results = []

    for model_file in os.listdir(models_dir):
        if not model_file.endswith(".pth"):
            continue

        model_path = os.path.join(models_dir, model_file)
        logging.info(f"Evaluating {model_path}")

        # Load model
        net = UNet(n_channels=1, n_classes=classes, bilinear=bilinear)
        net.to(device=device)
        state_dict = torch.load(model_path, map_location=device)
        mask_values = state_dict.pop('mask_values', [0, 1])
        net.load_state_dict(state_dict)
        net.eval()

        # Prediction
        pred_mask = predict_img(net=net,
                                full_img=img,
                                device=device,
                                scale_factor=scale,
                                out_threshold=threshold)

        if classes > 1:
            pred_mask = (pred_mask > 0).astype(np.uint8)

        # Compute metrics
        dice = compute_metrics(pred_mask, manual)
        logging.info(f"{model_file}:, Dice={dice:.4f}")

        results.append({"model": model_file, "Dice": dice})

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    logging.info(f"Saved results to {output_csv}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    models_dir = r"/home/fabiankock/PycharmProjects/Pytorch-UNet/checkpoints"                                                # Ordner mit .pth-Dateien
    input_image = r"/home/fabiankock/PycharmProjects/Pytorch-UNet/preedited_images/_029.tif"                                 # Testbild
    manual_mask = r"/home/fabiankock/PycharmProjects/BachelorarbeitNeuNeu/_029_label_cut.tif"      # Ground-Truth Maske
    output_csv = r"results.csv"          # Ergebnisse

    evaluate_models(models_dir, input_image, manual_mask, output_csv,
                    scale=0.5, threshold=0.5, classes=2, bilinear=False)
