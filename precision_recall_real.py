import os
import logging
import torch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score

from unet import UNet
from predict import predict_img


def load_mask(path: str):
    """Load manual segmentation as numpy array (binary mask)."""
    mask = Image.open(path).convert("L")  # grayscale
    mask = np.array(mask)
    mask = (mask > 127).astype(np.uint8)  # binär machen
    return mask


def compute_metrics(pred: np.ndarray, target: np.ndarray):
    """Compute Precision und Recall."""
    pred_flat = pred.flatten()
    target_flat = target.flatten()

    precision = precision_score(target_flat, pred_flat, zero_division=0)
    recall = recall_score(target_flat, pred_flat, zero_division=0)
    f_score = 2*precision*recall/(precision+recall)
    return precision, recall, f_score


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

        # Immer auf binär bringen (0/1)
        pred_mask = (pred_mask > 0).astype(np.uint8)

        # Compute metrics
        precision, recall, f_score = compute_metrics(pred_mask, manual)
        logging.info(f"{model_file}: Precision={precision:.4f}, Recall={recall:.4f}, f_score={recall:.4f}")

        results.append({"model": model_file, "Precision": precision, "Recall": recall, "f_score": f_score})

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    logging.info(f"Saved results to {output_csv}")

    # Plot Precision, Recall und F1-Score
    df_sorted = df.copy()
    df_sorted["epoch"] = df_sorted["model"].str.extract(r"epoch(\d+)")[0].astype(float)
    df_sorted = df_sorted.dropna(subset=["epoch"])
    df_sorted["epoch"] = df_sorted["epoch"].astype(int)
    df_sorted = df_sorted.sort_values("epoch")

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    ax1.plot(df_sorted["epoch"], df_sorted["Precision"], marker="o", label="Precision")
    ax1.set_ylabel("Precision")
    ax1.set_title("Precision")
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.plot(df_sorted["epoch"], df_sorted["Recall"], marker="o", color="orange", label="Recall")
    ax2.set_ylabel("Recall")
    ax2.set_title("Recall")
    ax2.grid(True, linestyle="--", alpha=0.6)

    # F1-Score berechnen
    ax3.plot(df_sorted["epoch"], df_sorted["f_score"], marker="o", color="green", label="F1-Score")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("F1-Score")
    ax3.set_title("F1-Score")
    ax3.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    models_dir = r"/home/fabiankock/PycharmProjects/Pytorch-UNet/checkpoints"  # Ordner mit .pth-Dateien
    input_image = r"/home/fabiankock/PycharmProjects/Pytorch-UNet/preedited_images/_029.tif"  # Testbild
    manual_mask = r"/home/fabiankock/PycharmProjects/BachelorarbeitNeuNeu/_029_label_cut.tif"  # Ground-Truth Maske
    output_csv = r"results_pr.csv"

    evaluate_models(models_dir, input_image, manual_mask, output_csv,
                    scale=0.5, threshold=0.5, classes=2, bilinear=False)