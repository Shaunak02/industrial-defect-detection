import sys
import cv2
import joblib
import argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms

sys.path.append(str(Path(__file__).parent.parent))
from src.patchcore import PatchCore


def preprocess_image(image_path, img_size=224):
    """
    Load and preprocess a single image for inference.
    
    Why these specific normalisation values? They are ImageNet's
    channel mean and standard deviation. Since our backbone was
    pretrained on ImageNet, it expects inputs normalised this way.
    Feeding an unnormalised image would produce garbage features —
    the network has never seen inputs in that range during training.
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std= [0.229, 0.224, 0.225]
        ),
    ])
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)  # add batch dimension → [1, 3, H, W]
    return tensor, np.array(img)


def save_result(original_image, anomaly_map, anomaly_score, is_defective, output_path):
    """
    Save a two-panel result image: original + anomaly heatmap overlay.
    
    Why two panels and not just the heatmap? Context matters. A recruiter
    or user looking at the output needs to see both what the image looked
    like AND where the model flagged anomalies. A heatmap alone is unreadable
    without the original beside it.
    """
    import matplotlib.pyplot as plt

    # Resize anomaly map to match original image
    h, w = original_image.shape[:2]
    anomaly_map_resized = cv2.resize(anomaly_map, (w, h))

    # Normalise to 0-255 for colormap
    anomaly_norm = (anomaly_map_resized - anomaly_map_resized.min()) / \
                   (anomaly_map_resized.max() - anomaly_map_resized.min() + 1e-8)

    heatmap = cv2.applyColorMap(
        (anomaly_norm * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Blend heatmap with original
    original_float = original_image.astype(np.float32)
    overlay = np.clip(0.5 * original_float + 0.5 * heatmap, 0, 255).astype(np.uint8)

    # Determine result label for the title
    # Threshold of 0.5 on normalised score is a reasonable default —
    # in production you'd calibrate this on a validation set
    label = "DEFECTIVE" if is_defective else "NORMAL"
    colour = "red" if is_defective else "green"

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(original_image)
    axes[0].set_title("Input image", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title("Anomaly map", fontsize=11)
    axes[1].axis("off")

    fig.suptitle(
        f"Result: {label}  |  Anomaly score: {anomaly_score:.4f}",
        fontsize=13, color=colour, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Result saved to {output_path}")



def run_inference(model_path, image_path, output_path=None, device="cpu"):
    print(f"\nLoading model from {model_path}...")
    model = PatchCore.load(model_path, device=device)

    # Load calibration — it sits next to the model file
    calib_path = Path(model_path).parent / "calibration.pkl"
    if calib_path.exists():
        calibration = joblib.load(calib_path)
        threshold  = calibration["threshold"]
        score_min  = calibration["score_min"]
        score_max  = calibration["score_max"]
        print(f"  Calibration loaded — threshold: {threshold:.4f}")
    else:
        # Fallback if no calibration file exists
        print("  Warning: no calibration file found, using default threshold")
        threshold  = None
        score_min  = 0.0
        score_max  = 100.0

    print(f"Processing image: {image_path}")
    image_tensor, image_np = preprocess_image(image_path)

    single_sample = [{"image": image_tensor.squeeze(0), "label": torch.tensor(0)}]
    loader = torch.utils.data.DataLoader(single_sample, batch_size=1)

    scores, maps = model.predict(loader)
    raw_score = float(scores[0])

    # Normalise using actual data range — not an arbitrary constant
    normalised = (raw_score - score_min) / (score_max - score_min + 1e-8)
    normalised = float(np.clip(normalised, 0, 1))

    # Use calibrated threshold if available
    if threshold is not None:
        is_defective = raw_score > threshold
    else:
        is_defective = normalised > 0.5

    print(f"\nAnomaly score (raw):        {raw_score:.4f}")
    print(f"Anomaly score (normalised): {normalised:.4f}")
    print(f"Threshold:                  {threshold:.4f}" if threshold else "")
    print(f"Verdict: {'DEFECTIVE ⚠' if is_defective else 'NORMAL ✓'}")

    if output_path is None:
        output_path = Path(image_path).stem + "_result.png"

    save_result(image_np, maps[0], normalised, is_defective, output_path)
    return normalised


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run PatchCore inference on a single image"
    )
    parser.add_argument(
        "--model",  required=True,
        help="Path to saved model file, e.g. results/hazelnut/model.pkl"
    )
    parser.add_argument(
        "--image",  required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to save result image (default: {image_name}_result.png)"
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"],
        help="Device to run on"
    )
    args = parser.parse_args()

    run_inference(
        model_path=args.model,
        image_path=args.image,
        output_path=args.output,
        device=args.device,
    )