import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image


def visualize_results(test_loader, anomaly_maps, category,
                      data_root="./data/mvtec",
                      output_dir="./results", n_samples=5):
    """
    Three-panel visualization for defective test images:
    1. Original image
    2. PatchCore anomaly map — where patch distances are high
    3. Ground truth mask — MVTec's pixel-level defect annotation

    No Grad-CAM — it answered a different question (ImageNet saliency)
    not relevant to defect localization.
    """
    output_path = Path(output_dir) / category / "visualizations"
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect defective samples with their index into anomaly_maps
    defective_samples = []
    global_idx = 0

    for batch in test_loader:
        batch_size = len(batch["label"])
        for i in range(batch_size):
            if batch["label"][i] == 1:
                defective_samples.append({
                    "image":      batch["image"][i],
                    "defect":     batch["defect"][i],
                    "global_idx": global_idx + i
                })
        global_idx += batch_size

    if not defective_samples:
        print(f"  No defective samples found for {category}")
        return

    samples = defective_samples[:n_samples]
    print(f"  Visualising {len(samples)} defective samples...")

    # Track per-defect-type count for mask path resolution
    defect_counters = {}

    for sample in samples:
        defect_name = sample["defect"]
        defect_counters[defect_name] = defect_counters.get(defect_name, 0)

        # Denormalise image
        img_np = sample["image"].permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        img_np = np.clip(img_np * std + mean, 0, 1).astype(np.float32)
        h, w = img_np.shape[:2]

        # Anomaly map — resize and normalise for display
        raw_map = anomaly_maps[sample["global_idx"]]
        anomaly_map = cv2.resize(raw_map, (w, h))
        anomaly_map_norm = (anomaly_map - anomaly_map.min()) / \
                           (anomaly_map.max() - anomaly_map.min() + 1e-8)

        # Overlay heatmap on original
        heatmap = cv2.applyColorMap(
            (anomaly_map_norm * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        anomaly_overlay = np.clip(
            0.5 * (img_np * 255) + 0.5 * heatmap, 0, 255
        ).astype(np.uint8)

        # Load ground truth mask
        gt_overlay = None
        defect_dir = Path(data_root) / category / "test" / defect_name
        mask_dir   = Path(data_root) / category / "ground_truth" / defect_name
        defect_images = sorted(defect_dir.glob("*.png"))
        count = defect_counters[defect_name]

        if count < len(defect_images):
            mask_path = mask_dir / f"{defect_images[count].stem}_mask.png"
            if mask_path.exists():
                gt_mask = np.array(Image.open(mask_path).resize((w, h)))
                gt_overlay = (img_np * 255).astype(np.uint8).copy()
                # Red overlay on defect region
                gt_overlay[gt_mask > 0] = [255, 0, 0]
                gt_overlay = np.clip(
                    0.6 * (img_np * 255) + 0.4 * gt_overlay, 0, 255
                ).astype(np.uint8)

        defect_counters[defect_name] += 1

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        panels = [
            ("Original",              (img_np * 255).astype(np.uint8)),
            ("Anomaly map",           anomaly_overlay),
            ("Ground truth",          gt_overlay),
        ]

        for ax, (title, img) in zip(axes, panels):
            if img is not None:
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "mask not found",
                        ha="center", va="center",
                        transform=ax.transAxes, color="gray")
                ax.set_facecolor("#f5f5f5")
            ax.set_title(title, fontsize=11)
            ax.axis("off")

        fig.suptitle(f"{category} — {defect_name}", fontsize=12)
        plt.tight_layout()

        save_path = output_path / f"{defect_name}_{count:02d}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_path}")

    print(f"  Done. Visualisations at {output_path}")