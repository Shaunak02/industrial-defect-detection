import sys
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.dataloader import get_dataloaders
from src.patchcore import PatchCore
from sklearn.metrics import roc_auc_score
from src.visualize import visualize_results
import joblib


def train_and_evaluate(category, data_root="./data"):
    print(f"\n{'='*50}")
    print(f"PatchCore — category: {category}")
    print(f"{'='*50}")

    train_loader, test_loader = get_dataloaders(data_root, category, img_size=224, batch_size=8)

    model = PatchCore(device="cpu")  # change to "cpu" if no GPU
    model.fit(train_loader)
    # Save immediately after fitting — before evaluation
    # This way even if evaluation crashes, the model is preserved
    model.save(f"./results/{category}/model.pkl")


    print("\nEvaluating on test set...")
    scores, maps = model.predict(test_loader)

    # Collect labels to compute calibration stats
    labels = []
    for batch in test_loader:
        labels.extend(batch["label"].numpy())
    labels = np.array(labels)

    normal_scores   = scores[labels == 0]
    defective_scores = scores[labels == 1]

    # Save calibration alongside the model
    calibration = {
        "threshold":  float(np.mean([normal_scores.max(), defective_scores.min()])),
        "score_min":  float(scores.min()),
        "score_max":  float(scores.max()),
    }
    joblib.dump(calibration, f"./results/{category}/calibration.pkl")
    print(f"  Threshold calibrated at: {calibration['threshold']:.4f}")
    print(f"  Normal scores:    {normal_scores.mean():.4f} ± {normal_scores.std():.4f}")
    print(f"  Defective scores: {defective_scores.mean():.4f} ± {defective_scores.std():.4f}")

    print(f"\nGenerating visualisations for {category}...")
    visualize_results(
    test_loader=test_loader,
    anomaly_maps=maps,
    category=category,
    data_root=data_root,
    n_samples=5      # save 5 defective examples per category
)

    # Collect ground truth labels
    labels = []
    for batch in test_loader:
        labels.extend(batch["label"].numpy())
    labels = np.array(labels)

    auroc = roc_auc_score(labels, scores)
    print(f"\nImage-AUROC: {auroc * 100:.1f}%")
    return auroc



CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper"
]

# Temporarily 
# CATEGORIES = ["capsule", "grid", "screw"]

# CATEGORIES = ["hazelnut"]

PAPER_RESULTS = {
    "bottle": 99.6, "cable": 99.5, "capsule": 98.1, "carpet": 98.7,
    "grid": 98.2, "hazelnut": 99.9, "leather": 100.0, "metal_nut": 99.9,
    "pill": 96.6, "screw": 98.1, "tile": 98.7, "toothbrush": 100.0,
    "transistor": 99.6, "wood": 99.2, "zipper": 99.4,
}

if __name__ == "__main__":
    import pandas as pd
    rows = []
    for cat in CATEGORIES:
        auroc = train_and_evaluate(cat)
        paper = PAPER_RESULTS[cat]
        rows.append({
            "category": cat,
            "our_auroc": round(auroc * 100, 1),
            "paper_auroc": paper,
            "delta": round((auroc * 100) - paper, 1)
        })

    df = pd.DataFrame(rows)
    df.to_csv("./results/metrics.csv", index=False)

    print("\n" + "="*60)
    print(df.to_string(index=False))
    print(f"\nOur mean:   {df['our_auroc'].mean():.1f}%")
    print(f"Paper mean: {df['paper_auroc'].mean():.1f}%")