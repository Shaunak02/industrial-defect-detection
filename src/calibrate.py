import sys
import numpy as np
import joblib
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.patchcore import PatchCore
from src.dataloader import get_dataloaders

def calibrate(category, data_root="./data"):
    print(f"Calibrating threshold for: {category}")
    
    model = PatchCore.load(f"./results/{category}/model.pkl")
    _, test_loader = get_dataloaders(data_root, category, batch_size=8)
    
    scores, _ = model.predict(test_loader)
    
    labels = []
    for batch in test_loader:
        labels.extend(batch["label"].numpy())
    labels = np.array(labels)

    normal_scores    = scores[labels == 0]
    defective_scores = scores[labels == 1]

    calibration = {
        "threshold": float(np.mean([normal_scores.max(), defective_scores.min()])),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
    }
    
    joblib.dump(calibration, f"./results/{category}/calibration.pkl")
    
    print(f"  Normal scores:    {normal_scores.mean():.4f} ± {normal_scores.std():.4f}")
    print(f"  Defective scores: {defective_scores.mean():.4f} ± {defective_scores.std():.4f}")
    print(f"  Threshold set at: {calibration['threshold']:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="hazelnut")
    args = parser.parse_args()
    calibrate(args.category)