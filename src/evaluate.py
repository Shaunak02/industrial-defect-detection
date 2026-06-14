import time
import pandas as pd
from pathlib import Path
from anomalib.engine import Engine
from anomalib.models import Patchcore
from anomalib.data import MVTec

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper"
]

# Published PatchCore numbers (from the original paper) — for comparison
PAPER_RESULTS = {
    "bottle": 99.6, "cable": 99.5, "capsule": 98.1, "carpet": 98.7,
    "grid": 98.2,   "hazelnut": 99.9, "leather": 100.0, "metal_nut": 99.9,
    "pill": 96.6,   "screw": 98.1, "tile": 98.7, "toothbrush": 100.0,
    "transistor": 99.6, "wood": 99.2, "zipper": 99.4,
}

def evaluate_all():
    rows = []
    for cat in CATEGORIES:
        print(f"\nEvaluating: {cat}")
        start = time.time()

        datamodule = MVTec(
            root="./data/mvtec",
            category=cat,
            image_size=256,
            eval_batch_size=32,
            num_workers=4,
        )
        model = Patchcore(
            backbone="wide_resnet50_2",
            layers_list=["layer2", "layer3"],
            coreset_sampling_ratio=0.1,
            num_neighbors=9,
        )
        engine = Engine(
            accelerator="gpu",
            devices=1,
            max_epochs=1,
            default_root_dir=f"./results/{cat}",
        )

        engine.fit(model=model, datamodule=datamodule)
        results = engine.test(model=model, datamodule=datamodule)

        elapsed = round(time.time() - start, 1)

        # Extract metrics from results dict
        img_auroc   = round(results[0].get("image_AUROC", 0) * 100, 1)
        pixel_auroc = round(results[0].get("pixel_AUROC", 0) * 100, 1)
        paper_val   = PAPER_RESULTS.get(cat, "-")
        delta       = round(img_auroc - paper_val, 1) if paper_val != "-" else "-"

        rows.append({
            "category":       cat,
            "image_auroc":    img_auroc,
            "pixel_auroc":    pixel_auroc,
            "paper_auroc":    paper_val,
            "vs_paper":       delta,
            "time_sec":       elapsed,
        })

        print(f"  Image-AUROC: {img_auroc}%  |  Pixel-AUROC: {pixel_auroc}%  "
              f"|  Paper: {paper_val}%  |  Δ {delta}  |  {elapsed}s")

    df = pd.DataFrame(rows)
    df.to_csv("./results/metrics.csv", index=False)

    print("\n" + "="*60)
    print(df.to_string(index=False))
    print(f"\nMean Image-AUROC: {df['image_auroc'].mean():.1f}%")
    print("Results saved to ./results/metrics.csv")

if __name__ == "__main__":
    evaluate_all()