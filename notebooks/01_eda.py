import os
from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np



mvtec_root = Path("D:\CVProjects\industrial-defect-detection\data")
categories = sorted([d.name for d in mvtec_root.iterdir() if d.is_dir()])

rows = []
for cat in categories:
    train_good = len(list((mvtec_root / cat / "train" / "good").glob("*.png")))
    test_good  = len(list((mvtec_root / cat / "test" / "good").glob("*.png")))
    defect_types = [d.name for d in (mvtec_root / cat / "test").iterdir()
                    if d.is_dir() and d.name != "good"]
    test_defect = sum(len(list((mvtec_root / cat / "test" / d).glob("*.png")))
                      for d in defect_types)
    rows.append({"category": cat, "train_normal": train_good,
                 "test_normal": test_good, "test_defect": test_defect,
                 "defect_types": len(defect_types)})

df = pd.DataFrame(rows)
# print(df.to_string(index=False))


cat = "capsule"
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

# Row 1: Normal samples
good_imgs = list((mvtec_root / cat / "train" / "good").glob("*.png"))[:5]

for ax, p in zip(axes[0], good_imgs):
    ax.imshow(Image.open(p)); ax.set_title("normal"); ax.axis("off")

# Row 2 : one of each defect type
defect_dirs = [d for d in (mvtec_root / cat / "test").iterdir() if d.name != "good"]

for ax, d in zip(axes[1], defect_dirs):
    img = list(d.glob("*.png"))[0]
    ax.imshow(Image.open(img)) ; ax.set_title(d.name); ax.axis("off")


plt.suptitle(f"MVTec AD — {cat}", fontsize=14)
plt.tight_layout()
# plt.savefig("D:\CVProjects\industrial-defect-detection/assets/eda_sample.png", dpi=150)

#Check image sizes

sizes = [Image.open(p).size for p in (mvtec_root / "capsule" / "train" / "good").glob("*.png")]
print(f"Min: {min(sizes)}  Max: {max(sizes)}  Unique: {set(sizes)}")

