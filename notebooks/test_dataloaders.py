import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.dataloader import get_dataloaders


train_loader, test_loader = get_dataloaders(r"D:\CVProjects\industrial-defect-detection\data", "hazelnut")

batch = next(iter(train_loader))
print(f"Images: {batch['image'].shape}")   # → [32, 3, 256, 256]
print(f"Labels: {batch['label']}")         # → all 0s (train is normal only)
print(f"Defects: {set(batch['defect'])}")  # → {'good'}