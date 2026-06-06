from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class MVTecDataset(Dataset):
    def __init__(self, root, category, split="train", img_size=256):
        self.root = Path(root) / category
        self.split = split
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        if self.split == "train":
            for p in (self.root / "train" / "good").glob("*.png"):
                samples.append({"image": p, "label": 0, "defect": "good"})
        else:
            for defect_dir in (self.root / "test").iterdir():
                label = 0 if defect_dir.name == "good" else 1
                for p in defect_dir.glob("*.png"):
                    samples.append({"image": p, "label": label,
                                    "defect": defect_dir.name})
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        image = self.transform(Image.open(s["image"]).convert("RGB"))
        return {"image": image, "label": s["label"], "defect": s["defect"]}


def get_dataloaders(root, category, img_size=256, batch_size=32):
    train_ds = MVTecDataset(root, category, split="train",  img_size=img_size)
    test_ds  = MVTecDataset(root, category, split="test",   img_size=img_size)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0),
    )