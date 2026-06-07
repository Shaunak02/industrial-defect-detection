# Industrial Defect Detection

End-to-end unsupervised defect detection pipeline using PatchCore on the MVTec AD dataset.
Motivated by real-world quality control challenges encountered at Fraunhofer IPT.

## Approach

PatchCore extracts patch-level features from a pretrained WideResNet-50 and builds a
memory bank of normal patch descriptors. At test time, anomaly scores are computed
as the nearest-neighbour distance from the memory bank — no labelled defects required.

## Results (Image-AUROC)

| Category | Ours | Paper |
|---|---|---|
| bottle | 99.9% | 99.6% |
| cable | 98.2% | 99.5% |
| capsule | 80.2% | 98.1% |
| carpet | 100.0% | 98.7% |
| grid | 83.0% | 98.2% |
| hazelnut | 100.0% | 99.9% |
| leather | 100.0% | 100.0% |
| metal_nut | 99.9% | 99.9% |
| pill | 95.6% | 96.6% |
| screw | 81.4% | 98.1% |
| tile | 99.9% | 98.7% |
| toothbrush | 87.5% | 100.0% |
| transistor | 98.4% | 99.6% |
| wood | 98.1% | 99.2% |
| zipper | 90.0% | 99.4% |
| **Mean** | **94.1%** | **99.0%** |

## Observations

*[To fill in — analysis of capsule, grid, screw gap]*

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python src/train.py --category hazelnut
```

## Stack
PyTorch · WideResNet-50 · scikit-learn · OpenCV