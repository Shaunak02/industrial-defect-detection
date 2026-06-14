import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from sklearn.neighbors import NearestNeighbors
import joblib
from pathlib import Path

class PatchCore:
    """
    PatchCore anomaly detector.
    
    Why this structure: we separate feature extraction (neural network)
    from anomaly scoring (nearest neighbour search). These are two 
    fundamentally different operations and keeping them separate makes
    the code easier to understand and modify.
    """

    def __init__(self, backbone="wide_resnet50_2", layers=["layer2", "layer3"],
                 coreset_ratio=0.1, n_neighbors=9, device="cuda"):
        self.device = device
        self.n_neighbors = n_neighbors
        self.coreset_ratio = coreset_ratio
        self.memory_bank = None

        # Load pretrained backbone and freeze it
        # Why freeze? We are NOT training this network. We only use it
        # as a fixed feature extractor — its ImageNet knowledge is what
        # we want, unchanged.
        backbone_model = models.wide_resnet50_2(
            weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1
        )
        backbone_model.eval()
        for param in backbone_model.parameters():
            param.requires_grad = False
        backbone_model = backbone_model.to(device)

        # Build hooks to capture intermediate layer outputs
        # Why hooks? We don't want the final classification output.
        # We want the rich spatial features from middle layers.
        self.features = {}
        self.hooks = []

        for layer_name in layers:
            layer = dict(backbone_model.named_children())[layer_name]
            hook = layer.register_forward_hook(self._make_hook(layer_name))
            self.hooks.append(hook)

        self.backbone = backbone_model
        self.layers = layers

    def _make_hook(self, name):
        """
        A hook is a function that PyTorch calls automatically every time
        a layer finishes its forward pass. We use it to grab intermediate
        outputs without modifying the network itself.
        """
        def hook(module, input, output):
            self.features[name] = output
        return hook

    def _extract_features(self, images):
        """
        Run images through the backbone and collect patch-level features
        from the specified layers.

        Why concatenate layers? Layer2 captures fine-grained texture.
        Layer3 captures coarser structure. Together they give richer
        descriptions of each patch than either alone.
        """
        self.features = {}
        with torch.no_grad():
            self.backbone(images.to(self.device))

        feature_maps = []
        for layer_name in self.layers:
            feat = self.features[layer_name]  # shape: [B, C, H, W]

            # Resize all feature maps to the same spatial size so we
            # can concatenate them. We use layer2's spatial size as target.
            target_size = self.features[self.layers[0]].shape[-2:]
            if feat.shape[-2:] != target_size:
                feat = nn.functional.interpolate(
                    feat, size=target_size, mode="bilinear", align_corners=False
                )
            feature_maps.append(feat)

        # Concatenate along channel dimension → [B, C_total, H, W]
        combined = torch.cat(feature_maps, dim=1)

        # Reshape to patch format → [B*H*W, C_total]
        # Each spatial position becomes one "patch descriptor"
        B, C, H, W = combined.shape
        patches = combined.permute(0, 2, 3, 1).reshape(-1, C)

        return patches.cpu().numpy(), (B, H, W)

    def fit(self, dataloader):
        """
        Build the memory bank from normal training images.
        This is the entire 'training' of PatchCore.
        """
        print("Building memory bank from normal training images...")
        all_patches = []

        for batch in dataloader:
            images = batch["image"]
            patches, _ = self._extract_features(images)
            all_patches.append(patches)

        all_patches = np.concatenate(all_patches, axis=0)
        all_patches = self._normalize(all_patches)   #L2 normalisation
        print(f"  Total patches extracted: {all_patches.shape[0]:,}")

        # Coreset subsampling — randomly keep a fraction of patches
        # Why not keep all? For 200 images at 256×256 with layer2 at 32×32,
        # you'd have 200 × 32 × 32 = 200,000+ patches. KNN search over
        # that at inference time would be too slow. We subsample to keep
        # it fast while retaining good coverage of the normal space.


        ## Random sampling of patches
        n_keep = max(1, int(len(all_patches) * self.coreset_ratio))
        indices = np.random.choice(len(all_patches), n_keep, replace=False)
        self.memory_bank = all_patches[indices]


        ## greedy sampling of patches for coreset
        # n_keep = max(1, int(len(all_patches) * self.coreset_ratio))
        # self.memory_bank = self._greedy_coreset(all_patches, n_keep)
        print(f"  Memory bank size after coreset sampling: {self.memory_bank.shape[0]:,}")

        # Fit the nearest neighbour index
        # Why sklearn's NearestNeighbors? It's battle-tested, supports
        # different distance metrics, and is fast enough for our memory
        # bank size. For very large scale you'd use FAISS instead.
        self.knn = NearestNeighbors(n_neighbors=self.n_neighbors, metric="euclidean", n_jobs=-1)
        self.knn.fit(self.memory_bank)
        print("  KNN index built. Model ready.")

    def predict(self, dataloader):
        """
        Score test images. Returns image-level anomaly scores and
        pixel-level anomaly maps.
        """
        image_scores = []
        anomaly_maps = []

        for batch in dataloader:
            images = batch["image"]
            patches, (B, H, W) = self._extract_features(images)
            patches = self._normalize(patches) 

            # Find distances to nearest neighbours in memory bank
            # Shape of distances: [B*H*W, n_neighbors]
            distances, _ = self.knn.kneighbors(patches)

            # Take the mean distance across neighbours for robustness
            patch_scores = distances.mean(axis=1)  # [B*H*W]

            # Reshape back to spatial map per image
            patch_scores = patch_scores.reshape(B, H, W)

            for i in range(B):
                score_map = patch_scores[i]  # [H, W]

                # Image-level score = max anomaly score across all patches
                # Why max and not mean? A single highly anomalous patch
                # IS a defect. Averaging would dilute it.
                image_scores.append(float(score_map.max()))
                anomaly_maps.append(score_map)

        return np.array(image_scores), anomaly_maps
    

    def _normalize(self, features):
        """
        L2 normalise each patch feature vector to unit length.
        
        Why: without this, KNN distances are dominated by large-magnitude
        dimensions rather than actual visual similarity. Normalising puts
        all patches on a unit sphere so distances reflect direction only —
        which is what we actually want when comparing "does this patch
        look like a normal patch?"
        """
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-10, a_max=None)  # avoid division by zero
        return features / norms
    

    def _greedy_coreset(self, features, n_samples):
        """
        Greedy coreset selection — picks patches that maximally cover
        the feature space rather than sampling randomly.
        
        Why this beats random sampling: random sampling might cluster
        samples in dense regions (common normal patches) and under-represent
        rare-but-normal patches. If a rare normal patch isn't covered,
        the model treats it as anomalous. Greedy selection spreads samples
        evenly across the space.
        
        Tradeoff: slower to compute than random sampling, but only happens
        once at training time, not at inference time.
        """
        print("  Running greedy coreset selection...")
        selected = [np.random.randint(0, len(features))]
        
        # Distance from each point to its nearest selected point
        min_distances = np.full(len(features), np.inf)
        
        for i in range(n_samples - 1):
            # Update distances based on the last selected point
            last = features[selected[-1]]
            dists = np.linalg.norm(features - last, axis=1)
            min_distances = np.minimum(min_distances, dists)
            
            # Pick the point furthest from all selected points
            selected.append(int(np.argmax(min_distances)))
            
            if (i + 1) % 500 == 0:
                print(f"    Selected {i+1}/{n_samples} coreset points...")
        
        return features[selected]
    

    def save(self, path):
        """
        Save the memory bank and fitted KNN index to disk.
        
        Why joblib and not pickle? joblib is optimised for large numpy
        arrays — it uses memory-mapped files internally, so saving and
        loading a 40,000 × 512 float32 array is significantly faster
        than pickle. It's the standard choice for sklearn objects too.
        
        Why not save the backbone? It's always loaded fresh from torchvision
        with fixed ImageNet weights. Saving it would add ~200MB to every
        model file for zero benefit.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "memory_bank": self.memory_bank,
            "knn":         self.knn,
        }, path)
        print(f"  Model saved to {path}")

    @classmethod
    def load(cls, path, device="cpu"):
        """
        Load a saved PatchCore model.
        
        Why a classmethod? Because we need to create a new PatchCore
        instance and populate it with saved state — we can't call this
        on an existing instance since one doesn't exist yet. It's the
        standard Python pattern for alternative constructors.
        """
        data = joblib.load(path)
        
        # Create a fresh instance — this rebuilds the backbone and hooks
        model = cls(device=device)
        
        # Restore the saved state
        model.memory_bank = data["memory_bank"]
        model.knn         = data["knn"]
        
        print(f"  Model loaded from {path}")
        return model
        
