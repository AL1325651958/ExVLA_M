"""
Load YOLOv5s pretrained backbone weights into our CSPDarknet.
Run once on the server to generate a pretrained checkpoint, then use in training.
"""
import torch
import torch.nn as nn
from model_yolo import ExcavatorVLAYolo
import warnings
warnings.filterwarnings("ignore")

_MODEL_CFG = dict(seq_len=8, img_size=224, hidden_dim=768, n_heads=12,
                   n_layers=6, ff_dim=3072, dropout=0.1, num_excavators=4)


def load_yolo_backbone(checkpoint_path="yolov5s.pt"):
    """Download & load official YOLOv5s.pt, extract backbone state_dict."""

    print("Loading official YOLOv5s weights...")
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except FileNotFoundError:
        import urllib.request
        url = "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt"
        print(f"  Downloading {url} ...")
        urllib.request.urlretrieve(url, checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model" in ckpt:
        yolo_state = ckpt["model"].float().state_dict()
    else:
        yolo_state = ckpt.float().state_dict()

    # YOLOv5s key prefix — we need model.0 to model.9 (backbone only)
    print(f"  YOLO keys: {len(yolo_state)} total")
    return yolo_state


class BackboneLoader:
    """Map YOLOv5s backbone weights → our CSPDarknet keys."""

    _YOLO_TO_OUR = {
        # YOLOv5s stem [model.0] → our stem
        "model.0.conv.weight": "rgb_backbone.stem.conv.weight",
        "model.0.bn.weight": "rgb_backbone.stem.bn.weight",
        "model.0.bn.bias": "rgb_backbone.stem.bn.bias",
        "model.0.bn.running_mean": "rgb_backbone.stem.bn.running_mean",
        "model.0.bn.running_var": "rgb_backbone.stem.bn.running_var",
        "model.0.bn.num_batches_tracked": "rgb_backbone.stem.bn.num_batches_tracked",

        # Stage 1 down [model.1] → s1_down
        "model.1.conv.weight": "rgb_backbone.s1_down.conv.weight",
        "model.1.bn.weight": "rgb_backbone.s1_down.bn.weight",
        "model.1.bn.bias": "rgb_backbone.s1_down.bn.bias",
        "model.1.bn.running_mean": "rgb_backbone.s1_down.bn.running_mean",
        "model.1.bn.running_var": "rgb_backbone.s1_down.bn.running_var",
        "model.1.bn.num_batches_tracked": "rgb_backbone.s1_down.bn.num_batches_tracked",
    }

    _YOLO_CSP_PREFIXES = {
        "model.2": "s1_csp",     # index 2 → stage1 CSP
        "model.4": "s2_csp",     # index 4 → stage2 CSP
        "model.6": "s3_csp",     # index 6 → stage3 CSP
        "model.9": "s4_csp",     # index 9 → stage4 CSP
    }

    # Downsample convs between CSP blocks
    _YOLO_DOWN = {
        "model.3": "s2_down",   # index 3 → s2_down
        "model.5": "s3_down",   # index 5 → s3_down
        "model.7": "s4_down",   # index 7 → s4_down
    }

    def __init__(self, our_model):
        self.model = our_model
        self.state = self.model.state_dict()

    def _copy_tensor(self, our_key, yolo_state, yolo_key):
        if yolo_key in yolo_state and our_key in self.state:
            w = yolo_state[yolo_key]
            if self.state[our_key].shape == w.shape:
                self.state[our_key].copy_(w)
                return True
        return False

    def _copy_csp(self, csp_prefix, yolo_prefix, yolo_csp_idx, yolo_state):
        """Copy all weights from a YOLO CSP block to our CSPLayer."""
        # cv1, cv2, cv3: 1x1 convs
        for i_conv, name in [(0, "cv1"), (1, "cv2"), (2, "cv3")]:
            yk = f"model.{yolo_csp_idx}.cv{i_conv+1}.conv.weight"
            ok = f"rgb_backbone.{csp_prefix}.{name}.conv.weight"
            self._copy_tensor(ok, yolo_state, yk)
            for bn_attr in ["weight", "bias", "running_mean", "running_var", "num_batches_tracked"]:
                self._copy_tensor(ok.replace(".conv.weight", f".bn.{bn_attr}"),
                                  yolo_state, yk.replace(".conv.weight", f".bn.{bn_attr}"))

        # Internal blocks: YOLO has m.0.cv1, m.0.cv2 → our blocks.0
        for b_idx in range(3):  # max 3 internal blocks
            for c_idx, conv_name in [(1, "cv1"), (2, "cv2")]:
                yk = f"model.{yolo_csp_idx}.m.{b_idx}.cv{c_idx}.conv.weight"
                ok = f"rgb_backbone.{csp_prefix}.blocks.{b_idx}.conv.weight" if conv_name == "cv1" else None
                # YOLO internal blocks are BottleneckCSP → skip complex mapping for now
                # Just try simple conv → conv mapping
                ok_simple = f"rgb_backbone.{csp_prefix}.blocks.{b_idx}.conv.weight"
                self._copy_tensor(ok_simple, yolo_state, yk)

    def _copy_down(self, csp_prefix, yolo_idx, yolo_state):
        yk = f"model.{yolo_idx}.conv.weight"
        ok = f"rgb_backbone.{csp_prefix}.conv.weight"
        self._copy_tensor(ok, yolo_state, yk)
        for bn_attr in ["weight", "bias", "running_mean", "running_var", "num_batches_tracked"]:
            self._copy_tensor(ok.replace(".conv.weight", f".bn.{bn_attr}"),
                              yolo_state, yk.replace(".conv.weight", f".bn.{bn_attr}"))

    def load(self, yolo_state):
        loaded = 0
        # Stem + first stage
        for yk, ok in self._YOLO_TO_OUR.items():
            if self._copy_tensor(ok, yolo_state, yk):
                loaded += 1

        # CSP blocks
        for yolo_prefix, csp_name in self._YOLO_CSP_PREFIXES.items():
            yolo_idx = int(yolo_prefix.split(".")[1])
            self._copy_csp(csp_name, yolo_prefix, yolo_idx, yolo_state)

        # Downsample convs
        for yolo_prefix, down_name in self._YOLO_DOWN.items():
            yolo_idx = int(yolo_prefix.split(".")[1])
            self._copy_down(down_name, yolo_idx, yolo_state)

        # Copy RGB backbone to Elevation backbone
        for key in list(self.state.keys()):
            if key.startswith("rgb_backbone."):
                elev_key = key.replace("rgb_backbone.", "elev_backbone.")
                if elev_key in self.state:
                    self.state[elev_key].copy_(self.state[key])

        self.model.load_state_dict(self.state, strict=False)
        print(f"  Loaded {loaded} exact matches + CSP blocks → RGB & Elevation backbones")
        return self.model


def main():
    model = ExcavatorVLAYolo(**_MODEL_CFG)
    yolo_state = load_yolo_backbone("yolov5s.pt")
    loader = BackboneLoader(model)
    loader.load(yolo_state)

    out_path = "output/checkpoints/yolo_backbone_pretrained.pt"
    torch.save({"model_state_dict": model.state_dict(), "description": "CSPDarknet backbone from YOLOv5s"}, out_path)
    print(f"\nSaved pretrained backbone to: {out_path}")
    print("Use: python vla_model/train_yolo.py --resume output/checkpoints/yolo_backbone_pretrained.pt")


if __name__ == "__main__":
    main()
