"""
Load YOLOv5s pretrained backbone weights into our CSPDarknet.
Run ONCE on the server — generates a pretrained checkpoint for training.

Strategy: create a minimal stub package so pickle can deserialize yolov5s.pt
without needing the full yolov5 repo (no network, no clone, no pip install).
"""
import os, sys
from pathlib import Path

# ── Step 0: create stub package (if not already present) ──────────────────
_STUB_DIR = Path(__file__).resolve().parent / "_yolo_stub"
_STUB_DIR.mkdir(exist_ok=True)

(_STUB_DIR / "models").mkdir(exist_ok=True)
(_STUB_DIR / "models" / "__init__.py").touch(exist_ok=True)
(_STUB_DIR / "models" / "experimental.py").write_text("def attempt_load(*a,**k): pass\n")
(_STUB_DIR / "utils").mkdir(exist_ok=True)
(_STUB_DIR / "utils" / "__init__.py").touch(exist_ok=True)

# models/common.py  —  classes pickle will encounter
(_STUB_DIR / "models" / "common.py").write_text("""
import torch.nn as nn
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
class BottleneckCSP(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
class Concat(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
class Detect(nn.Module):
    def __init__(self, nc=80, anchors=()):
        super().__init__()
class GhostConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
class GhostBottleneck(nn.Module):
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
class DWConv(Conv):
    pass
class Focus(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
class Contract(nn.Module):
    def __init__(self, gain=2):
        super().__init__()
class Expand(nn.Module):
    def __init__(self, gain=2):
        super().__init__()
class TransformerLayer(nn.Module):
    def __init__(self, c1, c2, num_heads=8):
        super().__init__()
""")

# models/yolo.py — YOLOv5 puts Model, DetectionModel, Detect, Segment, Pose here
(_STUB_DIR / "models" / "yolo.py").write_text("""
import torch
import torch.nn as nn
class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):
        super().__init__()
class DetectionModel(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):
        super().__init__()
class ClassificationModel(nn.Module):
    def __init__(self, cfg=None, model=None, ch=3, nc=1000, cutoff=10):
        super().__init__()
class Ensemble(nn.ModuleList):
    pass
class Detect(nn.Module):
    stride: torch.Tensor
    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):
        super().__init__()
class Segment(Detect):
    pass
class Pose(Detect):
    pass
class DDetect(nn.Module):
    def __init__(self, nc=80, anchors=(), ch=()):
        super().__init__()
class DSegment(DDetect):
    pass
class DPose(DDetect):
    pass
""")

# utils/ stuff (some checkpoints reference it)
(_STUB_DIR / "utils" / "general.py").write_text("def check_version(*a,**k): pass\n")
(_STUB_DIR / "utils" / "loss.py").write_text("class FocalLoss: pass\n")
(_STUB_DIR / "utils" / "activations.py").write_text("class SiLU: pass\n")
(_STUB_DIR / "utils" / "metrics.py").write_text("def fitness(*a,**k): return 0\n")

sys.path.insert(0, str(_STUB_DIR))


# ── Step 1: load checkpoint ───────────────────────────────────────────────
import torch
import torch.nn as nn

from model_yolo import ExcavatorVLAYolo

_MODEL_CFG = dict(seq_len=8, img_size=224, hidden_dim=768, n_heads=12,
                   n_layers=6, ff_dim=3072, dropout=0.1, num_excavators=4)


def load_yolov5s(checkpoint_path="yolov5s.pt"):
    """Load the raw YOLOv5s checkpoint with stub modules."""
    print(f"Loading {checkpoint_path} ...")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    print(f"  keys: {list(ckpt.keys())}")

    # YOLOv5 saves the full model in ckpt['model']; extract state_dict
    if "model" in ckpt:
        model_obj = ckpt["model"]
        if hasattr(model_obj, "state_dict"):
            state = {k: v.float() for k, v in model_obj.state_dict().items()}
        else:
            state = {k: v.float() for k, v in model_obj.float().items()}
    else:
        state = {k: v.float() for k, v in ckpt.items()}

    print(f"  {len(state)} weight tensors extracted")
    return state


# ── Step 2: map YOLO keys → our CSPDarknet keys ───────────────────────────
def map_conv(yolo_prefix, our_prefix, yolo_state, our_state):
    """Map a ConvBNSiLU block: conv.weight, bn.weight/bias/running_mean/running_var, bn.num_batches_tracked"""
    loaded = 0
    for yk, ok in [
        (f"{yolo_prefix}.conv.weight", f"{our_prefix}.conv.weight"),
        (f"{yolo_prefix}.bn.weight", f"{our_prefix}.bn.weight"),
        (f"{yolo_prefix}.bn.bias", f"{our_prefix}.bn.bias"),
        (f"{yolo_prefix}.bn.running_mean", f"{our_prefix}.bn.running_mean"),
        (f"{yolo_prefix}.bn.running_var", f"{our_prefix}.bn.running_var"),
        (f"{yolo_prefix}.bn.num_batches_tracked", f"{our_prefix}.bn.num_batches_tracked"),
    ]:
        if yk in yolo_state and ok in our_state and yolo_state[yk].shape == our_state[ok].shape:
            our_state[ok].copy_(yolo_state[yk])
            loaded += 1
    return loaded


def map_csp(yolo_idx, csp_name, yolo_state, our_state):
    """Map a whole CSPLayer: cv1, cv2, cv3 (+ BN), internal blocks."""
    loaded = 0
    prefix_our = f"rgb_backbone.{csp_name}"
    prefix_yolo = f"model.{yolo_idx}"

    # cv1, cv2, cv3: three 1x1 convs
    for cv_num in [1, 2, 3]:
        loaded += map_conv(f"{prefix_yolo}.cv{cv_num}", f"{prefix_our}.cv{cv_num}",
                           yolo_state, our_state)

    # Internal blocks: model.{idx}.m.{b}.cv1 / cv2  →  our blocks.{b}
    for b in range(3):
        our_blk = f"{prefix_our}.blocks.{b}"
        yolo_blk = f"{prefix_yolo}.m.{b}"
        # Try cv1 (1x1) → our block's conv
        if f"{yolo_blk}.cv1.conv.weight" in yolo_state:
            loaded += map_conv(f"{yolo_blk}.cv1", our_blk, yolo_state, our_state)
        elif f"{yolo_blk}.cv2.conv.weight" in yolo_state:
            loaded += map_conv(f"{yolo_blk}.cv2", our_blk, yolo_state, our_state)

    return loaded


def map_down(yolo_idx, down_name, yolo_state, our_state):
    """Map a downsampling conv: model.{idx} → rgb_backbone.{down_name}"""
    return map_conv(f"model.{yolo_idx}", f"rgb_backbone.{down_name}", yolo_state, our_state)


# ── Step 3: main ──────────────────────────────────────────────────────────
def main():
    yolo_state = load_yolov5s("yolov5s.pt")

    model = ExcavatorVLAYolo(**_MODEL_CFG)
    our_state = model.state_dict()

    loaded = 0

    # Stem: model.0 → rgb_backbone.stem
    loaded += map_conv("model.0", "rgb_backbone.stem", yolo_state, our_state)

    # Stage 1: model.1 = s1_down, model.2 = s1_csp
    loaded += map_down(1, "s1_down", yolo_state, our_state)
    loaded += map_csp(2, "s1_csp", yolo_state, our_state)

    # Stage 2: model.3 = s2_down, model.4 = s2_csp
    loaded += map_down(3, "s2_down", yolo_state, our_state)
    loaded += map_csp(4, "s2_csp", yolo_state, our_state)

    # Stage 3: model.5 = s3_down, model.6 = s3_csp
    loaded += map_down(5, "s3_down", yolo_state, our_state)
    loaded += map_csp(6, "s3_csp", yolo_state, our_state)

    # Stage 4: model.7 = s4_down, model.9 = s4_csp  (YOLOv5s: model.8 = C3, model.9 = SPPF)
    loaded += map_down(7, "s4_down", yolo_state, our_state)
    # model.9 in YOLOv5s is SPPF (not CSP) — skip or try partial
    loaded += map_csp(9, "s4_csp", yolo_state, our_state)

    # Copy RGB backbone → Elevation backbone
    for key in list(our_state.keys()):
        if key.startswith("rgb_backbone."):
            elev_key = key.replace("rgb_backbone.", "elev_backbone.")
            if elev_key in our_state:
                our_state[elev_key].copy_(our_state[key])

    model.load_state_dict(our_state, strict=False)
    print(f"\n  {loaded} weight tensors mapped → RGB & Elevation backbones")

    # Save
    out_dir = Path("output/checkpoints")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "yolo_backbone_pretrained.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "description": "CSPDarknet backbone from YOLOv5s",
    }, out_path)
    print(f"Saved → {out_path}")
    print(f"\nUse: python vla_model/train_yolo.py --resume {out_path}")


if __name__ == "__main__":
    main()
