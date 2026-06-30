"""Export ExcavatorVLA to ONNX format for deployment."""

import sys
import argparse
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vla_model.model import ExcavatorVLA
from vla_model.config import Config


def export_onnx(checkpoint_path: str, output_path: str, seq_len: int = 8,
                img_size: int = 160, opset: int = 14):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = Config()
    if "config" in ckpt:
        saved_config = ckpt["config"]
        if hasattr(saved_config, "hidden_dim"):
            config = saved_config

    print(f"Model config: hidden_dim={config.hidden_dim}, n_layers={config.n_layers}, "
          f"n_heads={config.n_heads}, img_size={img_size}")

    # Build model
    model = ExcavatorVLA(
        seq_len=seq_len,
        action_chunk=config.action_chunk,
        hidden_dim=config.hidden_dim,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        ff_dim=config.ff_dim,
        dropout=0.0,
        pretrained=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create dummy inputs
    dummy_rgb = torch.randn(1, seq_len, 3, img_size, img_size).to(device)
    dummy_elevation = torch.randn(1, seq_len, 3, img_size, img_size).to(device)
    dummy_qpos = torch.randn(1, seq_len, 4).to(device)
    dummy_excv_id = torch.tensor([0], dtype=torch.long).to(device)

    # Warm up
    with torch.no_grad():
        _ = model(dummy_rgb, dummy_elevation, dummy_qpos, dummy_excv_id)

    # Export
    print(f"Exporting to ONNX (opset={opset})...")
    torch.onnx.export(
        model,
        (dummy_rgb, dummy_elevation, dummy_qpos, dummy_excv_id),
        output_path,
        input_names=["rgb", "elevation", "qpos", "excavator_id"],
        output_names=["action"],
        dynamic_axes={
            "rgb": {0: "batch", 1: "seq_len"},
            "elevation": {0: "batch", 1: "seq_len"},
            "qpos": {0: "batch", 1: "seq_len"},
            "excavator_id": {0: "batch"},
            "action": {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )

    print(f"Exported to: {output_path}")

    # Verify
    import onnx
    import onnxruntime

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model check: PASSED")

    # Compare outputs
    ort_session = onnxruntime.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    with torch.no_grad():
        torch_out = model(dummy_rgb, dummy_elevation, dummy_qpos, dummy_excv_id).cpu().numpy()

    ort_inputs = {
        "rgb": dummy_rgb.cpu().numpy(),
        "elevation": dummy_elevation.cpu().numpy(),
        "qpos": dummy_qpos.cpu().numpy(),
        "excavator_id": dummy_excv_id.cpu().numpy(),
    }
    ort_out = ort_session.run(None, ort_inputs)[0]

    max_diff = np.max(np.abs(torch_out - ort_out))
    print(f"PyTorch vs ONNX max diff: {max_diff:.8f}")
    print(f"Output shape: {ort_out.shape}  (batch={ort_out.shape[0]}, chunk={ort_out.shape[1]}, dof={ort_out.shape[2]})")

    print("\nExport successful!")
    print(f"  Model: {output_path}")
    print(f"  Input:  rgb [B, T, 3, {img_size}, {img_size}] float32")
    print(f"          elevation [B, T, 3, {img_size}, {img_size}] float32")
    print(f"  Output: action [B, 1, 4] float32  (Boom, Arm, Bucket, Swing in rad)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="output/checkpoints/checkpoint_best.pt")
    parser.add_argument("--output", type=str, default="output/checkpoints/excavator_vla.onnx")
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=160)
    parser.add_argument("--opset", type=int, default=14)
    args = parser.parse_args()

    export_onnx(args.checkpoint, args.output, args.seq_len, args.img_size, args.opset)
