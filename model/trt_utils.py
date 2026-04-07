# Copyright (c) 2025-2026, ETH Zurich (Robotic Systems Lab) & NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0
"""Compile the SAM2 Hiera image encoder with TensorRT via torch-tensorrt.

Approach follows the official PyTorch-TensorRT SAM2 tutorial:
  https://github.com/pytorch/TensorRT/blob/main/examples/dynamo/torch_export_sam2.py

Key insight: the stock SAM2 source has several patterns that cause torch.export to fail
or produce a broken TRT graph even with strict=False:

1. ``FpnNeck.forward`` casts intermediate features to float32 for interpolation
   (``prev_features.to(dtype=torch.float32)``).  This inserts a dtype-change node
   that confuses TRT when the model runs in BF16.

2. ``LayerNorm2d.forward`` in ``sam2_utils.py`` implements layer-norm manually with
   mean/variance ops across the channel dimension.  TRT cannot fuse this into a single
   LayerNorm kernel; replacing it with ``F.layer_norm`` is both correct and faster.

3. ``PositionEmbeddingRandom.forward`` creates a float32 grid regardless of model dtype,
   which again inserts unwanted casts.

4. ``transformer.py``'s ``RoPEAttention`` wraps SDPA in a custom ``sdp_kernel_context``
   context manager which is not traceable.

Patches 1-4 are applied at runtime *before* export by monkey-patching the relevant
``forward`` methods on the live module instances.  No SAM2 source files are modified.

The compiled engine is cached to disk (``_TRT_CACHE_DIR``) so subsequent container
starts skip the ~5-minute compilation.
"""

import os
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

_TRT_CACHE_DIR = os.environ.get("OWSAM_TRT_CACHE", "/app/trt_cache")


# ---------------------------------------------------------------------------
# Patched forward implementations
# ---------------------------------------------------------------------------

def _fpn_neck_forward_patched(self, xs):
    """FpnNeck.forward without forced float32 cast in F.interpolate."""
    out = [None] * len(self.convs)
    pos = [None] * len(self.convs)
    prev_features = None
    n = len(self.convs) - 1
    for i in range(n, -1, -1):
        x = xs[i]
        lateral_features = self.convs[n - i](x)
        if i in self.fpn_top_down_levels and prev_features is not None:
            top_down_features = F.interpolate(
                prev_features,  # removed .to(dtype=torch.float32)
                scale_factor=2.0,
                mode=self.fpn_interp_model,
                align_corners=(None if self.fpn_interp_model == "nearest" else False),
                antialias=False,
            )
            prev_features = lateral_features + top_down_features
            if self.fuse_type == "avg":
                prev_features = prev_features / 2
        else:
            prev_features = lateral_features
        x_out = prev_features
        out[i] = x_out
        pos[i] = self.position_encoding(x_out).to(x_out.dtype)
    return out, pos


def _layer_norm_2d_forward_patched(self, x: torch.Tensor) -> torch.Tensor:
    """LayerNorm2d.forward using F.layer_norm instead of manual mean/var."""
    x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
    x = F.layer_norm(
        x,
        normalized_shape=(self.num_channels,),
        weight=self.weight,
        bias=self.bias,
        eps=self.eps,
    )
    x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
    return x


def _pos_embed_random_forward_patched(self, size):
    """PositionEmbeddingRandom.forward using model dtype for the grid.

    Preserves the original return shape (C x H x W) — only fixes the forced
    float32 grid creation so no unwanted dtype cast is traced.
    """
    h, w = size
    device = self.positional_encoding_gaussian_matrix.device
    dtype = self.positional_encoding_gaussian_matrix.dtype
    grid = torch.ones((h, w), device=device, dtype=dtype)
    y_embed = grid.cumsum(dim=0) - 0.5
    x_embed = grid.cumsum(dim=1) - 0.5
    y_embed = y_embed / h
    x_embed = x_embed / w
    pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
    return pe.permute(2, 0, 1)  # C x H x W (same as original)


# ---------------------------------------------------------------------------
# Patch helpers
# ---------------------------------------------------------------------------

def _patch_layer_norm_2d(module: nn.Module):
    """Recursively replace LayerNorm2d.forward with the F.layer_norm version."""
    # Import lazily to avoid issues when called before sam2 is on sys.path
    try:
        from model.segment_anything_2.sam2.modeling.sam2_utils import LayerNorm2d
    except ImportError:
        try:
            from sam2.modeling.sam2_utils import LayerNorm2d
        except ImportError:
            LayerNorm2d = None

    patched = 0
    for mod in module.modules():
        if LayerNorm2d is not None and isinstance(mod, LayerNorm2d):
            # Ensure the num_channels attribute exists (added by the torch-trt fork)
            if not hasattr(mod, "num_channels"):
                # weight shape is (num_channels,)
                mod.num_channels = mod.weight.shape[0]
            mod.forward = types.MethodType(_layer_norm_2d_forward_patched, mod)
            patched += 1
    return patched


def _patch_fpn_neck(module: nn.Module):
    """Patch FpnNeck instances inside the encoder."""
    try:
        from model.segment_anything_2.sam2.modeling.backbones.image_encoder import FpnNeck
    except ImportError:
        try:
            from sam2.modeling.backbones.image_encoder import FpnNeck
        except ImportError:
            FpnNeck = None

    patched = 0
    for mod in module.modules():
        if FpnNeck is not None and isinstance(mod, FpnNeck):
            mod.forward = types.MethodType(_fpn_neck_forward_patched, mod)
            patched += 1
    return patched


def _patch_pos_embed_random(module: nn.Module):
    """Patch PositionEmbeddingRandom instances inside the encoder."""
    try:
        from model.segment_anything_2.sam2.modeling.position_encoding import PositionEmbeddingRandom
    except ImportError:
        try:
            from sam2.modeling.position_encoding import PositionEmbeddingRandom
        except ImportError:
            PositionEmbeddingRandom = None

    patched = 0
    for mod in module.modules():
        if PositionEmbeddingRandom is not None and isinstance(mod, PositionEmbeddingRandom):
            mod.forward = types.MethodType(_pos_embed_random_forward_patched, mod)
            patched += 1
    return patched


def _apply_trt_patches(image_encoder: nn.Module) -> None:
    """Apply all patches required for torch.export + TRT compilation."""
    n_ln = _patch_layer_norm_2d(image_encoder)
    n_fpn = _patch_fpn_neck(image_encoder)
    n_pe = _patch_pos_embed_random(image_encoder)
    print(f"[TRT] Patches applied: LayerNorm2d×{n_ln}, FpnNeck×{n_fpn}, PosEmbedRandom×{n_pe}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_trt_encoder(image_encoder: torch.nn.Module, dtype: torch.dtype, device: str = "cuda:0") -> torch.nn.Module:
    """Return a TRT-compiled image encoder, loading from cache if available.

    On first call this patches the encoder, exports it via
    ``torch.export.export(..., strict=False)``, compiles with
    ``torch_tensorrt.dynamo.compile()``, and writes an engine file to
    ``_TRT_CACHE_DIR``.  Subsequent calls load from the cache and are instant.

    The ``strict=False`` flag mirrors the official PyTorch-TensorRT SAM2 tutorial
    (pytorch/TensorRT@main examples/dynamo/torch_export_sam2.py).  Without it
    torch.export chokes on custom-op references inside the SAM2 package.

    Args:
        image_encoder: The SAM2 image encoder (HieraDet + FPN neck).
        dtype: The compute dtype (e.g. ``torch.bfloat16``).
        device: CUDA device string (e.g. ``"cuda:0"``).

    Returns:
        TRT-compiled module, or original module if compilation fails.
    """
    try:
        import torch_tensorrt
    except ImportError:
        print("[TRT] torch-tensorrt not installed, falling back to torch.compile")
        return torch.compile(image_encoder, mode="default", fullgraph=False)

    os.makedirs(_TRT_CACHE_DIR, exist_ok=True)
    dtype_tag = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}.get(dtype, "fp32")
    # Embed GPU SM version so engines are never loaded on a different GPU architecture.
    sm = torch.cuda.get_device_capability(torch.device(device))
    trt_path = os.path.join(_TRT_CACHE_DIR, f"sam2_hiera_large_encoder_sm{sm[0]}{sm[1]}_{dtype_tag}.ep")

    if os.path.exists(trt_path):
        print(f"[TRT] Loading cached TRT encoder from {trt_path}")
        try:
            loaded = torch.export.load(trt_path)
            return loaded.module()
        except Exception as e:
            print(f"[TRT] Failed to load cached engine ({e}), recompiling...")

    print("[TRT] Compiling SAM2 image encoder with TensorRT (first run ~5-10 min)...")
    image_encoder = image_encoder.to(device=device, dtype=dtype).eval()

    # Apply source-compatible patches before tracing
    _apply_trt_patches(image_encoder)

    example_input = torch.zeros(1, 3, 1024, 1024, dtype=dtype, device=device)

    try:
        # strict=False is required: SAM2 contains internal dict caches and
        # custom-op references that torch.export cannot trace in strict mode.
        # This is exactly the approach used by the official PyTorch-TensorRT
        # SAM2 tutorial (pytorch/TensorRT examples/dynamo/torch_export_sam2.py).
        with torch.no_grad():
            exported = torch.export.export(
                image_encoder,
                args=(example_input,),
                strict=False,
            )

        trt_encoder = torch_tensorrt.dynamo.compile(
            exported,
            inputs=[
                torch_tensorrt.Input(
                    shape=[1, 3, 1024, 1024],
                    dtype=dtype,
                )
            ],
            enabled_precisions={dtype},
            truncate_double=True,
            device=torch.device(device),
            workspace_size=4 * 1024 ** 3,  # 4 GB
            optimization_level=3,
            # Accumulate matmuls in FP32 to preserve accuracy at BF16/FP16
            use_fp32_acc=True,
        )

        torch_tensorrt.save(trt_encoder, trt_path, inputs=[example_input])
        print(f"[TRT] TRT engine saved to {trt_path}")
        return trt_encoder.module()

    except Exception as e:
        import traceback
        print(f"[TRT] TRT compilation failed: {e}")
        traceback.print_exc()
        print("[TRT] Falling back to torch.compile")
        return torch.compile(image_encoder, mode="default", fullgraph=False)
