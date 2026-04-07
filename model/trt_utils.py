# Copyright (c) 2025-2026, ETH Zurich (Robotic Systems Lab) & NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0
"""Compile the SAM2 Hiera image encoder and mask decoder with TensorRT via torch-tensorrt.

Approach follows the official PyTorch-TensorRT SAM2 tutorial:
  https://github.com/pytorch/TensorRT/blob/main/examples/dynamo/torch_export_sam2.py

Key insight: the stock SAM2 source has several patterns that cause torch.export to fail
or produce a broken TRT graph even with strict=False:

Encoder patches (applied to image_encoder):
1. ``FpnNeck.forward`` casts intermediate features to float32 for interpolation.
2. ``LayerNorm2d.forward`` uses manual mean/var; replaced with ``F.layer_norm``.
3. ``PositionEmbeddingRandom.forward`` creates a float32 grid.

Decoder patch (applied to sam_mask_decoder):
4. ``Attention.forward`` in ``transformer.py`` wraps SDPA in a non-traceable
   ``torch.backends.cuda.sdp_kernel`` context manager.

All patches are applied at runtime by monkey-patching live module instances.
No SAM2 source files are modified.

Engines are cached to ``_TRT_CACHE_DIR`` so first-run compilation (~5-10 min each)
only happens once per GPU architecture.
"""

import os
import types
from typing import Tuple

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
    """Apply all patches required for torch.export + TRT compilation (encoder)."""
    n_ln = _patch_layer_norm_2d(image_encoder)
    n_fpn = _patch_fpn_neck(image_encoder)
    n_pe = _patch_pos_embed_random(image_encoder)
    print(f"[TRT] Encoder patches applied: LayerNorm2d×{n_ln}, FpnNeck×{n_fpn}, PosEmbedRandom×{n_pe}")


# ---------------------------------------------------------------------------
# Decoder patch — removes torch.backends.cuda.sdp_kernel context manager
# ---------------------------------------------------------------------------

def _attention_forward_patched(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Attention.forward without the non-traceable sdp_kernel context manager."""
    q = self.q_proj(q)
    k = self.k_proj(k)
    v = self.v_proj(v)
    q = self._separate_heads(q, self.num_heads)
    k = self._separate_heads(k, self.num_heads)
    v = self._separate_heads(v, self.num_heads)
    dropout_p = self.dropout_p if self.training else 0.0
    # Drop the non-traceable context manager; PyTorch will auto-select the best kernel.
    out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
    out = self._recombine_heads(out)
    out = self.out_proj(out)
    return out


def _patch_attention(module: nn.Module) -> int:
    """Patch base Attention modules in the decoder (not RoPEAttention)."""
    try:
        from model.segment_anything_2.sam2.modeling.sam.transformer import Attention, RoPEAttention
    except ImportError:
        try:
            from sam2.modeling.sam.transformer import Attention, RoPEAttention
        except ImportError:
            return 0

    patched = 0
    for mod in module.modules():
        # Only patch the base Attention class, not RoPEAttention (used in image encoder, already handled)
        if type(mod) is Attention:
            mod.forward = types.MethodType(_attention_forward_patched, mod)
            patched += 1
    return patched


# ---------------------------------------------------------------------------
# Decoder wrapper — fixed API for torch.export
# ---------------------------------------------------------------------------

class _DecoderWrapper(nn.Module):
    """Thin wrapper around SAM2 MaskDecoder for TRT export.

    Bakes in:
    - ``multimask_output=False`` and ``repeat_image=True``
    - The no-mask dense embedding (no PromptEncoder call needed)
    - high_res_features as two separate tensor args instead of a list

    The prompt encoder call is bypassed entirely: when only text_embeds are
    used (points/boxes/masks=None), sparse_embeddings == text_embeds and
    dense_embeddings == no_mask_embed expanded over the batch.
    """

    def __init__(self, decoder: nn.Module, no_mask_embed_weight: torch.Tensor) -> None:
        super().__init__()
        self.decoder = decoder
        # [1, 256, 1, 1] — expanded to [N, 256, 64, 64] at runtime
        self.register_buffer("no_mask_embed", no_mask_embed_weight.reshape(1, -1, 1, 1))

    def forward(
        self,
        image_embeddings: torch.Tensor,  # [1, 256, 64, 64]
        image_pe: torch.Tensor,           # [1, 256, 64, 64]
        sparse_embeddings: torch.Tensor,  # [N, T, 256]
        high_res_s0: torch.Tensor,        # [1, 32, 256, 256]
        high_res_s1: torch.Tensor,        # [1, 64, 128, 128]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        N = sparse_embeddings.shape[0]
        dense_embeddings = self.no_mask_embed.expand(
            N, -1, image_embeddings.shape[2], image_embeddings.shape[3]
        )
        masks, iou_pred, _, _ = self.decoder.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            repeat_image=True,
            high_res_features=[high_res_s0, high_res_s1],
        )
        # Return single-mask slice (multimask_output=False path)
        return masks[:, 0:1, :, :], iou_pred[:, 0:1]


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


class _TRTDecoderAdapter(nn.Module):
    """Drop-in ``nn.Module`` replacement for ``sam_mask_decoder``.

    Accepts the original ``MaskDecoder.forward`` keyword signature and internally
    calls the TRT ``_DecoderWrapper``.

    ``SAM2Base.forward_image()`` calls ``self.sam_mask_decoder.conv_s0 / conv_s1``
    directly to project FPN backbone features.  These convolutions are preserved
    as submodules here so that attribute access still works after the swap.
    """

    def __init__(
        self,
        trt_wrapper: nn.Module,
        conv_s0: nn.Module | None = None,
        conv_s1: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.trt_wrapper = trt_wrapper
        # Keep the projection convolutions that SAM2Base.forward_image() calls directly.
        if conv_s0 is not None:
            self.conv_s0 = conv_s0
        if conv_s1 is not None:
            self.conv_s1 = conv_s1

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,  # ignored — baked into TRT wrapper
        multimask_output: bool = False,          # ignored — baked as False
        repeat_image: bool = True,               # ignored — baked as True
        high_res_features=None,
    ):
        low_res_masks, iou_pred = self.trt_wrapper(
            image_embeddings,
            image_pe,
            sparse_prompt_embeddings,
            high_res_features[0],
            high_res_features[1],
        )
        # Return 4-tuple matching the original MaskDecoder.forward() signature.
        # Downstream code in open_world_sam2.py only uses the first two outputs.
        return low_res_masks, iou_pred, None, None


def get_trt_decoder(
    decoder: nn.Module,
    no_mask_embed_weight: torch.Tensor,
    dtype: torch.dtype,
    device: str = "cuda:0",
    max_n_prompts: int = 400,
) -> nn.Module:
    """Return a TRT-compiled SAM2 mask decoder wrapper, loading from cache if available.

    The wrapper bypasses the PromptEncoder (handles only the text-embeddings-only
    path used by OWSAM) and compiles with dynamic batch size so it works for any
    vocabulary size up to ``max_n_prompts``.

    Args:
        decoder: The SAM2 ``MaskDecoder`` module.
        no_mask_embed_weight: ``sam_prompt_encoder.no_mask_embed.weight`` tensor.
        dtype: Compute dtype (e.g. ``torch.bfloat16``).
        device: CUDA device string.
        max_n_prompts: Maximum number of prompts (vocab_size × num_tokens). Used to
            set the upper bound for TRT dynamic-shape optimisation.

    Returns:
        TRT-compiled ``_DecoderWrapper``, or original wrapper if compilation fails.
    """
    try:
        import torch_tensorrt
    except ImportError:
        print("[TRT] torch-tensorrt not installed, decoder will run in eager mode")
        wrapper = _DecoderWrapper(decoder, no_mask_embed_weight)
        wrapper = wrapper.to(device=device, dtype=dtype).eval()
        _patch_attention(wrapper)
        return wrapper

    os.makedirs(_TRT_CACHE_DIR, exist_ok=True)
    dtype_tag = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}.get(dtype, "fp32")
    sm = torch.cuda.get_device_capability(torch.device(device))
    trt_path = os.path.join(
        _TRT_CACHE_DIR,
        f"sam2_hiera_large_decoder_sm{sm[0]}{sm[1]}_{dtype_tag}.ep",
    )

    if os.path.exists(trt_path):
        print(f"[TRT] Loading cached TRT decoder from {trt_path}")
        try:
            loaded = torch.export.load(trt_path)
            return loaded.module()
        except Exception as e:
            print(f"[TRT] Failed to load cached decoder engine ({e}), recompiling...")

    print("[TRT] Compiling SAM2 mask decoder with TensorRT (first run ~5-10 min)...")

    wrapper = _DecoderWrapper(decoder, no_mask_embed_weight)
    wrapper = wrapper.to(device=device, dtype=dtype).eval()

    # Patch Attention modules to remove sdp_kernel context manager
    n_attn = _patch_attention(wrapper)
    print(f"[TRT] Decoder patches applied: Attention×{n_attn}")

    # Example inputs — use opt (typical) shapes for tracing
    # N = vocab_size × num_tokens; T = BEiT-3 visual tokens (fixed at 100 for 480×640 → 224×224)
    opt_n = min(100, max_n_prompts)
    T = 100  # BEiT-3 token count for the warmup image size; decoder is insensitive to this
    ex_image_emb    = torch.zeros(1,  256,  64,  64, dtype=dtype, device=device)
    ex_image_pe     = torch.zeros(1,  256,  64,  64, dtype=dtype, device=device)
    ex_sparse       = torch.zeros(opt_n, T, 256,     dtype=dtype, device=device)
    ex_hr_s0        = torch.zeros(1,  32, 256, 256,  dtype=dtype, device=device)
    ex_hr_s1        = torch.zeros(1,  64, 128, 128,  dtype=dtype, device=device)

    try:
        # Declare N as a dynamic dimension so torch.export does not freeze it as a constant.
        N_dim = torch.export.Dim("N", min=1, max=max_n_prompts)
        with torch.no_grad():
            exported = torch.export.export(
                wrapper,
                args=(ex_image_emb, ex_image_pe, ex_sparse, ex_hr_s0, ex_hr_s1),
                dynamic_shapes={
                    "image_embeddings": None,
                    "image_pe": None,
                    "sparse_embeddings": {0: N_dim},
                    "high_res_s0": None,
                    "high_res_s1": None,
                },
                strict=False,
            )

        trt_decoder = torch_tensorrt.dynamo.compile(
            exported,
            inputs=[
                # image_embeddings — fixed shape
                torch_tensorrt.Input(shape=[1, 256, 64, 64], dtype=dtype),
                # image_pe — fixed shape
                torch_tensorrt.Input(shape=[1, 256, 64, 64], dtype=dtype),
                # sparse_embeddings — dynamic N (vocab × tokens)
                torch_tensorrt.Input(
                    min_shape=[1,  T, 256],
                    opt_shape=[opt_n, T, 256],
                    max_shape=[max_n_prompts, T, 256],
                    dtype=dtype,
                ),
                # high_res_s0 — fixed shape
                torch_tensorrt.Input(shape=[1, 32, 256, 256], dtype=dtype),
                # high_res_s1 — fixed shape
                torch_tensorrt.Input(shape=[1, 64, 128, 128], dtype=dtype),
            ],
            enabled_precisions={dtype},
            truncate_double=True,
            device=torch.device(device),
            workspace_size=4 * 1024 ** 3,
            optimization_level=3,
            use_fp32_acc=True,
        )

        torch_tensorrt.save(trt_decoder, trt_path, inputs=[ex_image_emb, ex_image_pe, ex_sparse, ex_hr_s0, ex_hr_s1])
        print(f"[TRT] TRT decoder engine saved to {trt_path}")
        return trt_decoder.module()

    except Exception as e:
        import traceback
        print(f"[TRT] Decoder TRT compilation failed: {e}")
        traceback.print_exc()
        # Fall back to torch.compile (mode="default" avoids CUDA-graph FakeTensor issues)
        print("[TRT] Falling back to torch.compile for decoder")
        return torch.compile(wrapper, mode="default", fullgraph=False)
