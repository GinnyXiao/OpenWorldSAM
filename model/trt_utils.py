# Copyright (c) 2025-2026, ETH Zurich (Robotic Systems Lab) & NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0
"""Compile the SAM2 Hiera image encoder with TensorRT via torch-tensorrt."""

import os
import torch

_TRT_CACHE_DIR = os.environ.get("OWSAM_TRT_CACHE", "/app/trt_cache")


def get_trt_encoder(image_encoder: torch.nn.Module, dtype: torch.dtype, device: str = "cuda:0") -> torch.nn.Module:
    """Return a TRT-compiled image encoder, loading from cache if available.

    On first call this compiles the encoder (~5 min) and writes an engine file to
    ``_TRT_CACHE_DIR``.  Subsequent calls load from the cache and are instant.

    Args:
        image_encoder: The SAM2 image encoder (HieraDet + FPN neck).
        dtype: The compute dtype to use (e.g. ``torch.bfloat16``).
        device: CUDA device string.

    Returns:
        TRT-compiled module, or the original module if compilation fails.
    """
    try:
        import torch_tensorrt
    except ImportError:
        print("[TRT] torch-tensorrt not installed, skipping TRT compilation")
        return image_encoder

    os.makedirs(_TRT_CACHE_DIR, exist_ok=True)
    dtype_tag = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}.get(dtype, "fp32")
    trt_path = os.path.join(_TRT_CACHE_DIR, f"sam2_hiera_large_encoder_{dtype_tag}.ep")

    if os.path.exists(trt_path):
        print(f"[TRT] Loading cached TRT encoder from {trt_path}")
        try:
            trt_encoder = torch.export.load(trt_path)
            return trt_encoder.module()
        except Exception as e:
            print(f"[TRT] Failed to load cached engine ({e}), recompiling...")

    print("[TRT] Compiling SAM2 image encoder with TensorRT (first run ~5 min)...")
    image_encoder = image_encoder.to(device=device, dtype=dtype).eval()
    example_input = torch.zeros(1, 3, 1024, 1024, dtype=dtype, device=device)

    try:
        with torch.no_grad():
            exported = torch.export.export(
                image_encoder,
                args=(example_input,),
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
        )

        torch.export.save(trt_encoder, trt_path)
        print(f"[TRT] TRT engine saved to {trt_path}")
        return trt_encoder.module()

    except Exception as e:
        print(f"[TRT] TRT compilation failed ({e}), falling back to torch.compile")
        return torch.compile(image_encoder, mode="reduce-overhead", fullgraph=False)
