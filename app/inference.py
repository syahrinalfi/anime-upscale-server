import os
import threading

import torch
from spandrel import ModelLoader

# Two models, selected per request by name (see main.py's /upscale?model=...):
#  - "hfa2k": 2x SPAN (HFA2k_LUDVAE, CC-BY-4.0) - trained on still anime illustrations with
#    realistic degradations (compression/noise/blur) - a good content match for
#    manga/manhwa pages. Fits a full 1060x1500 page in ~2.9GB VRAM with no tiling needed.
#    ~200ms warm steady-state pure inference - still comfortably under a 1s round trip once
#    encode/decode/HTTP overhead is added. Replaced an earlier 4x ESRGAN (IllustrationJaNai)
#    entry that was correctness-fine but too slow in practice (~6.8-9s/page) for daily use.
#  - "animesharp": 2x RCAN PixelUnshuffle (AnimeSharpV4 Fast, CC-BY-NC-SA-4.0) - sharper
#    lines at roughly twice the latency.
MODEL_CONFIGS = {
    "hfa2k": {
        "path": os.environ.get("HFA2K_MODEL_PATH", "/models/2x-HFA2k-LUDVAE-SPAN.pth"),
        "tile": None,  # ~2.9GB peak VRAM at full 1060x1500 res, comfortably under 6GB
        "pad": 0,
    },
    "animesharp": {
        "path": os.environ.get(
            "ANIMESHARP_MODEL_PATH", "/models/2x-AnimeSharpV4-Fast-RCAN-PU.safetensors",
        ),
        "tile": None,  # ~2.1GB peak VRAM at full 1060x1500 res, comfortably under 6GB
        "pad": 0,
    },
}

_lock = threading.Lock()
_models: dict[str, object] = {}


def _tile_bounds(total: int, tile: int) -> list[tuple[int, int]]:
    bounds = []
    i = 0
    while i < total:
        bounds.append((i, min(tile, total - i)))
        i += tile
    return bounds


def load_models():
    torch.backends.cudnn.benchmark = True
    for name, cfg in MODEL_CONFIGS.items():
        model = ModelLoader().load_from_file(cfg["path"])
        model = model.to("cuda").eval()
        # spandrel's own .half() wrapper raises UnsupportedDtypeError for some
        # architectures (e.g. RCAN/AnimeSharpV4) as a conservative guard, but calling
        # .half() on the underlying raw nn.Module directly works fine and produces sane
        # output (confirmed via manual benchmarking) - bypass the wrapper's guard here.
        model.model.half()
        _models[name] = model


def get_scale(model_name: str) -> int:
    return _models[model_name].scale


def available_models() -> list[str]:
    return list(_models.keys())


def upscale(model_name: str, rgb_tensor: torch.Tensor) -> torch.Tensor:
    """
    rgb_tensor: float16 CUDA tensor, shape (1, 3, H, W), values in [0, 1].
    Returns a float16 CUDA tensor, shape (1, 3, H*scale, W*scale).

    Serialized behind _lock since spandrel/ncnn-style single-GPU inference isn't safe
    for concurrent calls on one model instance - shared across both models since they'd
    otherwise contend for the same GPU anyway.
    """
    model = _models[model_name]
    tile = MODEL_CONFIGS[model_name]["tile"]
    pad = MODEL_CONFIGS[model_name]["pad"]

    with _lock:
        scale = model.scale

        if tile is None:
            with torch.no_grad():
                return model(rgb_tensor)

        _, _, h, w = rgb_tensor.shape
        xb = _tile_bounds(w, tile)
        yb = _tile_bounds(h, tile)

        out = torch.zeros(1, 3, h * scale, w * scale, device="cuda", dtype=torch.float16)

        with torch.no_grad():
            for ty, th in yb:
                for tx, tw in xb:
                    y0, y1 = max(0, ty - pad), min(h, ty + th + pad)
                    x0, x1 = max(0, tx - pad), min(w, tx + tw + pad)
                    tile_in = rgb_tensor[:, :, y0:y1, x0:x1]
                    tile_out = model(tile_in)
                    oy0, ox0 = (ty - y0) * scale, (tx - x0) * scale
                    oy1, ox1 = oy0 + th * scale, ox0 + tw * scale
                    out[:, :, ty * scale:(ty + th) * scale, tx * scale:(tx + tw) * scale] = (
                        tile_out[:, :, oy0:oy1, ox0:ox1]
                    )
        return out
