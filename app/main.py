import io
import os

import numpy as np
import torch
from fastapi import FastAPI, Header, HTTPException, Request, Response
from PIL import Image

from . import inference

AUTH_TOKEN = os.environ.get("UPSCALE_TOKEN", "")
# Request limits, so one oversized upload can't exhaust memory or tie up the GPU. On a 6GB
# card, inputs above ~2.5 megapixels overflow VRAM into system memory and slow from ~2s to
# a minute or more. Raise MAX_PIXELS on a GPU with more VRAM.
MAX_BODY_BYTES = int(os.environ.get("MAX_BODY_BYTES", 20 * 1024 * 1024))
MAX_PIXELS = int(os.environ.get("MAX_PIXELS", 2_500_000))

app = FastAPI(title="anime-upscale-server")


@app.on_event("startup")
def _startup():
    inference.load_models()


@app.get("/health")
def health():
    return {"status": "ok", "models": inference.available_models()}


def _check_token(x_upscale_token: str | None):
    # Shared-secret check only - this is a LAN-only personal tool, not meant to be
    # internet-facing.
    if not AUTH_TOKEN:
        return
    if x_upscale_token != AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="invalid token")


@app.post("/upscale")
async def upscale(
    request: Request,
    model: str = "hfa2k",
    x_upscale_token: str | None = Header(default=None),
):
    _check_token(x_upscale_token)

    if model not in inference.available_models():
        raise HTTPException(status_code=400, detail=f"unknown model '{model}', available: {inference.available_models()}")

    too_large = HTTPException(status_code=413, detail=f"request body over {MAX_BODY_BYTES} bytes")
    if int(request.headers.get("content-length") or 0) > MAX_BODY_BYTES:
        raise too_large
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_BODY_BYTES:
            raise too_large

    try:
        img = Image.open(io.BytesIO(body))  # lazy: reads the header only, no pixel decode yet
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"could not decode image: {e}")
    width, height = img.size
    if width * height > MAX_PIXELS:
        raise HTTPException(
            status_code=413,
            detail=f"image is {width}x{height} ({width * height} pixels), over the {MAX_PIXELS}-pixel limit",
        )
    try:
        img = img.convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"could not decode image: {e}")

    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to("cuda").half()

    out_tensor = inference.upscale(model, tensor)

    out_arr = out_tensor.float().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    out_img = Image.fromarray((out_arr * 255.0 + 0.5).astype("uint8"), mode="RGB")

    buf = io.BytesIO()
    # Lossy WEBP at quality ~90 - this is a client-side display cache, not an archival
    # copy, so pixel-perfect losslessness isn't needed here. Deliberately NOT using
    # lossless WEBP at quality=100 - in testing that cost 5-27 SECONDS to encode a much
    # smaller image (the "quality" param for lossless WEBP is a compression-effort knob,
    # not fidelity).
    out_img.save(buf, format="WEBP", quality=90)
    return Response(content=buf.getvalue(), media_type="image/webp")
