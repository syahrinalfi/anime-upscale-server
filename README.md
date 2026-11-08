# Anime upscale server

A small LAN-only HTTP service that upscales anime and manga images 2× on an NVIDIA GPU. Send it an image and get back a sharper WEBP twice the size. I built it as a backend for an Android manga reader, so a phone can hand each page to a desktop GPU instead of upscaling on-device.

It runs in Docker with [FastAPI](https://fastapi.tiangolo.com/) and [spandrel](https://github.com/chaiNNer-org/spandrel), and serves two models, picked per request:

| Model | Key | Architecture | Speed (warm, end-to-end) | Notes |
|---|---|---|---|---|
| [HFA2k LUDVAE SPAN](https://openmodeldb.info/models/2x-HFA2k-LUDVAE-SPAN) by Helaman | `hfa2k` (default) | SPAN, 2× | ~0.7–0.95 s | Fast tier. Trained on anime illustrations with realistic compression, noise and blur. |
| [AnimeSharpV4 Fast](https://openmodeldb.info/models/2x-AnimeSharpV4-Fast-RCAN-PU) by Kim2091 | `animesharp` | RCAN + PixelUnshuffle, 2× | ~1.7–2.0 s | Sharper tier: cleaner lines and no color fringing, at about twice the latency. |

Times are for a 1060×1500 page on an RTX 3060 Laptop GPU (6 GB). Rejected during benchmarking: a 4× ESRGAN model (about 9 s per page) and the full, non-"Fast" AnimeSharpV4 (about 4.8 s per page).

**Requirements:** an NVIDIA GPU with 6 GB VRAM or more, Docker with GPU support (Docker Desktop on Windows/WSL2, or `nvidia-container-toolkit` on Linux), and the client on the same Wi-Fi/LAN.

## 1. Get the model weights

The weights aren't included in this repo. Download both:

```sh
mkdir -p models
curl -L -o models/2x-HFA2k-LUDVAE-SPAN.pth \
  "https://drive.google.com/uc?export=download&confirm=1&id=1_0X8xXtrtGOv0WAVlIxT_29GQOnmxBJF"
curl -L -o models/2x-AnimeSharpV4-Fast-RCAN-PU.safetensors \
  "https://github.com/Kim2091/Kim2091-Models/releases/download/2x-AnimeSharpV4/2x-AnimeSharpV4_Fast_RCAN_PU.safetensors"
```

You should end up with `models/2x-HFA2k-LUDVAE-SPAN.pth` (about 8.5 MB) and `models/2x-AnimeSharpV4-Fast-RCAN-PU.safetensors` (about 30 MB). The server loads every model in `MODEL_CONFIGS` at startup, so if you want only one, remove the other's entry from `app/inference.py`.

## 2. Set a shared token

```sh
cp .env.example .env
# edit .env and set UPSCALE_TOKEN to any random string
```

This is a simple shared-secret check, not real authentication. It's fine on a home LAN, but don't expose the server to the internet. Leaving it blank disables the check.

## 3. Start the server

```sh
docker compose up -d
docker compose logs -f   # wait for "Uvicorn running on http://0.0.0.0:8000"
```

Check it:

```sh
curl http://localhost:8000/health
# {"status":"ok","models":["hfa2k","animesharp"]}
```

If a model is missing from that list, its weight file wasn't downloaded or is at the wrong path. `docker compose logs` shows a load error naming the file.

## 4. Use the API

`POST /upscale` with the raw image bytes as the request body. Any format Pillow can read works. The response is a 2× WEBP (quality 90).

```sh
curl -X POST "http://localhost:8000/upscale?model=hfa2k" \
  -H "X-Upscale-Token: $UPSCALE_TOKEN" \
  --data-binary @page.jpg -o page-2x.webp
```

- `model`: `hfa2k` (default) or `animesharp`.
- `X-Upscale-Token`: only needed if `UPSCALE_TOKEN` is set.
- Errors: `400` for an unknown model or an image that can't be decoded, `401` for a wrong token, `413` for a request over the size limits below.

**Size limits.** Requests over 20 MB or images over 2.5 megapixels (for example 1300×1900) are rejected. On a 6 GB card, larger inputs overflow GPU memory into system memory and take a minute or more while blocking other requests. A typical 1060×1500 page is about 1.6 megapixels. On a GPU with more VRAM, raise the limits with the `MAX_PIXELS` and `MAX_BODY_BYTES` environment variables in `docker-compose.yml`.

Requests are processed one at a time on the GPU. The first request per model after startup pays a one-time cuDNN warm-up of about 1.5–2 s.

## 5. Reach it from another device

Other devices need the server machine's LAN address, not `localhost`.

- **Linux/macOS:** find the IPv4 address of your Wi-Fi or Ethernet interface (`ip addr` or `ifconfig`), something like `192.168.x.x`, and use it with port 8000.
- **Windows, Docker not on WSL2:** `ipconfig`'s Wi-Fi/Ethernet IPv4 address works directly.
- **Windows with Docker Desktop's WSL2 backend** (the common case, and how this was built): WSL2 has its own NAT'd network, so the port is only reachable from the Windows host. Run this once in an **elevated PowerShell on Windows**, not inside WSL:
  ```powershell
  # Find WSL's current IP first (run inside WSL): ip addr show eth0 | grep 'inet '
  netsh interface portproxy add v4tov4 listenport=8000 listenaddress=0.0.0.0 connectport=8000 connectaddress=<WSL_IP>
  netsh advfirewall firewall add rule name="WSL Upscale Server 8000" dir=in action=allow protocol=TCP localport=8000
  ```
  Then use the Windows host's own LAN IP. WSL's internal IP can change after a reboot or network change, so if the server stops being reachable, redo the `portproxy add` line with the new WSL IP. `netsh interface portproxy show all` lists the current rules.

## Troubleshooting

- **`/health` works on the server machine but not from another device:** almost always LAN reachability. Check both devices are on the same Wi-Fi (not a guest network, which usually isolates clients). On Windows+WSL2, a missing or stale `portproxy` rule looks exactly like this.
- **GPU not detected, or a CUDA/driver error at startup:** first confirm `docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi` works on its own. That separates a Docker/driver problem from this app. If it worked before and suddenly doesn't, even outside Docker, a laptop may have switched to integrated graphics after a power change; a full restart fixes it.
- **Connection reset, or `docker ps` shows `8000/tcp` with no `0.0.0.0:8000->` mapping:** something on the host already uses port 8000, often a leftover `portproxy` rule from step 5. Docker Desktop then can't publish the port, without an obvious error. Check with `netsh interface portproxy show all`, then delete the stale rule (`netsh interface portproxy delete v4tov4 listenport=8000 listenaddress=0.0.0.0`) or change the host port in `docker-compose.yml`, for example `"8010:8000"`.
- **401 Unauthorized:** the client's token doesn't match `UPSCALE_TOKEN` in `.env`, or one is set and the other is blank.
- **`UnsupportedDtypeError: ... does not support half precision (fp16)`** (only if you add a model): spandrel's `.half()` wrapper conservatively blocks some architectures, such as RCAN, that run fine in fp16. `load_models()` calls `.half()` on the underlying `model.model` instead; do the same for a new architecture that hits this.
- **A model you added needs more VRAM than one pass allows:** `upscale()` supports tiling through the `tile` and `pad` keys in `MODEL_CONFIGS` (for example `384` and `16`). Both bundled models fit a full 1060×1500 page untiled (about 2.9 GB and 2.1 GB peak VRAM on a 6 GB card), so tiling is currently unused.

## License

The code is under the [MIT License](LICENSE). The model weights are not part of this repo and keep their own licenses:

- **HFA2k LUDVAE SPAN** by Helaman, trained on musl's HFA2k_LUDVAE dataset: [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).
- **AnimeSharpV4 Fast RCAN PU** by Kim2091: [CC-BY-NC-SA-4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Non-commercial use only, and derivatives must use the same license.
