import argparse
import math
import warnings
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import time

from EF_LIC import model

warnings.filterwarnings("ignore")

FORCE_INDS = range(5)
PAD_MULTIPLE = 64


_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda t: t * 2.0 - 1.0),
])


def list_images(root: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    files = sorted(p for p in root.iterdir() if p.suffix.lower() in exts)
    if not files:
        raise RuntimeError(f"No images found in {root.resolve()}")
    return files


def load_image(path: Path, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return _to_tensor(img).unsqueeze(0).to(device, non_blocking=True)


def replicate_pad(x: torch.Tensor, h: int, w: int, p: int = PAD_MULTIPLE):
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    pad_h, pad_w = new_h - h, new_w - w
    if pad_h == 0 and pad_w == 0:
        return x
    return F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")


def mse01(x_hat: torch.Tensor, x: torch.Tensor) -> float:
    return F.mse_loss(
        ((x_hat + 1.0) * 0.5).clamp_(0, 1),
        ((x + 1.0) * 0.5).clamp_(0, 1),
    ).item()


def psnr_from_mse(mse: float) -> float:
    return -10.0 * math.log10(mse)


def _to_bits(t: torch.Tensor, k: int) -> np.ndarray:
    a = t.detach().reshape(-1).to("cpu", torch.long).numpy().astype(np.uint32, copy=False)
    return ((a[:, None] >> np.arange(k - 1, -1, -1, dtype=np.uint32)) & 1).astype(np.uint8).reshape(-1)


def _from_bits(bits: np.ndarray, shape, k: int, device: torch.device) -> torch.Tensor:
    n = int(np.prod(shape))
    b = bits.reshape(n, k).astype(np.uint32, copy=False)
    w = 1 << np.arange(k - 1, -1, -1, dtype=np.uint32)
    a = (b * w).sum(axis=1).astype(np.int64, copy=False)
    return torch.from_numpy(a).to(device=device).view(*shape)


def pack_inds(network, inds):
    n_e = tuple(int(x) for x in network.n_e)
    k = [(n - 1).bit_length() for n in n_e]
    items = [(inds["z_inds"], k[-1])] + [(inds["y_inds"][i], k[i]) for i in range(4)]
    meta = [(tuple(t.shape), bits) for t, bits in items]
    raw_bits = np.concatenate([_to_bits(t, bits) for t, bits in items])
    payload = np.packbits(raw_bits, bitorder="big").tobytes()
    total_valid_bits = int(sum(np.prod(shape) * bits for shape, bits in meta))
    return payload, meta, total_valid_bits


def unpack_inds(payload: bytes, meta, total_valid_bits: int, device: torch.device):
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="big")[:total_valid_bits]
    tensors, pos = [], 0
    for shape, k in meta:
        nbits = int(np.prod(shape)) * k
        tensors.append(_from_bits(bits[pos:pos + nbits], shape, k, device))
        pos += nbits
    return {"z_inds": tensors[0], "y_inds": tensors[1:]}


@torch.inference_mode()
def reconstruct_and_bpp(network, frame: torch.Tensor, force_ind: int):
    B, _, H, W = frame.shape
    padded = replicate_pad(frame, H, W)
    device = padded.device
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    inds = network.compress(padded, force_ind=force_ind)
    payload, meta, total_bits = pack_inds(network, inds)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    compress_time = (time.perf_counter() - start_time) * 1000
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    inds = unpack_inds(payload, meta, total_bits, device)
    x_hat = network.decompress(inds, force_ind=force_ind)[:, :, :H, :W]
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    decompress_time = (time.perf_counter() - start_time) * 1000
    
    bpp = len(payload) * 8.0 / float(B * H * W)
    return x_hat, bpp, compress_time, decompress_time


def load_checkpoint(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate EF-LIC on an image folder.")
    parser.add_argument("--kodak-dir", type=Path, default=Path("kodak"), help="Path to Kodak image directory.")
    parser.add_argument("--ckpt-path", type=Path, default=Path("ckpt") / "checkpoint.pth.tar", help="Path to model checkpoint.")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="PyTorch device.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    import lpips
    import DISTS_pytorch as dists

    lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()
    dists_fn = dists.DISTS().to(device).eval()

    images = list_images(args.kodak_dir)
    net = model().to(device).eval()
    net.load_state_dict(load_checkpoint(args.ckpt_path, device), strict=True)

    print(f"Device: {device}")
    print(f"Images: {len(images)} from {args.kodak_dir.resolve()}")
    print(f"Checkpoint: {args.ckpt_path}\n")

    for force_ind in FORCE_INDS:
        net.prepare_inference_(force_ind=force_ind)
        psnr_list, lpips_list, dists_list, bpp_list = [], [], [], []
        compress_times, decompress_times = [], []

        print(f"========== force_ind={force_ind} ==========")
        
        if len(images) > 0:
            warmup_frame = load_image(images[0], device)
            for _ in range(3):
                _ = reconstruct_and_bpp(net, warmup_frame, force_ind)
            if device.type == 'cuda':
                torch.cuda.synchronize()

        for idx, path in enumerate(images, 1):
            frame = load_image(path, device)
            x_hat, bpp, comp_t, decomp_t = reconstruct_and_bpp(net, frame, force_ind)

            mse = mse01(x_hat, frame)
            psnr = psnr_from_mse(mse)
            lp = lpips_fn(x_hat.clamp(-1, 1), frame.clamp(-1, 1)).mean().item()
            ds = dists_fn(
                ((x_hat + 1.0) * 0.5).clamp(0, 1),
                ((frame + 1.0) * 0.5).clamp(0, 1),
                require_grad=False,
            ).detach().mean().item()

            psnr_list.append(psnr)
            lpips_list.append(lp)
            dists_list.append(ds)
            bpp_list.append(bpp)
            compress_times.append(comp_t)
            decompress_times.append(decomp_t)

            print(f"[{idx:02d}/{len(images):02d}] {path.name:16s} | "
                  f"Enc: {comp_t:6.2f} ms | Dec: {decomp_t:6.2f} ms | "
                  f"PSNR={psnr:8.4f} | LPIPS={lp:8.5f} | DISTS={ds:8.5f} | BPP={bpp:8.5f}")

        avg_psnr = float(np.mean(psnr_list))
        avg_lpips = float(np.mean(lpips_list))
        avg_dists = float(np.mean(dists_list))
        avg_bpp = float(np.mean(bpp_list))
        avg_compress = float(np.mean(compress_times))
        avg_decompress = float(np.mean(decompress_times))

        print(f"---- AVG force_ind={force_ind} | "
              f"Enc={avg_compress:.2f} ms | Dec={avg_decompress:.2f} ms | "
              f"PSNR={avg_psnr:.4f} | LPIPS={avg_lpips:.5f} | "
              f"DISTS={avg_dists:.5f} | BPP={avg_bpp:.5f} ----\n")


if __name__ == "__main__":
    main()