"""Evaluate a released model on a released test dataset.

Example:
    python code/evaluate.py --model models/dilresnet.pt --data data/manually_defined_patterns.h5

Prints the mean and median relative error e = ||H_pred - H_ref||_2 / ||H_ref||_2 (in percent) over the file,
and optionally saves the predicted fields with --save.
"""

import argparse

import h5py
import numpy as np
import torch

from networks import load_model

H, W = 128, 64


def make_input(eps_r: np.ndarray) -> torch.Tensor:
    """eps_r (N, 128, 64) -> network input (N, 3, 128, 64): normalized permittivity and x, y coordinates."""
    lo = np.minimum(2.1, eps_r.min(axis=(1, 2), keepdims=True))
    hi = np.maximum(16.0, eps_r.max(axis=(1, 2), keepdims=True))
    eps_n = np.clip((eps_r - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    yy, xx = np.meshgrid(np.linspace(-1.0, 1.0, H), np.linspace(-1.0, 1.0, W), indexing="ij")
    n = eps_r.shape[0]
    x = np.stack([eps_n, np.broadcast_to(xx, (n, H, W)), np.broadcast_to(yy, (n, H, W))], axis=1)
    return torch.from_numpy(x.astype(np.float32))


@torch.no_grad()
def predict(model, ck, eps_r: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Complex H_z prediction (N, 128, 64) for permittivity maps eps_r (N, 128, 64)."""
    out = model(make_input(eps_r).to(device)).float().cpu().numpy()
    mu = np.asarray(ck["output_mean"])[None, :, None, None]
    sd = np.asarray(ck["output_std"])[None, :, None, None]
    out = out * sd + mu
    return out[:, 0] + 1j * out[:, 1]


def relative_error(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    d = (pred - ref).reshape(len(ref), -1)
    return np.linalg.norm(d, axis=1) / np.linalg.norm(ref.reshape(len(ref), -1), axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to a released .pt file")
    ap.add_argument("--data", required=True, help="path to a released .h5 file")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save", default=None, help="optional .npy path for the predicted complex fields")
    a = ap.parse_args()

    model, ck = load_model(a.model, a.device)
    errs, preds = [], []
    with h5py.File(a.data, "r") as f:
        n = f["eps_r"].shape[0]
        for i in range(0, n, a.batch):
            eps = f["eps_r"][i:i + a.batch]
            ref = f["hz"][i:i + a.batch]
            p = predict(model, ck, eps, a.device)
            errs.append(relative_error(p, ref))
            if a.save:
                preds.append(p.astype(np.complex64))
    e = 100 * np.concatenate(errs)
    print(f"{ck['paper_label']} on {a.data}: {n} samples, mean e = {e.mean():.3f} %, median e = {np.median(e):.3f} %")
    if a.save:
        np.save(a.save, np.concatenate(preds))


if __name__ == "__main__":
    main()
