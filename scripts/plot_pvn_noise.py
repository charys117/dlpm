#!/usr/bin/env python3
"""
Sample PVN training noise using a DLPM config and plot its distribution.

Example:
    python scripts/plot_pvn_noise.py --config dlpm/configs/mnist_pvn.yml \\
        --num-samples 4096 --save-path artifacts/pvn_noise_hist.png
"""

import argparse
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "matplotlib is required for plotting. Install it with `pip install matplotlib`."
    ) from exc

import torch
import yaml

from bem.datasets.Distributions import gen_sas_pvn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize PVN noise used during DLPM training.")
    parser.add_argument(
        "--config",
        default="dlpm/configs/mnist_pvn.yml",
        help="Config file with training parameters (defaults to mnist_pvn).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4096,
        help="Number of epsilon samples to draw (each sample has full image shape).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=200,
        help="Number of bins to use in the histogram.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for sampling (cpu or cuda).",
    )
    parser.add_argument(
        "--save-path",
        default="artifacts/pvn_noise_hist.png",
        help="Where to save the histogram image.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def extract_pvn_params(cfg: dict) -> dict:
    method_key = cfg.get("method", "dlpm")
    method_cfg = cfg.get(method_key, {})
    train_cfg = cfg.get("training", {}).get(method_key, {})
    data_cfg = cfg.get("data", {})

    params = {
        "alpha": method_cfg.get("alpha"),
        "beta": method_cfg.get("beta", 0.0),
        "isotropic": method_cfg.get("isotropic", True),
        "clamp_eps": train_cfg.get("clamp_eps"),
        "clamp_a": train_cfg.get("clamp_a"),
        "clamp_v": train_cfg.get("clamp_v"),
        "channels": data_cfg.get("channels", 1),
        "image_size": data_cfg.get("image_size", 32),
    }
    missing = [k for k in ("alpha", "beta") if params.get(k) is None]
    if missing:
        raise ValueError(f"Missing required config fields: {missing}")
    return params


def build_sample_shape(params: dict, num_samples: int) -> tuple:
    return (num_samples, params["channels"], params["image_size"], params["image_size"])


def sample_noise(params: dict, num_samples: int, device: torch.device) -> torch.Tensor:
    shape = build_sample_shape(params, num_samples)
    with torch.no_grad():
        eps = gen_sas_pvn(
            alpha=params["alpha"],
            beta=params["beta"],
            size=shape,
            device=device,
            isotropic=params["isotropic"],
            clamp_eps=params["clamp_eps"],
            clamp_a=params["clamp_a"],
            clamp_v=params["clamp_v"],
        )
    return eps


def summarize_noise(eps: torch.Tensor) -> dict:
    # Compute summary stats (include skewness to quantify asymmetry)
    flat = eps.flatten().double()
    abs_flat = flat.abs()
    mean = flat.mean()
    std = flat.std()
    skew = ((flat - mean) ** 3).mean() / (std.pow(3) + 1e-12)
    summary = {
        "mean": mean.item(),
        "std": std.item(),
        "skewness": skew.item(),
        "min": flat.min().item(),
        "max": flat.max().item(),
        "abs_p90": torch.quantile(abs_flat, 0.90).item(),
        "abs_p99": torch.quantile(abs_flat, 0.99).item(),
    }
    return summary


def plot_histogram(
    eps: torch.Tensor,
    params: dict,
    bins: int,
    save_path: str,
) -> None:
    values = eps.cpu().view(-1).numpy()
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(values, bins=bins, density=True, color="#2b8cbe", alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("epsilon value")
    ax.set_ylabel("density (log scale)")
    ax.set_title(
        f"PVN noise: alpha={params['alpha']}, beta={params['beta']}, "
        f"clamp_eps={params['clamp_eps']}, clamp_v={params['clamp_v']}"
    )

    if params.get("clamp_eps") is not None:
        clamp = params["clamp_eps"]
        ax.axvline(clamp, color="crimson", linestyle="--", linewidth=1, label="clamp_eps")
        ax.axvline(-clamp, color="crimson", linestyle="--", linewidth=1)

    # focus on the bulk of the density for readability
    lower, upper = torch.quantile(torch.from_numpy(values), torch.tensor([0.001, 0.999]))
    limit = max(abs(lower.item()), abs(upper.item()))
    if params.get("clamp_eps") is not None:
        limit = min(limit, abs(params["clamp_eps"]))
    ax.set_xlim(-limit, limit)

    if ax.get_legend_handles_labels()[0]:
        ax.legend()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    params = extract_pvn_params(cfg)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but not available.")

    torch.manual_seed(args.seed)

    eps = sample_noise(params, args.num_samples, device)
    summary = summarize_noise(eps)

    print("PVN noise summary:")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}")

    plot_histogram(eps, params, args.bins, args.save_path)
    print(f"Saved histogram to {args.save_path}")


if __name__ == "__main__":
    main()
