from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.mlp import MLP
from plotting import plot_tsne_embeddings
from utils import resolve_device


def get_test_loader(data_dir: str, batch_size: int, num_workers: int) -> DataLoader:
    """
    Build MNIST test DataLoader with the same normalization used in training.

    Args:
        data_dir: Dataset root directory.
        batch_size: Batch size.
        num_workers: Number of DataLoader workers.

    Returns:
        Test DataLoader.
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="t-SNE analysis for trained MLP features on MNIST.")

    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="./results/plots/tsne.png")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[512, 256, 128])
    parser.add_argument("--activation", choices=["relu", "gelu"], default="relu")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--use_batch_norm", action="store_true")

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--random_state", type=int, default=42)

    args = parser.parse_args()

    device = resolve_device(args.device)

    model = MLP(
        input_size=784,
        hidden_sizes=args.hidden_sizes,
        num_classes=10,
        activation=args.activation,
        dropout=args.dropout,
        use_batch_norm=args.use_batch_norm,
    )

    state_dict = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    test_loader = get_test_loader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    plot_tsne_embeddings(
        model=model,
        loader=test_loader,
        device=device,
        output_path=args.output_path,
        max_samples=args.max_samples,
        perplexity=args.perplexity,
        random_state=args.random_state,
        title="t-SNE of MLP Hidden Features on MNIST",
    )

    print("t-SNE plot saved to:", args.output_path)


if __name__ == "__main__":
    main()