from __future__ import annotations

import argparse
from pathlib import Path

from plotting import (
    plot_accuracy_curve,
    plot_confusion_matrix,
    plot_learning_rate_curve,
    plot_loss_curve,
)
from utils import ensure_dir, load_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate plots from saved training/test results.")
    parser.add_argument("--history_path", type=str, required=True,
                        help="Path to *_history.json")
    parser.add_argument("--test_results_path", type=str, required=True,
                        help="Path to *_test_results.json")
    parser.add_argument("--output_dir", type=str, default="./results/plots")
    parser.add_argument("--experiment_name", type=str, default="experiment")
    parser.add_argument("--normalize_cm", action="store_true",
                        help="Normalize confusion matrix rows")
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)

    history = load_json(args.history_path)
    test_results = load_json(args.test_results_path)

    plot_loss_curve(
        train_loss=history["train_loss"],
        val_loss=history["val_loss"],
        output_path=str(Path(output_dir) / f"{args.experiment_name}_loss.png"),
        title=f"{args.experiment_name} - Loss Curve",
    )

    plot_accuracy_curve(
        train_acc=history["train_acc"],
        val_acc=history["val_acc"],
        output_path=str(Path(output_dir) / f"{args.experiment_name}_accuracy.png"),
        title=f"{args.experiment_name} - Accuracy Curve",
    )

    plot_learning_rate_curve(
        learning_rates=history["learning_rate"],
        output_path=str(Path(output_dir) / f"{args.experiment_name}_lr.png"),
        title=f"{args.experiment_name} - Learning Rate Curve",
    )

    plot_confusion_matrix(
        confusion_matrix=test_results["confusion_matrix"],
        output_path=str(Path(output_dir) / f"{args.experiment_name}_confusion_matrix.png"),
        normalize=args.normalize_cm,
        title=f"{args.experiment_name} - Confusion Matrix",
    )

    print("Plots saved to:", output_dir)


if __name__ == "__main__":
    main()