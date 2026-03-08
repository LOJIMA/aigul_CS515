import os
import random
import ssl
from typing import Final

import numpy as np
import torch
import torch.nn as nn

from models.mlp import MLP
from parameters import Config, get_config
from test import run_test
from train import run_training


# Optional SSL workaround for some systems while downloading datasets
ssl._create_default_https_context = ssl._create_unverified_context


DEFAULT_NUM_CLASSES: Final[int] = 10


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_output_paths(config: Config) -> None:
    """
    Create output directories if they do not already exist.

    Args:
        config: Full experiment configuration.
    """
    os.makedirs(config.run.save_dir, exist_ok=True)

    save_parent = os.path.dirname(config.run.save_path)
    if save_parent:
        os.makedirs(save_parent, exist_ok=True)


def count_trainable_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in the model.

    Args:
        model: PyTorch model.

    Returns:
        Number of trainable parameters.
    """
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def build_model(config: Config) -> nn.Module:
    """
    Build the model specified by the configuration.

    Since HW1a is scoped to MNIST classification with MLP, this function
    currently constructs only the configurable MLP model.

    Args:
        config: Full experiment configuration.

    Returns:
        Instantiated PyTorch model.

    Raises:
        ValueError: If an unsupported dataset is requested.
    """
    if config.data.dataset != "mnist":
        raise ValueError(
            f"Unsupported dataset '{config.data.dataset}'. "
            "HW1a is scoped to MNIST classification with MLP."
        )

    model = MLP(
        input_size=config.model.input_size,
        hidden_sizes=config.model.hidden_sizes,
        num_classes=config.model.num_classes,
        activation=config.model.activation,
        dropout=config.model.dropout,
        use_batch_norm=config.model.use_batch_norm,
    )
    return model


def print_config_summary(config: Config, model: nn.Module, device: torch.device) -> None:
    """
    Print a concise experiment summary.

    Args:
        config: Full experiment configuration.
        model: Instantiated model.
        device: Torch device used for execution.
    """
    print("=" * 70)
    print("CS515 HW1a - MNIST Classification with MLP")
    print("=" * 70)
    print(f"Mode              : {config.run.mode}")
    print(f"Experiment name   : {config.run.experiment_name}")
    print(f"Seed              : {config.run.seed}")
    print(f"Device            : {device}")
    print(f"Dataset           : {config.data.dataset}")
    print(f"Input size        : {config.data.input_size}")
    print(f"Num classes       : {config.data.num_classes}")
    print(f"Hidden sizes      : {config.model.hidden_sizes}")
    print(f"Activation        : {config.model.activation}")
    print(f"Dropout           : {config.model.dropout}")
    print(f"BatchNorm         : {config.model.use_batch_norm}")
    print(f"Epochs            : {config.train.epochs}")
    print(f"Batch size        : {config.train.batch_size}")
    print(f"Learning rate     : {config.train.learning_rate}")
    print(f"Optimizer         : {config.train.optimizer}")
    print(f"Scheduler         : {config.train.scheduler}")
    print(f"Regularizer       : {config.train.regularizer}")
    print(f"reg_lambda        : {config.train.reg_lambda}")
    print(f"Weight decay      : {config.train.weight_decay}")
    print(f"Checkpoint path   : {config.run.save_path}")
    print(f"Trainable params  : {count_trainable_parameters(model):,}")
    print("=" * 70)


def main() -> None:
    """
    Main entry point for running training and/or testing.

    Workflow:
    1. Parse CLI arguments into dataclass-based config.
    2. Set seeds for reproducibility.
    3. Prepare output paths.
    4. Build MLP model.
    5. Run training and/or testing depending on mode.
    """
    config = get_config()

    set_seed(config.run.seed)
    ensure_output_paths(config)

    device = torch.device(config.run.device)
    model = build_model(config).to(device)

    print_config_summary(config, model, device)
    print(model)

    if config.run.mode in ("train", "both"):
        run_training(model=model, config=config, device=device)

    if config.run.mode in ("test", "both"):
        if not os.path.exists(config.run.save_path) and config.run.mode == "test":
            raise FileNotFoundError(
                f"Checkpoint not found at: {config.run.save_path}. "
                "Train the model first or provide a valid --save_path."
            )
        run_test(model=model, config=config, device=device)


if __name__ == "__main__":
    main()