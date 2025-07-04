# mini-vit

[![Python](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)
[![Tests](https://github.com/<your-username>/mini-vit/actions/workflows/python-app.yml/badge.svg)](https://github.com/<your-username>/mini-vit/actions)

A minimal Vision Transformer (ViT) implementation for image classification, focused on CIFAR-10. This project provides a clean, modular, and extensible codebase for experimenting with transformer-based vision models.

## Why use `uv`?

This project recommends [uv](https://github.com/astral-sh/uv) for fast, reliable Python package management and script execution. `uv` is a drop-in replacement for pip and venv, providing much faster installs and seamless script running. All examples below use `uv`.

## Features

- **Vision Transformer (ViT) Model**: Modular implementation with configurable depth, embedding size, attention heads, and pooling.
- **CIFAR-10 Data Pipeline**: Automated download, augmentation, and normalization.
- **Training API**: Command-line interface for training with rich logging and progress bars.
- **Evaluation**: Built-in validation and metrics reporting.
- **Extensive Unit Tests**: For model components and data loaders.
- **Rich Logging**: Uses `rich` for beautiful CLI output.

## Project Structure

```
mini-vit/
├── src/
│   └── mini_vit/
│       ├── model/
│       │   └── components.py  # ViT and submodules
│       ├── data/
│       │   └── cifar_data.py  # CIFAR-10 loading & transforms
│       ├── train/
│       │   └── train.py       # Training loop
│       └── api/
│           ├── train_api.py   # CLI for training
│           └── data_api.py    # CLI for data utilities
├── tests/
│   ├── test_data.py           # Data loader tests
│   └── model/
│       └── test_components.py # Model component tests
├── pyproject.toml             # Dependencies and scripts
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd mini-vit
   ```

2. **Install dependencies with uv:**
   ```bash
   uv pip install .
   # or for development
   uv pip install -e .[dev]
   ```

   Requires Python 3.13+ and [uv](https://github.com/astral-sh/uv) (install with `pip install uv` or see their docs).

## Usage

### Training

Train a Vision Transformer on CIFAR-10:

```bash
uv run train --epochs 20 --batch-size 128 --learning-rate 0.0005
```

**Configurable arguments:**
- `--epochs`: Number of training epochs (default: 10)
- `--learning-rate`: Learning rate (default: 3e-4)
- `--batch-size`: Batch size (default: 64)
- `--embedding-dim`, `--num-heads`, `--ff-dim`, `--depth`, `--dropout`, `--pool`, etc.

See all options:
```bash
uv run train --help
```

### Data Utilities

Inspect or load CIFAR-10 data:

```bash
uv run data cifar --split train --batch-size 64 --shuffle
```

## Model Overview

- **Patch Embedding**: Splits images into patches, flattens, and projects to embedding space.
- **Transformer Encoder**: Stack of multi-head self-attention and feed-forward blocks.
- **Pooling**: Supports both class token (`cls`) and mean pooling.
- **Classification Head**: MLP for final prediction.

See [`src/mini_vit/model/components.py`](src/mini_vit/model/components.py) for details.

## Data Pipeline

- **Augmentation**: Random crop, horizontal flip, color jitter, and auto-augment for training.
- **Normalization**: Standard CIFAR-10 mean and std.
- **Automatic Download**: Data is downloaded to `./cifar_data` if not present.

## Testing

Run all tests with:

```bash
uv pip install -e .[dev]  # if not already installed
pytest
```

- Data loader tests: `tests/test_data.py`
- Model component tests: `tests/model/test_components.py`

## Dependencies

- torch
- torchvision
- einops
- pillow
- rich
- tqdm

(See `pyproject.toml` for full list.)

## Development

- Jupyter and matplotlib for experiments (`[dev]` group).
- Linting and formatting via `ruff`.
- Markers for unit, integration, and slow tests in pytest.

## License

MIT License (add your license here if different).

---
