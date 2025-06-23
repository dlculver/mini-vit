"""Module for running training pipelines"""

import torch
from torch.utils.data import DataLoader

import logging
from rich.logging import RichHandler
from rich.progress import track

from mini_vit.model import VisionTransformer

# Set up rich logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",  # RichHandler handles formatting, so keep this simple
    datefmt="[%Y-%m-%d %H:%M:%S]",  # Custom date format
    handlers=[
        RichHandler(rich_tracebacks=True, show_path=False)  # Rich console output
    ],
)
logger = logging.getLogger(__name__)


def train_epoch(
    model: VisionTransformer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion,
    epoch: int,
    num_epochs: int,
    scheduler=None,
) -> None:
    """A single training epoch."""
    model.train()
    total_loss = 0.0
    for batch in track(
        dataloader, description=f"Training Epoch {epoch + 1}/{num_epochs}"
    ):
        inputs, targets = batch
        inputs, targets = inputs.to(model.device), targets.to(model.device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()

    avg_loss = total_loss / len(dataloader)
    logger.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")


def train(
    model: VisionTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion,
    epochs: int,
    scheduler=None,
) -> None:
    """Full training loop."""
    for epoch in track(range(epochs), description="Overall Training Progress"):
        train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            epoch=epoch,
            num_epochs=epochs,
            scheduler=scheduler,
        )
        evaluate(model=model, val_loader=val_loader, criterion=criterion)
    logger.info("Training complete.")


def evaluate(
    model: VisionTransformer,
    val_loader: DataLoader,
    criterion,
) -> None:
    """Evaluate the model on the test dataset."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in track(val_loader, description="Evaluating Model"):
            inputs, targets = inputs.to(model.device), targets.to(model.device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            total += targets.size(0)
            correct += (preds == targets).sum().item()

    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total
    logger.info(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
