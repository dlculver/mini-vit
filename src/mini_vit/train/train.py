"""Module for running training pipelines"""

import torch
from torch.utils.data import DataLoader
from typing import Optional

import logging
from rich.logging import RichHandler
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn
from rich.progress import TaskID

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
    device: torch.device,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion,
    epoch: int,
    num_epochs: int,
    progress: Progress,
    epoch_task_id: TaskID,
    scheduler=None,
) -> float:
    """A single training epoch."""
    model.train()
    total_loss = 0.0
    
    # Create a nested progress bar for batches
    batch_task_id = progress.add_task(
        f"[cyan]Training Epoch {epoch + 1}/{num_epochs}",
        total=len(dataloader)
    )

    for batch in dataloader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if scheduler:
            scheduler.step()
            
        # Update batch progress
        progress.advance(batch_task_id)

    avg_loss = total_loss / len(dataloader)
    progress.remove_task(batch_task_id)
    logger.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
    return avg_loss


def train(
    model: VisionTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float = 0.01,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> None:
    """Full training loop."""
    model.to(device)

    # set up optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=weight_decay,
    )
    
    # Create progress instance with custom columns
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        expand=True
    )
    
    logger.info("Starting training...")
    with progress:
        # Create main progress task for epochs
        epoch_task = progress.add_task("[green]Training Progress", total=epochs)
        
        for epoch in range(epochs):
            # Train for one epoch
            train_loss = train_epoch(
                model=model,
                device=device,
                dataloader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                epoch=epoch,
                num_epochs=epochs,
                progress=progress,
                epoch_task_id=epoch_task,
                scheduler=scheduler,
            )
            
            # Evaluate
            val_loss, val_acc = evaluate(
                model=model,
                device=device,
                val_loader=val_loader,
                criterion=criterion,
                progress=progress
            )
            
            # Update main progress
            progress.advance(epoch_task)
            
            # Log metrics
            logger.info(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Acc: {val_acc:.2f}%"
            )
            
    logger.info("Training complete.")


def evaluate(
    model: VisionTransformer,
    device: torch.device,
    val_loader: DataLoader,
    criterion,
    progress: Progress,
) -> tuple[float, float]:
    """Evaluate the model on the test dataset."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    # Create evaluation progress bar
    eval_task_id = progress.add_task("[yellow]Evaluating", total=len(val_loader))

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            total += targets.size(0)
            correct += (preds == targets).sum().item()
            
            # Update evaluation progress
            progress.advance(eval_task_id)

    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total
    
    # Clean up the evaluation progress bar
    progress.remove_task(eval_task_id)
    
    return avg_loss, accuracy
