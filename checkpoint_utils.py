"""
Checkpoint utilities for distogram training script.
"""

import os
import glob
import torch


def save_checkpoint(checkpoint_dir, model, optimizer, epoch, loss, prefix, target):
    """
    Save a training checkpoint.

    Args:
        checkpoint_dir: Directory to save checkpoints (e.g. "checkpoints_{prefix}").
        model:          The Conv model being trained.
        optimizer:      Adam optimizer.
        epoch:          Current epoch number.
        loss:           Current training loss (float).
        prefix:         Run prefix string (from argv[3]).
        target:         Target protein name (from argv[1]).
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "prefix": prefix,
        "target": target,
    }

    path = os.path.join(checkpoint_dir, f"{target}_epoch{epoch:04d}.pth")
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")
    return path


def load_checkpoint(checkpoint_path, model, optimizer=None, device="cuda"):
    """
    Load a checkpoint and restore model (and optionally optimizer) state.

    Args:
        checkpoint_path: Path to the .pth file.
        model:           The Conv model to load weights into.
        optimizer:       Pass the optimizer to also restore its state.
        device:          Device to map tensors to.

    Returns:
        dict with keys: epoch, loss, prefix, target
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(
        f"Loaded checkpoint: {checkpoint_path} "
        f"(epoch={checkpoint['epoch']}, loss={checkpoint['loss']:.6f})"
    )
    return checkpoint


def get_latest_checkpoint(checkpoint_dir, target):
    """
    Find the most recent checkpoint for a given target protein.

    Args:
        checkpoint_dir: Directory containing .pth files.
        target:         Target protein name to filter by.

    Returns:
        Path to the latest checkpoint, or None if none found.
    """
    pattern = os.path.join(checkpoint_dir, f"{target}_epoch*.pth")
    checkpoints = sorted(glob.glob(pattern))
    return checkpoints[-1] if checkpoints else None


def maybe_resume(checkpoint_dir, target, model, optimizer=None, device="cuda"):
    """
    Resume from the latest checkpoint for this target if one exists.

    Args:
        checkpoint_dir: Directory to search for checkpoints.
        target:         Target protein name.
        model:          The Conv model.
        optimizer:      Pass to also restore optimizer state.
        device:         Device to map tensors to.

    Returns:
        start_epoch (int): Epoch to resume from (0 if no checkpoint found).
    """
    ckpt_path = get_latest_checkpoint(checkpoint_dir, target)
    if ckpt_path is None:
        print(f"No checkpoint found for {target} — starting from scratch.")
        return 0

    state = load_checkpoint(ckpt_path, model, optimizer=optimizer, device=device)
    return state["epoch"] + 1  # resume from next epoch


def keep_last_n_checkpoints(checkpoint_dir, target, n=3):
    """
    Delete older checkpoints, keeping only the n most recent for this target.

    Args:
        checkpoint_dir: Directory containing .pth files.
        target:         Target protein name.
        n:              Number of checkpoints to keep.
    """
    pattern = os.path.join(checkpoint_dir, f"{target}_epoch*.pth")
    checkpoints = sorted(glob.glob(pattern))
    for old in checkpoints[:-n]:
        os.remove(old)
        print(f"Removed old checkpoint: {old}")
    