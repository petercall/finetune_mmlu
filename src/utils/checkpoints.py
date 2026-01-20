"""
Checkpoint management utilities for the fine-tuning experiment.

Handles saving model checkpoints (weights only, no optimizer state)
for later evaluation with MMLU.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


def get_checkpoint_dir(output_dir: str = "outputs/checkpoints") -> Path:
    """
    Get the checkpoint directory, creating it if it doesn't exist.
    
    Args:
        output_dir: Base directory for checkpoints
        
    Returns:
        Path to checkpoint directory
    """
    checkpoint_dir = Path(output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def save_checkpoint(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    output_dir: str,
    checkpoint_name: str,
    metadata: Optional[Dict[str, Any]] = None,
    accelerator=None,
) -> str:
    """
    Save a model checkpoint (weights only, no optimizer state).
    
    Args:
        model: The model to save
        tokenizer: The tokenizer to save alongside
        output_dir: Base directory for checkpoints
        checkpoint_name: Name for this checkpoint (e.g., 'epoch-0.5', 'final')
        metadata: Optional metadata to save (epoch, step, loss, etc.)
        accelerator: Accelerate accelerator for distributed saving
        
    Returns:
        Path to saved checkpoint directory
    """
    checkpoint_dir = get_checkpoint_dir(output_dir)
    checkpoint_path = checkpoint_dir / checkpoint_name
    
    # Only save on main process in distributed setting
    if accelerator is not None:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            _save_checkpoint_files(model, tokenizer, checkpoint_path, metadata, accelerator)
        accelerator.wait_for_everyone()
    else:
        _save_checkpoint_files(model, tokenizer, checkpoint_path, metadata)
    
    return str(checkpoint_path)


def _save_checkpoint_files(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    checkpoint_path: Path,
    metadata: Optional[Dict[str, Any]] = None,
    accelerator=None,
):
    """Internal function to save checkpoint files."""
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    # Unwrap model if using accelerate
    if accelerator is not None:
        unwrapped_model = accelerator.unwrap_model(model)
    else:
        unwrapped_model = model
    
    # Save model weights and config
    unwrapped_model.save_pretrained(
        checkpoint_path,
        safe_serialization=True,  # Use safetensors format
    )
    
    # Save tokenizer
    tokenizer.save_pretrained(checkpoint_path)
    
    # Save metadata
    if metadata is not None:
        metadata_with_timestamp = {
            **metadata,
            "saved_at": datetime.now().isoformat(),
        }
        metadata_path = checkpoint_path / "training_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata_with_timestamp, f, indent=2)
    
    print(f"Saved checkpoint to {checkpoint_path}")


def get_checkpoint_paths(output_dir: str = "outputs/checkpoints") -> List[Path]:
    """
    Get all checkpoint paths in the output directory, sorted by name.
    
    Args:
        output_dir: Base directory for checkpoints
        
    Returns:
        List of checkpoint directory paths
    """
    checkpoint_dir = Path(output_dir)
    
    if not checkpoint_dir.exists():
        return []
    
    # Find all subdirectories that look like checkpoints
    checkpoints = []
    for path in checkpoint_dir.iterdir():
        if path.is_dir() and (path / "config.json").exists():
            checkpoints.append(path)
    
    # Sort by name (assumes names like 'epoch-0.5', 'epoch-1.0', etc.)
    return sorted(checkpoints, key=lambda p: p.name)


def load_checkpoint_metadata(checkpoint_path: str) -> Optional[Dict[str, Any]]:
    """
    Load training metadata from a checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint directory
        
    Returns:
        Metadata dictionary, or None if not found
    """
    metadata_path = Path(checkpoint_path) / "training_metadata.json"
    
    if not metadata_path.exists():
        return None
    
    with open(metadata_path, "r") as f:
        return json.load(f)


def get_checkpoint_epoch(checkpoint_path: str) -> Optional[float]:
    """
    Get the epoch number from a checkpoint (from name or metadata).
    
    Args:
        checkpoint_path: Path to checkpoint directory
        
    Returns:
        Epoch number, or None if not determinable
    """
    path = Path(checkpoint_path)
    
    # Try to parse from directory name
    name = path.name
    if name.startswith("epoch-"):
        try:
            return float(name.replace("epoch-", ""))
        except ValueError:
            pass
    
    # Try to get from metadata
    metadata = load_checkpoint_metadata(checkpoint_path)
    if metadata and "epoch" in metadata:
        return float(metadata["epoch"])
    
    return None


def cleanup_old_checkpoints(
    output_dir: str = "outputs/checkpoints",
    keep_last: int = 3,
    keep_best: bool = True,
) -> List[str]:
    """
    Remove old checkpoints, keeping only the most recent ones.
    
    Args:
        output_dir: Base directory for checkpoints
        keep_last: Number of recent checkpoints to keep
        keep_best: Whether to also keep the checkpoint marked as 'best'
        
    Returns:
        List of paths that were removed
    """
    checkpoints = get_checkpoint_paths(output_dir)
    
    if len(checkpoints) <= keep_last:
        return []
    
    removed = []
    checkpoints_to_consider = checkpoints[:-keep_last]
    
    for checkpoint_path in checkpoints_to_consider:
        # Skip 'best' checkpoint if keep_best is True
        if keep_best and checkpoint_path.name == "best":
            continue
        
        # Skip 'final' checkpoint
        if checkpoint_path.name == "final":
            continue
        
        # Remove checkpoint
        import shutil
        shutil.rmtree(checkpoint_path)
        removed.append(str(checkpoint_path))
        print(f"Removed old checkpoint: {checkpoint_path}")
    
    return removed


if __name__ == "__main__":
    # Test checkpoint utilities
    print("Testing checkpoint utilities...")
    
    # List any existing checkpoints
    checkpoints = get_checkpoint_paths()
    print(f"Found {len(checkpoints)} existing checkpoints:")
    for cp in checkpoints:
        metadata = load_checkpoint_metadata(cp)
        epoch = get_checkpoint_epoch(cp)
        print(f"  - {cp.name}: epoch={epoch}, metadata={metadata is not None}")
    
    print("\nCheckpoint utilities test complete!")
