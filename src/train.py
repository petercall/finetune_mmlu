"""
Main training script for fine-tuning OLMo-1B on the finemath dataset.

Uses HuggingFace Accelerate for multi-GPU training with a custom training loop.
See plans/specification.md for full experiment documentation.

Usage:
    # Single GPU
    python src/train.py

    # Multi-GPU (e.g., 4 GPUs)
    accelerate launch --num_processes=4 src/train.py

    # With custom config
    python src/train.py --config configs/config.yaml
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.data import load_finemath_dataset, tokenize_dataset, create_dataloaders
from utils.checkpoints import save_checkpoint


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune OLMo-1B on finemath dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file",
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> LambdaLR:
    """
    Create a linear warmup + linear decay learning rate scheduler.
    
    Args:
        optimizer: The optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        
    Returns:
        Learning rate scheduler
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        else:
            # Linear decay
            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps)
            )
            return max(0.0, 1.0 - progress)
    
    return LambdaLR(optimizer, lr_lambda)


def compute_gradient_accumulation_steps(
    effective_batch_size: int,
    per_gpu_batch_size: int,
    num_processes: int,
) -> int:
    """
    Compute gradient accumulation steps to achieve effective batch size.
    
    Args:
        effective_batch_size: Target effective batch size
        per_gpu_batch_size: Batch size per GPU
        num_processes: Number of GPUs/processes
        
    Returns:
        Number of gradient accumulation steps
    """
    total_batch_per_step = per_gpu_batch_size * num_processes
    grad_accum = effective_batch_size // total_batch_per_step
    
    # Ensure at least 1 accumulation step
    grad_accum = max(1, grad_accum)
    
    # Log actual effective batch size
    actual_effective_batch = total_batch_per_step * grad_accum
    if actual_effective_batch != effective_batch_size:
        print(f"Note: Actual effective batch size is {actual_effective_batch} "
              f"(target was {effective_batch_size})")
    
    return grad_accum


def validate(
    model: torch.nn.Module,
    val_dataloader: torch.utils.data.DataLoader,
    accelerator: Accelerator,
) -> float:
    """
    Run validation and compute average loss.
    
    Args:
        model: The model to validate
        val_dataloader: Validation data loader
        accelerator: Accelerator instance
        
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validating", disable=not accelerator.is_main_process):
            outputs = model(**batch)
            loss = outputs.loss
            
            # Gather loss across processes
            gathered_loss = accelerator.gather(loss.unsqueeze(0)).mean()
            total_loss += gathered_loss.item()
            num_batches += 1
    
    model.train()
    return total_loss / num_batches


def save_training_plots(
    metrics: Dict,
    output_dir: str,
):
    """
    Save training loss curves and learning rate schedule as PNG files.
    
    Args:
        metrics: Dictionary containing training metrics
        output_dir: Directory to save plots
    """
    import matplotlib.pyplot as plt
    
    plots_dir = Path(output_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Loss curves
    if "train_losses" in metrics and "val_losses" in metrics:
        plt.figure(figsize=(10, 6))
        
        # Training loss
        train_steps = [m["step"] for m in metrics["train_losses"]]
        train_losses = [m["loss"] for m in metrics["train_losses"]]
        plt.plot(train_steps, train_losses, label="Training Loss", alpha=0.7)
        
        # Validation loss
        val_steps = [m["step"] for m in metrics["val_losses"]]
        val_losses = [m["loss"] for m in metrics["val_losses"]]
        plt.plot(val_steps, val_losses, label="Validation Loss", marker="o")
        
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(plots_dir / "loss_curves.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved loss curves to {plots_dir / 'loss_curves.png'}")
    
    # Learning rate schedule
    if "learning_rates" in metrics:
        plt.figure(figsize=(10, 6))
        
        steps = [m["step"] for m in metrics["learning_rates"]]
        lrs = [m["lr"] for m in metrics["learning_rates"]]
        plt.plot(steps, lrs)
        
        plt.xlabel("Training Steps")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedule")
        plt.grid(True, alpha=0.3)
        plt.savefig(plots_dir / "lr_schedule.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved LR schedule to {plots_dir / 'lr_schedule.png'}")


def train(config: Dict):
    """
    Main training function.
    
    Args:
        config: Configuration dictionary
    """
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=1,  # Will be set properly below
        mixed_precision=config["training"]["mixed_precision"],
        log_with="wandb" if config["logging"]["wandb"]["project"] else None,
    )
    
    # Set seed for reproducibility
    set_seed(config["training"]["seed"])
    
    # Compute gradient accumulation steps
    grad_accum_steps = compute_gradient_accumulation_steps(
        effective_batch_size=config["training"]["effective_batch_size"],
        per_gpu_batch_size=config["training"]["per_gpu_batch_size"],
        num_processes=accelerator.num_processes,
    )
    
    # Update accelerator with correct gradient accumulation
    accelerator.gradient_accumulation_steps = grad_accum_steps
    
    if accelerator.is_main_process:
        print("=" * 60)
        print("Fine-tuning OLMo-1B on finemath dataset")
        print("=" * 60)
        print(f"Number of processes: {accelerator.num_processes}")
        print(f"Mixed precision: {config['training']['mixed_precision']}")
        print(f"Per-GPU batch size: {config['training']['per_gpu_batch_size']}")
        print(f"Gradient accumulation steps: {grad_accum_steps}")
        print(f"Effective batch size: {config['training']['per_gpu_batch_size'] * accelerator.num_processes * grad_accum_steps}")
        print("=" * 60)
    
    # Initialize wandb tracking
    if accelerator.is_main_process and config["logging"]["wandb"]["project"]:
        accelerator.init_trackers(
            project_name=config["logging"]["wandb"]["project"],
            config=config,
        )
    
    # Load tokenizer
    if accelerator.is_main_process:
        print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    if accelerator.is_main_process:
        print("Loading model...")
    
    # Determine torch dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(config["model"]["torch_dtype"], torch.bfloat16)
    
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"],
        torch_dtype=torch_dtype,
    )
    
    if accelerator.is_main_process:
        print(f"Model loaded with dtype: {model.dtype}")
        print(f"Model parameters: {model.num_parameters():,}")
    
    # Load and prepare dataset
    if accelerator.is_main_process:
        print("\nLoading dataset...")
    
    train_dataset, val_dataset = load_finemath_dataset(
        target_size_gb=config["dataset"]["target_size_gb"],
        validation_split=config["dataset"]["validation_split"],
        seed=config["training"]["seed"],
        streaming=config["dataset"]["streaming"],
    )
    
    if accelerator.is_main_process:
        print("\nTokenizing datasets...")
    
    train_dataset = tokenize_dataset(
        train_dataset,
        tokenizer,
        max_length=config["tokenization"]["max_length"],
        text_column=config["dataset"]["text_column"],
    )
    
    val_dataset = tokenize_dataset(
        val_dataset,
        tokenizer,
        max_length=config["tokenization"]["max_length"],
        text_column=config["dataset"]["text_column"],
    )
    
    # Create dataloaders
    train_dataloader, val_dataloader = create_dataloaders(
        train_dataset,
        val_dataset,
        tokenizer,
        batch_size=config["training"]["per_gpu_batch_size"],
        max_length=config["tokenization"]["max_length"],
    )
    
    # Calculate training steps
    num_update_steps_per_epoch = len(train_dataloader) // grad_accum_steps
    max_epochs = config["training"]["max_epochs"]
    total_training_steps = num_update_steps_per_epoch * max_epochs
    
    # Calculate steps for validation/checkpointing
    validation_frequency = config["checkpointing"]["validation_frequency"]
    checkpoint_frequency = config["checkpointing"]["checkpoint_frequency"]
    steps_per_validation = int(num_update_steps_per_epoch * validation_frequency)
    steps_per_checkpoint = int(num_update_steps_per_epoch * checkpoint_frequency)
    
    if accelerator.is_main_process:
        print(f"\nTraining configuration:")
        print(f"  Steps per epoch: {num_update_steps_per_epoch}")
        print(f"  Total training steps: {total_training_steps}")
        print(f"  Validation every: {steps_per_validation} steps ({validation_frequency} epoch)")
        print(f"  Checkpoint every: {steps_per_checkpoint} steps ({checkpoint_frequency} epoch)")
    
    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    
    # Initialize learning rate scheduler
    num_warmup_steps = int(total_training_steps * config["training"]["lr_scheduler"]["warmup_ratio"])
    lr_scheduler = get_lr_scheduler(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_training_steps,
    )
    
    if accelerator.is_main_process:
        print(f"  Warmup steps: {num_warmup_steps}")
    
    # Prepare with accelerator
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )
    
    # Training state
    global_step = 0
    best_val_loss = float("inf")
    patience_counter = 0
    early_stop_triggered = False
    steps_after_trigger = 0
    
    # Metrics storage
    metrics = {
        "train_losses": [],
        "val_losses": [],
        "learning_rates": [],
    }
    
    # Create output directories
    if accelerator.is_main_process:
        Path(config["checkpointing"]["output_dir"]).mkdir(parents=True, exist_ok=True)
        Path(config["logging"]["local"]["log_dir"]).mkdir(parents=True, exist_ok=True)
        Path(config["visualization"]["plots_dir"]).mkdir(parents=True, exist_ok=True)
    
    # Training loop
    if accelerator.is_main_process:
        print("\n" + "=" * 60)
        print("Starting training...")
        print("=" * 60)
    
    model.train()
    
    for epoch in range(max_epochs):
        if accelerator.is_main_process:
            print(f"\nEpoch {epoch + 1}/{max_epochs}")
        
        epoch_loss = 0.0
        epoch_steps = 0
        
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}",
            disable=not accelerator.is_main_process,
        )
        
        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Only count global steps after full gradient accumulation
            if (step + 1) % grad_accum_steps == 0:
                global_step += 1
                epoch_steps += 1
                epoch_loss += loss.item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}",
                })
                
                # Store metrics
                if accelerator.is_main_process:
                    current_lr = lr_scheduler.get_last_lr()[0]
                    metrics["train_losses"].append({
                        "step": global_step,
                        "epoch": epoch + (step + 1) / len(train_dataloader),
                        "loss": loss.item(),
                    })
                    metrics["learning_rates"].append({
                        "step": global_step,
                        "lr": current_lr,
                    })
                    
                    # Log to wandb
                    if config["logging"]["wandb"]["project"] and global_step % config["logging"]["wandb"]["log_interval"] == 0:
                        accelerator.log({
                            "train/loss": loss.item(),
                            "train/learning_rate": current_lr,
                            "train/epoch": epoch + (step + 1) / len(train_dataloader),
                        }, step=global_step)
                    
                    # Print progress
                    if global_step % config["logging"]["console"]["print_every"] == 0:
                        avg_loss = epoch_loss / epoch_steps
                        print(f"  Step {global_step}: loss={avg_loss:.4f}, lr={current_lr:.2e}")
                
                # Validation check
                if global_step % steps_per_validation == 0:
                    if accelerator.is_main_process:
                        print(f"\n  Running validation at step {global_step}...")
                    
                    val_loss = validate(model, val_dataloader, accelerator)
                    
                    if accelerator.is_main_process:
                        print(f"  Validation loss: {val_loss:.4f}")
                        
                        metrics["val_losses"].append({
                            "step": global_step,
                            "epoch": epoch + (step + 1) / len(train_dataloader),
                            "loss": val_loss,
                        })
                        
                        if config["logging"]["wandb"]["project"]:
                            accelerator.log({
                                "val/loss": val_loss,
                            }, step=global_step)
                        
                        # Early stopping check
                        if val_loss < best_val_loss - config["early_stopping"]["min_delta"]:
                            best_val_loss = val_loss
                            patience_counter = 0
                            print(f"  New best validation loss: {best_val_loss:.4f}")
                        else:
                            patience_counter += 1
                            print(f"  No improvement. Patience: {patience_counter}/{config['early_stopping']['patience']}")
                            
                            if patience_counter >= config["early_stopping"]["patience"]:
                                if not early_stop_triggered:
                                    early_stop_triggered = True
                                    steps_to_continue = int(
                                        num_update_steps_per_epoch * 
                                        config["early_stopping"]["continue_after_trigger"]
                                    )
                                    print(f"\n  Early stopping triggered! Continuing for {steps_to_continue} more steps...")
                
                # Checkpoint saving
                if global_step % steps_per_checkpoint == 0:
                    current_epoch = epoch + (step + 1) / len(train_dataloader)
                    checkpoint_name = f"epoch-{current_epoch:.1f}"
                    
                    if accelerator.is_main_process:
                        print(f"\n  Saving checkpoint: {checkpoint_name}")
                    
                    save_checkpoint(
                        model=model,
                        tokenizer=tokenizer,
                        output_dir=config["checkpointing"]["output_dir"],
                        checkpoint_name=checkpoint_name,
                        metadata={
                            "epoch": current_epoch,
                            "step": global_step,
                            "train_loss": epoch_loss / epoch_steps if epoch_steps > 0 else 0,
                            "val_loss": val_loss if "val_loss" in dir() else None,
                            "best_val_loss": best_val_loss,
                        },
                        accelerator=accelerator,
                    )
                
                # Check for early stop completion
                if early_stop_triggered:
                    steps_after_trigger += 1
                    if steps_after_trigger >= steps_to_continue:
                        if accelerator.is_main_process:
                            print("\n  Saving final checkpoint after early stopping...")
                        
                        save_checkpoint(
                            model=model,
                            tokenizer=tokenizer,
                            output_dir=config["checkpointing"]["output_dir"],
                            checkpoint_name="final",
                            metadata={
                                "epoch": epoch + (step + 1) / len(train_dataloader),
                                "step": global_step,
                                "train_loss": epoch_loss / epoch_steps if epoch_steps > 0 else 0,
                                "val_loss": val_loss if "val_loss" in dir() else None,
                                "best_val_loss": best_val_loss,
                                "early_stopped": True,
                            },
                            accelerator=accelerator,
                        )
                        
                        # Save metrics and plots
                        if accelerator.is_main_process:
                            _save_final_outputs(config, metrics)
                        
                        print("\n" + "=" * 60)
                        print("Training complete (early stopped)")
                        print("=" * 60)
                        
                        if accelerator.is_main_process and config["logging"]["wandb"]["project"]:
                            accelerator.end_training()
                        
                        return
        
        # End of epoch summary
        if accelerator.is_main_process:
            avg_epoch_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0
            print(f"\nEpoch {epoch + 1} complete. Average loss: {avg_epoch_loss:.4f}")
    
    # Training complete (max epochs reached)
    if accelerator.is_main_process:
        print("\n  Saving final checkpoint...")
    
    save_checkpoint(
        model=model,
        tokenizer=tokenizer,
        output_dir=config["checkpointing"]["output_dir"],
        checkpoint_name="final",
        metadata={
            "epoch": max_epochs,
            "step": global_step,
            "best_val_loss": best_val_loss,
            "early_stopped": False,
        },
        accelerator=accelerator,
    )
    
    # Save metrics and plots
    if accelerator.is_main_process:
        _save_final_outputs(config, metrics)
    
    print("\n" + "=" * 60)
    print("Training complete")
    print("=" * 60)
    
    if accelerator.is_main_process and config["logging"]["wandb"]["project"]:
        accelerator.end_training()


def _save_final_outputs(config: Dict, metrics: Dict):
    """Save final training outputs (metrics JSON and plots)."""
    # Save metrics JSON
    log_dir = Path(config["logging"]["local"]["log_dir"])
    metrics_path = log_dir / f"training_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved training metrics to {metrics_path}")
    
    # Save plots
    if config["visualization"]["loss_curves"] or config["visualization"]["lr_schedule"]:
        save_training_plots(metrics, config["visualization"]["plots_dir"])


def main():
    """Main entry point."""
    args = parse_args()
    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
