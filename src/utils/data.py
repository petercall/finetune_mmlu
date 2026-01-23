"""
Data loading utilities for the fine-tuning experiment.

Handles loading and preprocessing of the HuggingFaceTB/finemath-4plus dataset
for causal language modeling training.
"""

import os
from typing import Dict, Optional, Tuple

import torch
from datasets import load_dataset, Dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer


def estimate_dataset_size_gb(dataset: Dataset, sample_size: int = 1000) -> float:
    """
    Estimate the size of a dataset in GB by sampling.
    
    Args:
        dataset: The dataset to estimate size for
        sample_size: Number of samples to use for estimation
        
    Returns:
        Estimated size in GB
    """
    # Sample a subset for estimation
    sample = dataset.select(range(min(sample_size, len(dataset))))
    
    # Get the text column (assuming 'text' is the main content column)
    text_column = "text" if "text" in sample.column_names else sample.column_names[0]
    
    # Calculate average bytes per sample
    total_bytes = sum(len(s.encode('utf-8')) for s in sample[text_column])
    avg_bytes_per_sample = total_bytes / len(sample)
    
    # Estimate total size
    estimated_bytes = avg_bytes_per_sample * len(dataset)
    estimated_gb = estimated_bytes / (1024 ** 3)
    
    return estimated_gb


def load_finemath_dataset(
    target_size_gb: float = 10.0,
    validation_split: float = 0.1,
    seed: int = 42,
    streaming: bool = False,
) -> Tuple[Dataset, Dataset]:
    """
    Load the finemath-4plus dataset, streaming approximately the target size.
    
    Args:
        target_size_gb: Target size of data to load in GB (default: 10GB)
        validation_split: Fraction of data to use for validation (default: 0.1)
        seed: Random seed for reproducibility
        streaming: Whether to use streaming mode (recommended for large datasets)
        
    Returns:
        Tuple of (train_dataset, validation_dataset)
    """
    print(f"Loading HuggingFaceTB/finemath-4plus dataset (target: {target_size_gb}GB)...")
    
    if streaming:
        # Load dataset in streaming mode
        dataset = load_dataset(
            "HuggingFaceTB/finemath",
            "finemath-4plus",
            split="train",
            streaming=True
        )
        
        # Estimate samples needed for target size
        # finemath-4plus has ~500 bytes average per sample (rough estimate)
        avg_bytes_per_sample = 500
        target_bytes = target_size_gb * (1024 ** 3)
        target_samples = int(target_bytes / avg_bytes_per_sample)
        
        print(f"Streaming approximately {target_samples:,} samples...")
        
        # Take samples from the stream
        samples = []
        for i, sample in enumerate(dataset):
            if i >= target_samples:
                break
            samples.append(sample)
            if (i + 1) % 100000 == 0:
                print(f"  Loaded {i + 1:,} samples...")
        
        # Convert to Dataset
        full_dataset = Dataset.from_list(samples)
        print(f"Loaded {len(full_dataset):,} samples")
        
    else:
        # Load full dataset (may be slow/memory-intensive)
        full_dataset = load_dataset(
            "HuggingFaceTB/finemath",
            "finemath-4plus",
            split="train",
            streaming=False
        )
    
    # Split into train and validation
    split_dataset = full_dataset.train_test_split(
        test_size=validation_split,
        seed=seed,
    )
    
    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]
    
    print(f"Train set: {len(train_dataset):,} samples")
    print(f"Validation set: {len(val_dataset):,} samples")
    
    return train_dataset, val_dataset


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
    text_column: str = "text",
    num_proc: Optional[int] = None,
) -> Dataset:
    """
    Tokenize a dataset for causal language modeling.
    
    Args:
        dataset: The dataset to tokenize
        tokenizer: The tokenizer to use
        max_length: Maximum sequence length (default: 2048)
        text_column: Name of the column containing text
        num_proc: Number of processes for parallel tokenization
        
    Returns:
        Tokenized dataset with input_ids and attention_mask
    """
    def tokenize_function(examples):
        # Tokenize the text
        tokenized = tokenizer(
            examples[text_column],
            truncation=True,
            max_length=max_length,
            padding=False,  # We'll pad in the collator
            return_tensors=None,
        )
        
        # For causal LM, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    # Determine number of processes
    if num_proc is None:
        num_proc = min(os.cpu_count() or 1, 8)
    
    # Remove original columns and add tokenized columns
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )
    
    return tokenized_dataset


class DataCollatorForCausalLM:
    """
    Data collator for causal language modeling with dynamic padding.
    
    Pads sequences to the maximum length in the batch (not the global max length),
    which is more efficient than padding to a fixed length.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        pad_to_max: bool = False,
    ):
        """
        Args:
            tokenizer: The tokenizer (for pad token id)
            max_length: Maximum sequence length
            pad_to_max: If True, pad to max_length; if False, pad to batch max
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_max = pad_to_max
        self.pad_token_id = tokenizer.pad_token_id
        
        # If no pad token, use eos token
        if self.pad_token_id is None:
            self.pad_token_id = tokenizer.eos_token_id
    
    def __call__(self, features: list) -> Dict[str, torch.Tensor]:
        # Get max length in this batch
        if self.pad_to_max:
            max_len = self.max_length
        else:
            max_len = min(
                max(len(f["input_ids"]) for f in features),
                self.max_length
            )
        
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        
        for feature in features:
            input_ids = feature["input_ids"][:max_len]
            labels = feature["labels"][:max_len]
            
            # Calculate padding needed
            padding_length = max_len - len(input_ids)
            
            # Pad input_ids and attention_mask
            batch["input_ids"].append(
                input_ids + [self.pad_token_id] * padding_length
            )
            batch["attention_mask"].append(
                [1] * len(input_ids) + [0] * padding_length
            )
            
            # Pad labels with -100 (ignored in loss calculation)
            batch["labels"].append(
                labels + [-100] * padding_length
            )
        
        # Convert to tensors
        return {
            key: torch.tensor(value, dtype=torch.long)
            for key, value in batch.items()
        }


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 8,
    max_length: int = 2048,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders.
    
    Args:
        train_dataset: Tokenized training dataset
        val_dataset: Tokenized validation dataset
        tokenizer: The tokenizer (for padding)
        batch_size: Batch size per GPU
        max_length: Maximum sequence length
        num_workers: Number of workers for data loading
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        max_length=max_length,
        pad_to_max=False,  # Dynamic padding for efficiency
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_dataloader, val_dataloader


if __name__ == "__main__":
    # Test the data loading
    from transformers import AutoTokenizer
    
    print("Testing data loading utilities...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B-hf")
    
    # Load a small subset for testing
    train_data, val_data = load_finemath_dataset(
        target_size_gb=0.01,  # Just 10MB for testing
        streaming=True
    )
    
    print(f"\nTrain sample: {train_data[0]}")
    
    # Tokenize
    train_tokenized = tokenize_dataset(train_data, tokenizer, max_length=2048)
    val_tokenized = tokenize_dataset(val_data, tokenizer, max_length=2048)
    
    print(f"\nTokenized train sample: {train_tokenized[0]}")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_tokenized, val_tokenized, tokenizer, batch_size=2
    )
    
    # Get a batch
    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    for key, value in batch.items():
        print(f"  {key}: {value.shape}")
    
    print("\nData loading test complete!")
