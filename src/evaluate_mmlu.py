"""
MMLU evaluation script for measuring model capabilities.

Evaluates model performance on the MMLU benchmark using 2-shot prompting.
See plans/specification.md for full experiment documentation.

Usage:
    # Evaluate specific checkpoint
    python src/evaluate_mmlu.py --checkpoint outputs/checkpoints/epoch-1.0

    # Evaluate base model (before fine-tuning)
    python src/evaluate_mmlu.py --model allenai/OLMo-1B-hf

    # Evaluate all checkpoints in a directory
    python src/evaluate_mmlu.py --checkpoint-dir outputs/checkpoints
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.mmlu_categories import (
    MMLU_SUBJECTS,
    MMLU_CATEGORIES,
    get_category_for_subject,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate model on MMLU benchmark")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--checkpoint",
        type=str,
        help="Path to a specific checkpoint to evaluate",
    )
    group.add_argument(
        "--model",
        type=str,
        help="HuggingFace model name to evaluate (e.g., allenai/OLMo-1B-hf)",
    )
    group.add_argument(
        "--checkpoint-dir",
        type=str,
        help="Directory containing multiple checkpoints to evaluate",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/mmlu_results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--num-fewshot",
        type=int,
        default=2,
        help="Number of few-shot examples",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for few-shot example selection",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for evaluation",
    )
    
    return parser.parse_args()


def format_question(question: str, choices: List[str]) -> str:
    """
    Format a question with answer choices.
    
    Args:
        question: The question text
        choices: List of answer choices
        
    Returns:
        Formatted question string
    """
    labels = ["A", "B", "C", "D"]
    choice_str = " ".join(f"({labels[i]}) {choice}" for i, choice in enumerate(choices))
    return f"Q: {question}\n{choice_str}\nA:"


def format_fewshot_example(question: str, choices: List[str], answer_idx: int) -> str:
    """
    Format a few-shot example with question and answer.
    
    Args:
        question: The question text
        choices: List of answer choices
        answer_idx: Index of the correct answer (0-3)
        
    Returns:
        Formatted example string with answer
    """
    labels = ["A", "B", "C", "D"]
    choice_str = " ".join(f"({labels[i]}) {choice}" for i, choice in enumerate(choices))
    return f"Q: {question}\n{choice_str}\nA: {labels[answer_idx]}"


def load_mmlu_data() -> Dict[str, Dict]:
    """
    Load MMLU dataset for all subjects.
    
    Returns:
        Dictionary mapping subject names to their test data
    """
    print("Loading MMLU dataset...")
    
    all_data = {}
    
    for subject in tqdm(MMLU_SUBJECTS, desc="Loading subjects"):
        try:
            dataset = load_dataset("cais/mmlu", subject, split="test")
            all_data[subject] = dataset
        except Exception as e:
            print(f"Warning: Failed to load {subject}: {e}")
    
    print(f"Loaded {len(all_data)} subjects")
    return all_data


def select_fewshot_examples(
    all_data: Dict[str, Dict],
    num_examples: int,
    seed: int,
) -> Tuple[List[Dict], set]:
    """
    Select random few-shot examples from the full MMLU dataset.
    
    Args:
        all_data: Dictionary of subject -> dataset
        num_examples: Number of examples to select
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (list of example dicts, set of (subject, idx) pairs to exclude)
    """
    random.seed(seed)
    
    # Collect all examples with their locations
    all_examples = []
    for subject, dataset in all_data.items():
        for idx in range(len(dataset)):
            all_examples.append((subject, idx, dataset[idx]))
    
    # Select random examples
    selected = random.sample(all_examples, min(num_examples, len(all_examples)))
    
    # Build exclusion set
    excluded_indices = {(subject, idx) for subject, idx, _ in selected}
    
    # Format examples
    examples = []
    for subject, idx, item in selected:
        examples.append({
            "subject": subject,
            "question": item["question"],
            "choices": item["choices"],
            "answer_idx": item["answer"],
        })
    
    return examples, excluded_indices


def extract_answer(generated_text: str) -> Optional[str]:
    """
    Extract the answer letter from model output.
    
    Args:
        generated_text: The text generated by the model
        
    Returns:
        Single letter (A, B, C, D) or None if invalid
    """
    # Strip whitespace and get first character
    text = generated_text.strip()
    
    if not text:
        return None
    
    first_char = text[0].upper()
    
    if first_char in ["A", "B", "C", "D"]:
        return first_char
    
    return None


def evaluate_model(
    model_path: str,
    all_data: Dict[str, Dict],
    fewshot_examples: List[Dict],
    excluded_indices: set,
    batch_size: int = 8,
    device: str = "cuda",
) -> Dict:
    """
    Evaluate a model on MMLU.
    
    Args:
        model_path: Path to model checkpoint or HuggingFace model name
        all_data: Dictionary of subject -> dataset
        fewshot_examples: List of few-shot example dicts
        excluded_indices: Set of (subject, idx) pairs to exclude from evaluation
        batch_size: Batch size for evaluation
        device: Device to use
        
    Returns:
        Dictionary containing evaluation results
    """
    print(f"\nLoading model from {model_path}...")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Build few-shot prompt prefix
    fewshot_prefix = "\n\n".join(
        format_fewshot_example(ex["question"], ex["choices"], ex["answer_idx"])
        for ex in fewshot_examples
    )
    fewshot_prefix += "\n\n"
    
    # Results storage
    results = {
        "model_path": model_path,
        "subjects": {},
        "categories": {},
        "overall": {},
    }
    
    subject_correct = defaultdict(int)
    subject_total = defaultdict(int)
    
    # Evaluate each subject
    for subject in tqdm(MMLU_SUBJECTS, desc="Evaluating subjects"):
        if subject not in all_data:
            continue
        
        dataset = all_data[subject]
        
        # Prepare prompts (excluding few-shot examples)
        prompts = []
        answers = []
        indices = []
        
        for idx in range(len(dataset)):
            if (subject, idx) in excluded_indices:
                continue
            
            item = dataset[idx]
            prompt = fewshot_prefix + format_question(item["question"], item["choices"])
            prompts.append(prompt)
            answers.append(["A", "B", "C", "D"][item["answer"]])
            indices.append(idx)
        
        if not prompts:
            continue
        
        # Batch evaluation
        correct = 0
        total = 0
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_answers = answers[i:i + batch_size]
            
            # Tokenize
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(device)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            # Extract new tokens only
            generated_tokens = outputs[:, inputs["input_ids"].shape[1]:]
            generated_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            
            # Check answers
            for gen_text, correct_answer in zip(generated_texts, batch_answers):
                predicted = extract_answer(gen_text)
                if predicted == correct_answer:
                    correct += 1
                total += 1
        
        subject_correct[subject] = correct
        subject_total[subject] = total
        
        accuracy = correct / total if total > 0 else 0
        results["subjects"][subject] = {
            "correct": correct,
            "total": total,
            "accuracy": accuracy,
        }
    
    # Aggregate by category
    for category in MMLU_CATEGORIES:
        cat_correct = 0
        cat_total = 0
        
        for subject in MMLU_SUBJECTS:
            if get_category_for_subject(subject) == category:
                cat_correct += subject_correct.get(subject, 0)
                cat_total += subject_total.get(subject, 0)
        
        accuracy = cat_correct / cat_total if cat_total > 0 else 0
        results["categories"][category] = {
            "correct": cat_correct,
            "total": cat_total,
            "accuracy": accuracy,
        }
    
    # Overall accuracy
    total_correct = sum(subject_correct.values())
    total_questions = sum(subject_total.values())
    results["overall"] = {
        "correct": total_correct,
        "total": total_questions,
        "accuracy": total_correct / total_questions if total_questions > 0 else 0,
    }
    
    return results


def save_results(results: Dict, output_dir: str, model_name: str):
    """
    Save evaluation results to JSON file.
    
    Args:
        results: Results dictionary
        output_dir: Directory to save results
        model_name: Name to use in filename
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Clean model name for filename
    clean_name = model_name.replace("/", "_").replace("\\", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mmlu_{clean_name}_{timestamp}.json"
    
    filepath = output_path / filename
    
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {filepath}")
    return filepath


def print_results(results: Dict):
    """Print formatted evaluation results."""
    print("\n" + "=" * 60)
    print("MMLU Evaluation Results")
    print("=" * 60)
    
    print(f"\nOverall Accuracy: {results['overall']['accuracy']:.1%}")
    print(f"  ({results['overall']['correct']}/{results['overall']['total']} questions)")
    
    print("\n" + "-" * 40)
    print("Results by Category:")
    print("-" * 40)
    
    for category in sorted(results["categories"].keys()):
        cat_data = results["categories"][category]
        print(f"  {category}: {cat_data['accuracy']:.1%} ({cat_data['correct']}/{cat_data['total']})")
    
    print("\n" + "-" * 40)
    print("Results by Subject (top 10 / bottom 10):")
    print("-" * 40)
    
    sorted_subjects = sorted(
        results["subjects"].items(),
        key=lambda x: x[1]["accuracy"],
        reverse=True,
    )
    
    print("\nTop 10:")
    for subject, data in sorted_subjects[:10]:
        print(f"  {subject}: {data['accuracy']:.1%}")
    
    print("\nBottom 10:")
    for subject, data in sorted_subjects[-10:]:
        print(f"  {subject}: {data['accuracy']:.1%}")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load MMLU data
    all_data = load_mmlu_data()
    
    # Select few-shot examples
    fewshot_examples, excluded_indices = select_fewshot_examples(
        all_data,
        num_examples=args.num_fewshot,
        seed=args.seed,
    )
    
    print(f"\nSelected {len(fewshot_examples)} few-shot examples")
    for i, ex in enumerate(fewshot_examples):
        print(f"  {i+1}. {ex['subject']}: {ex['question'][:50]}...")
    
    # Determine models to evaluate
    if args.checkpoint:
        models_to_eval = [args.checkpoint]
    elif args.model:
        models_to_eval = [args.model]
    else:
        # Evaluate all checkpoints in directory
        checkpoint_dir = Path(args.checkpoint_dir)
        models_to_eval = [
            str(p) for p in sorted(checkpoint_dir.iterdir())
            if p.is_dir() and (p / "config.json").exists()
        ]
        print(f"\nFound {len(models_to_eval)} checkpoints to evaluate")
    
    # Evaluate each model
    all_results = []
    
    for model_path in models_to_eval:
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {model_path}")
        print("=" * 60)
        
        results = evaluate_model(
            model_path=model_path,
            all_data=all_data,
            fewshot_examples=fewshot_examples,
            excluded_indices=excluded_indices,
            batch_size=args.batch_size,
            device=args.device,
        )
        
        # Print results
        print_results(results)
        
        # Save results
        model_name = Path(model_path).name if "/" not in model_path or "\\" not in model_path else model_path.split("/")[-1]
        save_results(results, args.output_dir, model_name)
        
        all_results.append(results)
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
