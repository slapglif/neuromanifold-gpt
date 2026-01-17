"""
Zero-shot benchmark evaluation for language models.

Implements evaluation on standard NLP benchmarks:
- LAMBADA: Perplexity on final word prediction
- HellaSwag, PIQA, WinoGrande: Multiple-choice accuracy metrics

Usage:
    from neuromanifold_gpt.benchmarks.zero_shot import evaluate_lambada

    perplexity = evaluate_lambada(
        model=model,
        device='cuda',
        max_examples=None  # Evaluate on full dataset
    )
"""
import math
import os
from contextlib import nullcontext
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from neuromanifold_gpt.benchmarks.datasets import (
    download_arc,
    download_hellaswag,
    download_lambada,
    download_mmlu,
    download_piqa,
    download_winogrande,
    load_jsonl,
    load_labels,
)
from neuromanifold_gpt.model.gpt import NeuroManifoldGPT


def evaluate_lambada(
    model: torch.nn.Module,
    tokenizer: Any,
    device: str = "cuda",
    dtype: str = "bfloat16",
    max_examples: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Evaluate model on LAMBADA benchmark (final word prediction perplexity).

    LAMBADA tests a model's ability to predict the final word of a passage
    that requires understanding broad context.

    Args:
        model: Language model to evaluate (NeuroManifoldGPT or GPT)
        tokenizer: Tokenizer with encode() method
        device: Device to run evaluation on ('cuda' or 'cpu')
        dtype: Data type for autocast ('bfloat16', 'float16', or 'float32')
        max_examples: Maximum number of examples to evaluate (None = all)
        verbose: Print progress information

    Returns:
        Dictionary with metrics:
            - perplexity: Overall perplexity on final token
            - loss: Average negative log-likelihood
            - accuracy: Exact match accuracy on final token
            - num_examples: Number of examples evaluated
    """
    # Download and load LAMBADA dataset
    dataset_path = download_lambada()
    examples = load_jsonl(dataset_path)

    if max_examples is not None:
        examples = examples[:max_examples]

    if verbose:
        print(f"\nEvaluating LAMBADA ({len(examples)} examples)...")

    # Setup autocast context
    device_type = "cuda" if "cuda" in device else "cpu"
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )

    model.eval()

    total_loss = 0.0
    total_correct = 0
    num_evaluated = 0

    with torch.no_grad():
        with ctx:
            for i, example in enumerate(examples):
                # Each LAMBADA example has a 'text' field with the full passage
                text = example["text"]

                # Tokenize the full text
                tokens = tokenizer.encode(text)

                if len(tokens) < 2:
                    # Skip examples that are too short
                    continue

                # Convert to tensor
                x = torch.tensor(
                    tokens[:-1], dtype=torch.long, device=device
                ).unsqueeze(0)
                y_target = tokens[-1]  # Final token to predict

                # Forward pass
                # Handle different model interfaces
                if isinstance(model, NeuroManifoldGPT):
                    # NeuroManifoldGPT returns (logits, loss, aux_losses)
                    logits, _, _ = model(x)
                else:
                    # Standard GPT returns (logits, loss)
                    logits, _ = model(x)

                # Get logits for the last position (predicting final token)
                final_logits = logits[0, -1, :]  # Shape: [vocab_size]

                # Calculate loss (negative log-likelihood)
                loss = F.cross_entropy(
                    final_logits.unsqueeze(0), torch.tensor([y_target], device=device)
                )
                total_loss += loss.item()

                # Check if prediction is correct
                pred_token = torch.argmax(final_logits).item()
                if pred_token == y_target:
                    total_correct += 1

                num_evaluated += 1

                # Progress update
                if verbose and (i + 1) % 100 == 0:
                    current_ppl = math.exp(total_loss / num_evaluated)
                    current_acc = total_correct / num_evaluated
                    print(
                        f"  Progress: {i + 1}/{len(examples)} - PPL: {current_ppl:.2f}, Acc: {current_acc:.4f}"
                    )

    # Calculate final metrics
    avg_loss = total_loss / num_evaluated
    perplexity = math.exp(avg_loss)
    accuracy = total_correct / num_evaluated

    results = {
        "perplexity": perplexity,
        "loss": avg_loss,
        "accuracy": accuracy,
        "num_examples": num_evaluated,
    }

    if verbose:
        print("\nLAMBADA Results:")
        print(f"  Perplexity: {perplexity:.2f}")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Examples: {num_evaluated}")

    return results


def evaluate_multiple_choice(
    model: torch.nn.Module,
    tokenizer: Any,
    benchmark: str,
    device: str = "cuda",
    dtype: str = "bfloat16",
    max_examples: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Evaluate model on multiple-choice benchmarks (HellaSwag, PIQA, WinoGrande).

    For each question, compute the log-likelihood of each choice and select
    the one with the highest likelihood.

    Args:
        model: Language model to evaluate (NeuroManifoldGPT or GPT)
        tokenizer: Tokenizer with encode() method
        benchmark: Benchmark name ('hellaswag', 'piqa', or 'winogrande')
        device: Device to run evaluation on ('cuda' or 'cpu')
        dtype: Data type for autocast ('bfloat16', 'float16', or 'float32')
        max_examples: Maximum number of examples to evaluate (None = all)
        verbose: Print progress information

    Returns:
        Dictionary with metrics:
            - accuracy: Proportion of correctly answered questions
            - num_examples: Number of examples evaluated
    """
    if benchmark == "hellaswag":
        dataset_path = download_hellaswag()
        examples = load_jsonl(dataset_path)

        # HellaSwag examples have 'ctx', 'endings', and 'label' fields
        def get_context(ex):
            return ex["ctx"]

        def get_choices(ex):
            return ex["endings"]

        def get_label(ex):
            return int(ex["label"])

    elif benchmark == "piqa":
        paths = download_piqa()
        examples = load_jsonl(paths["questions"])
        labels = load_labels(paths["labels"])

        # PIQA examples have 'goal', 'sol1', 'sol2'
        def get_context(ex):
            return ex["goal"]

        def get_choices(ex):
            return [ex["sol1"], ex["sol2"]]

        def get_label(ex_idx):
            return labels[ex_idx]

    elif benchmark == "winogrande":
        paths = download_winogrande()
        examples = load_jsonl(paths["questions"])
        labels = load_labels(paths["labels"])

        # WinoGrande examples have 'sentence', 'option1', 'option2'
        def get_context(ex):
            return ex["sentence"]

        def get_choices(ex):
            return [ex["option1"], ex["option2"]]

        def get_label(ex_idx):
            return labels[ex_idx] - 1  # Convert 1-indexed to 0-indexed

    else:
        raise ValueError(
            f"Unknown benchmark: {benchmark}. Must be 'hellaswag', 'piqa', or 'winogrande'"
        )

    if max_examples is not None:
        examples = examples[:max_examples]
        if benchmark in ["piqa", "winogrande"]:
            labels = labels[:max_examples]

    if verbose:
        print(f"\nEvaluating {benchmark.upper()} ({len(examples)} examples)...")

    # Setup autocast context
    device_type = "cuda" if "cuda" in device else "cpu"
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )

    model.eval()

    total_correct = 0
    num_evaluated = 0

    with torch.no_grad():
        with ctx:
            for i, example in enumerate(examples):
                # Get context and choices
                if benchmark in ["piqa", "winogrande"]:
                    context = get_context(example)
                    choices = get_choices(example)
                    label = get_label(i)
                else:
                    context = get_context(example)
                    choices = get_choices(example)
                    label = get_label(example)

                # Compute log-likelihood for each choice
                choice_lls = []
                for choice in choices:
                    # Concatenate context and choice
                    full_text = context + " " + choice
                    tokens = tokenizer.encode(full_text)

                    if len(tokens) < 2:
                        # Skip invalid examples
                        choice_lls.append(float("-inf"))
                        continue

                    # Convert to tensor
                    x = torch.tensor(
                        tokens[:-1], dtype=torch.long, device=device
                    ).unsqueeze(0)
                    torch.tensor(tokens[1:], dtype=torch.long, device=device).unsqueeze(
                        0
                    )

                    # Forward pass
                    if isinstance(model, NeuroManifoldGPT):
                        # NeuroManifoldGPT returns (logits, loss, aux_losses)
                        logits, _, _ = model(x)
                    else:
                        # Standard GPT returns (logits, loss)
                        logits, _ = model(x)

                    # Calculate log-likelihood of the choice
                    # Use only the tokens corresponding to the choice part
                    context_tokens = tokenizer.encode(context)
                    choice_start_idx = len(context_tokens)

                    # Sum log-likelihoods for choice tokens
                    log_probs = F.log_softmax(logits[0, choice_start_idx - 1 :], dim=-1)
                    choice_ll = 0.0
                    for j, token_id in enumerate(tokens[choice_start_idx:]):
                        if j < log_probs.size(0):
                            choice_ll += log_probs[j, token_id].item()

                    choice_lls.append(choice_ll)

                # Select choice with highest log-likelihood
                pred_choice = choice_lls.index(max(choice_lls))
                if pred_choice == label:
                    total_correct += 1

                num_evaluated += 1

                # Progress update
                if verbose and (i + 1) % 100 == 0:
                    current_acc = total_correct / num_evaluated
                    print(
                        f"  Progress: {i + 1}/{len(examples)} - Acc: {current_acc:.4f}"
                    )

    # Calculate final metrics
    accuracy = total_correct / num_evaluated

    results = {
        "accuracy": accuracy,
        "num_examples": num_evaluated,
    }

    if verbose:
        print(f"\n{benchmark.upper()} Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Examples: {num_evaluated}")

    return results


def evaluate_mmlu(
    model: torch.nn.Module,
    tokenizer: Any,
    device: str = "cuda",
    dtype: str = "bfloat16",
    max_examples: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Evaluate model on MMLU (Massive Multitask Language Understanding) benchmark.

    MMLU is a multiple-choice dataset covering 57 subjects for evaluating
    multitask accuracy across diverse knowledge domains.

    For each question, compute the log-likelihood of each choice and select
    the one with the highest likelihood.

    Args:
        model: Language model to evaluate (NeuroManifoldGPT or GPT)
        tokenizer: Tokenizer with encode() method
        device: Device to run evaluation on ('cuda' or 'cpu')
        dtype: Data type for autocast ('bfloat16', 'float16', or 'float32')
        max_examples: Maximum number of examples to evaluate per subject (None = all)
        verbose: Print progress information

    Returns:
        Dictionary with metrics:
            - accuracy: Overall accuracy across all subjects
            - num_examples: Total number of examples evaluated
            - subject_accuracies: Dictionary mapping subject names to accuracies
    """
    import csv
    import glob

    # Download and get MMLU dataset directory
    dataset_dir = download_mmlu()

    # Load all test CSV files from subjects
    test_dir = os.path.join(dataset_dir, "test")
    if not os.path.exists(test_dir):
        # Try alternate structure (data/test/)
        test_dir = os.path.join(dataset_dir, "data", "test")

    if not os.path.exists(test_dir):
        raise ValueError(f"MMLU test directory not found at {dataset_dir}")

    # Get all CSV files in test directory
    csv_files = glob.glob(os.path.join(test_dir, "*.csv"))

    if not csv_files:
        raise ValueError(f"No CSV files found in {test_dir}")

    # For small sample testing, limit number of subjects to avoid OOM
    if max_examples is not None and max_examples <= 10:
        csv_files = sorted(csv_files)[:5]  # Only evaluate 5 subjects for quick testing

    if verbose:
        print(f"\nEvaluating MMLU ({len(csv_files)} subjects)...")

    # Setup autocast context
    device_type = "cuda" if "cuda" in device else "cpu"
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )

    model.eval()

    total_correct = 0
    total_evaluated = 0
    subject_accuracies = {}

    with torch.no_grad():
        with ctx:
            for csv_file in csv_files:
                # Extract subject name from filename
                subject_name = (
                    os.path.basename(csv_file).replace(".csv", "").replace("_", " ")
                )

                # Load questions from CSV
                questions = []
                with open(csv_file, "r", encoding="utf-8") as f:
                    csv_reader = csv.reader(f)
                    for row in csv_reader:
                        if len(row) >= 6:
                            # MMLU format: Question, ChoiceA, ChoiceB, ChoiceC, ChoiceD, Answer
                            questions.append(
                                {
                                    "question": row[0],
                                    "choices": [row[1], row[2], row[3], row[4]],
                                    "answer": row[5],  # 'A', 'B', 'C', or 'D'
                                }
                            )

                if max_examples is not None:
                    questions = questions[:max_examples]

                subject_correct = 0
                subject_evaluated = 0

                for i, question_data in enumerate(questions):
                    question = question_data["question"]
                    choices = question_data["choices"]
                    answer_letter = question_data["answer"].strip().upper()

                    # Convert answer letter to index (A=0, B=1, C=2, D=3)
                    if answer_letter not in ["A", "B", "C", "D"]:
                        continue
                    label = ord(answer_letter) - ord("A")

                    # Compute log-likelihood for each choice
                    choice_lls = []
                    for choice in choices:
                        # Concatenate question and choice
                        full_text = question + " " + choice
                        tokens = tokenizer.encode(full_text)

                        if len(tokens) < 2:
                            # Skip invalid examples
                            choice_lls.append(float("-inf"))
                            continue

                        # Convert to tensor
                        x = torch.tensor(
                            tokens[:-1], dtype=torch.long, device=device
                        ).unsqueeze(0)

                        # Forward pass
                        if isinstance(model, NeuroManifoldGPT):
                            # NeuroManifoldGPT returns (logits, loss, aux_losses)
                            logits, _, _ = model(x)
                        else:
                            # Standard GPT returns (logits, loss)
                            logits, _ = model(x)

                        # Calculate log-likelihood of the choice
                        # Use only the tokens corresponding to the choice part
                        question_tokens = tokenizer.encode(question)
                        choice_start_idx = len(question_tokens)

                        # Sum log-likelihoods for choice tokens
                        log_probs = F.log_softmax(
                            logits[0, choice_start_idx - 1 :], dim=-1
                        )
                        choice_ll = 0.0
                        for j, token_id in enumerate(tokens[choice_start_idx:]):
                            if j < log_probs.size(0):
                                choice_ll += log_probs[j, token_id].item()

                        choice_lls.append(choice_ll)

                    # Select choice with highest log-likelihood
                    pred_choice = choice_lls.index(max(choice_lls))
                    if pred_choice == label:
                        subject_correct += 1
                        total_correct += 1

                    subject_evaluated += 1
                    total_evaluated += 1

                # Calculate subject accuracy
                if subject_evaluated > 0:
                    subject_accuracy = subject_correct / subject_evaluated
                    subject_accuracies[subject_name] = subject_accuracy

                    if verbose:
                        print(
                            f"  {subject_name}: {subject_accuracy:.4f} ({subject_correct}/{subject_evaluated})"
                        )

    # Calculate overall metrics
    accuracy = total_correct / total_evaluated if total_evaluated > 0 else 0.0

    results = {
        "accuracy": accuracy,
        "num_examples": total_evaluated,
        "subject_accuracies": subject_accuracies,
    }

    if verbose:
        print("\nMMLU Overall Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Examples: {total_evaluated}")
        print(f"  Subjects: {len(subject_accuracies)}")

    return results


def evaluate_arc(
    model: torch.nn.Module,
    tokenizer: Any,
    variant: str = "easy",
    device: str = "cuda",
    dtype: str = "bfloat16",
    max_examples: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Evaluate model on ARC (AI2 Reasoning Challenge) benchmark.

    ARC is a multiple-choice question answering dataset for science questions
    with two difficulty levels: Easy and Challenge.

    For each question, compute the log-likelihood of each choice and select
    the one with the highest likelihood.

    Args:
        model: Language model to evaluate (NeuroManifoldGPT or GPT)
        tokenizer: Tokenizer with encode() method
        variant: ARC variant to evaluate ('easy' or 'challenge')
        device: Device to run evaluation on ('cuda' or 'cpu')
        dtype: Data type for autocast ('bfloat16', 'float16', or 'float32')
        max_examples: Maximum number of examples to evaluate (None = all)
        verbose: Print progress information

    Returns:
        Dictionary with metrics:
            - accuracy: Proportion of correctly answered questions
            - num_examples: Number of examples evaluated
            - variant: Which ARC variant was evaluated
    """
    if variant not in ["easy", "challenge"]:
        raise ValueError(
            f"Unknown ARC variant: {variant}. Must be 'easy' or 'challenge'"
        )

    # Download and load ARC dataset
    paths = download_arc()
    dataset_path = paths[variant]
    examples = load_jsonl(dataset_path)

    if max_examples is not None:
        examples = examples[:max_examples]

    if verbose:
        print(f"\nEvaluating ARC-{variant.capitalize()} ({len(examples)} examples)...")

    # Setup autocast context
    device_type = "cuda" if "cuda" in device else "cpu"
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )

    model.eval()

    total_correct = 0
    num_evaluated = 0

    with torch.no_grad():
        with ctx:
            for i, example in enumerate(examples):
                # ARC examples have 'question', 'choices', and 'answerKey' fields
                # 'question' contains the question text
                # 'choices' is a dict with 'text' list and 'label' list
                # 'answerKey' is the correct answer (e.g., 'A', 'B', 'C', 'D')
                question_text = example["question"]["stem"]
                choices = example["question"]["choices"]
                answer_key = example["answerKey"]

                # Extract choice texts and labels
                choice_texts = [choice["text"] for choice in choices]
                choice_labels = [choice["label"] for choice in choices]

                # Find the index of the correct answer
                try:
                    label_idx = choice_labels.index(answer_key)
                except ValueError:
                    # Skip if answer key not found in choices
                    continue

                # Compute log-likelihood for each choice
                choice_lls = []
                for choice_text in choice_texts:
                    # Concatenate question and choice
                    full_text = question_text + " " + choice_text
                    tokens = tokenizer.encode(full_text)

                    if len(tokens) < 2:
                        # Skip invalid examples
                        choice_lls.append(float("-inf"))
                        continue

                    # Convert to tensor
                    x = torch.tensor(
                        tokens[:-1], dtype=torch.long, device=device
                    ).unsqueeze(0)

                    # Forward pass
                    if isinstance(model, NeuroManifoldGPT):
                        # NeuroManifoldGPT returns (logits, loss, aux_losses)
                        logits, _, _ = model(x)
                    else:
                        # Standard GPT returns (logits, loss)
                        logits, _ = model(x)

                    # Calculate log-likelihood of the choice
                    # Use only the tokens corresponding to the choice part
                    question_tokens = tokenizer.encode(question_text)
                    choice_start_idx = len(question_tokens)

                    # Sum log-likelihoods for choice tokens
                    log_probs = F.log_softmax(logits[0, choice_start_idx - 1 :], dim=-1)
                    choice_ll = 0.0
                    for j, token_id in enumerate(tokens[choice_start_idx:]):
                        if j < log_probs.size(0):
                            choice_ll += log_probs[j, token_id].item()

                    choice_lls.append(choice_ll)

                # Select choice with highest log-likelihood
                pred_choice = choice_lls.index(max(choice_lls))
                if pred_choice == label_idx:
                    total_correct += 1

                num_evaluated += 1

                # Progress update
                if verbose and (i + 1) % 100 == 0:
                    current_acc = total_correct / num_evaluated
                    print(
                        f"  Progress: {i + 1}/{len(examples)} - Acc: {current_acc:.4f}"
                    )

    # Calculate final metrics
    accuracy = total_correct / num_evaluated if num_evaluated > 0 else 0.0

    results = {
        "accuracy": accuracy,
        "num_examples": num_evaluated,
        "variant": variant,
    }

    if verbose:
        print(f"\nARC-{variant.capitalize()} Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Examples: {num_evaluated}")

    return results
