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
import json
import math
from typing import Optional, Dict, Any, List, Tuple, Protocol, Union, cast
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from neuromanifold_gpt.benchmarks.datasets import (
    download_lambada,
    download_hellaswag,
    download_piqa,
    download_winogrande,
    load_jsonl,
    load_labels,
)


class Tokenizer(Protocol):
    """Protocol for tokenizer objects used in zero-shot evaluation."""

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        ...

    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text."""
        ...


def evaluate_lambada(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    device: str = 'cuda',
    dtype: str = 'bfloat16',
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
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx: Any = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    model.eval()

    total_loss = 0.0
    total_correct = 0
    num_evaluated = 0

    with torch.no_grad():
        with ctx:
            for i, example in enumerate(examples):
                # Each LAMBADA example has a 'text' field with the full passage
                text = example['text']

                # Tokenize the full text
                tokens = tokenizer.encode(text)

                if len(tokens) < 2:
                    # Skip examples that are too short
                    continue

                # Convert to tensor
                x = torch.tensor(tokens[:-1], dtype=torch.long, device=device).unsqueeze(0)
                y_target = tokens[-1]  # Final token to predict

                # Forward pass
                # Handle different model interfaces
                if hasattr(model, 'config'):
                    # NeuroManifoldGPT or models with config
                    logits, _, _ = model(x)
                else:
                    # Standard GPT
                    logits = model(x)

                # Get logits for the last position (predicting final token)
                final_logits = logits[0, -1, :]  # Shape: [vocab_size]

                # Calculate loss (negative log-likelihood)
                loss = F.cross_entropy(final_logits.unsqueeze(0), torch.tensor([y_target], device=device))
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
                    print(f"  Progress: {i + 1}/{len(examples)} - PPL: {current_ppl:.2f}, Acc: {current_acc:.4f}")

    # Calculate final metrics
    avg_loss = total_loss / num_evaluated
    perplexity = math.exp(avg_loss)
    accuracy = total_correct / num_evaluated

    results = {
        'perplexity': perplexity,
        'loss': avg_loss,
        'accuracy': accuracy,
        'num_examples': num_evaluated,
    }

    if verbose:
        print(f"\nLAMBADA Results:")
        print(f"  Perplexity: {perplexity:.2f}")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Examples: {num_evaluated}")

    return results


def evaluate_multiple_choice(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    benchmark: str,
    device: str = 'cuda',
    dtype: str = 'bfloat16',
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
    if benchmark == 'hellaswag':
        dataset_path = download_hellaswag()
        examples = load_jsonl(dataset_path)
        # HellaSwag examples have 'ctx', 'endings', and 'label' fields
        get_context = lambda ex: ex['ctx']
        get_choices = lambda ex: ex['endings']
        get_label = lambda ex: int(ex['label'])

    elif benchmark == 'piqa':
        paths = download_piqa()
        examples = load_jsonl(paths['questions'])
        labels = load_labels(paths['labels'])
        # PIQA examples have 'goal', 'sol1', 'sol2'
        get_context = lambda ex: ex['goal']
        get_choices = lambda ex: [ex['sol1'], ex['sol2']]
        get_label = lambda ex_idx: labels[ex_idx]

    elif benchmark == 'winogrande':
        paths = download_winogrande()
        examples = load_jsonl(paths['questions'])
        labels = load_labels(paths['labels'])
        # WinoGrande examples have 'sentence', 'option1', 'option2'
        get_context = lambda ex: ex['sentence']
        get_choices = lambda ex: [ex['option1'], ex['option2']]
        get_label = lambda ex_idx: labels[ex_idx] - 1  # Convert 1-indexed to 0-indexed

    else:
        raise ValueError(f"Unknown benchmark: {benchmark}. Must be 'hellaswag', 'piqa', or 'winogrande'")

    if max_examples is not None:
        examples = examples[:max_examples]
        if benchmark in ['piqa', 'winogrande']:
            labels = labels[:max_examples]

    if verbose:
        print(f"\nEvaluating {benchmark.upper()} ({len(examples)} examples)...")

    # Setup autocast context
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx: Any = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    model.eval()

    total_correct = 0
    num_evaluated = 0

    with torch.no_grad():
        with ctx:
            for i, example in enumerate(examples):
                # Get context and choices
                if benchmark in ['piqa', 'winogrande']:
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
                    full_text = context + ' ' + choice
                    tokens = tokenizer.encode(full_text)

                    if len(tokens) < 2:
                        # Skip invalid examples
                        choice_lls.append(float('-inf'))
                        continue

                    # Convert to tensor
                    x = torch.tensor(tokens[:-1], dtype=torch.long, device=device).unsqueeze(0)
                    y = torch.tensor(tokens[1:], dtype=torch.long, device=device).unsqueeze(0)

                    # Forward pass
                    if hasattr(model, 'config'):
                        # NeuroManifoldGPT or models with config
                        logits, _, _ = model(x)
                    else:
                        # Standard GPT
                        logits = model(x)

                    # Calculate log-likelihood of the choice
                    # Use only the tokens corresponding to the choice part
                    context_tokens = tokenizer.encode(context)
                    choice_start_idx = len(context_tokens)

                    # Sum log-likelihoods for choice tokens
                    log_probs = F.log_softmax(logits[0, choice_start_idx-1:], dim=-1)
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
                    print(f"  Progress: {i + 1}/{len(examples)} - Acc: {current_acc:.4f}")

    # Calculate final metrics
    accuracy = total_correct / num_evaluated

    results = {
        'accuracy': accuracy,
        'num_examples': num_evaluated,
    }

    if verbose:
        print(f"\n{benchmark.upper()} Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Examples: {num_evaluated}")

    return results
