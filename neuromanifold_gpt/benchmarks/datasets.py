"""
Benchmark dataset downloader and preprocessor.

Downloads standard NLP benchmark datasets (LAMBADA, HellaSwag, MMLU, PIQA, WinoGrande)
and caches them locally for evaluation. Follows the pattern from data/shakespeare_char/prepare.py.
"""
import json
import os
from typing import Dict, List, Optional

import requests

# Base directory for caching benchmark datasets
BENCHMARK_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "benchmarks"
)


def ensure_benchmark_dir():
    """Create benchmark directory if it doesn't exist."""
    os.makedirs(BENCHMARK_DIR, exist_ok=True)


def download_lambada(cache_dir: Optional[str] = None) -> str:
    """
    Download the LAMBADA dataset for language modeling evaluation.

    LAMBADA tests a model's ability to predict the final word of a passage
    that requires understanding broad context.

    Args:
        cache_dir: Optional custom cache directory. Defaults to data/benchmarks/lambada

    Returns:
        Path to the downloaded dataset file
    """
    ensure_benchmark_dir()

    if cache_dir is None:
        cache_dir = os.path.join(BENCHMARK_DIR, "lambada")
    os.makedirs(cache_dir, exist_ok=True)

    output_file = os.path.join(cache_dir, "lambada_test.jsonl")

    if not os.path.exists(output_file):
        # Download from OpenAI public repository
        data_url = (
            "https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl"
        )

        print(f"Downloading LAMBADA dataset from {data_url}...")
        response = requests.get(data_url)
        response.raise_for_status()

        # Save the JSONL file directly
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(response.text)

        # Count examples
        lines = response.text.strip().split("\n")
        print(f"LAMBADA dataset downloaded: {len(lines)} examples")
    else:
        print(f"LAMBADA dataset already cached at {output_file}")

    return output_file


def download_hellaswag(cache_dir: Optional[str] = None) -> str:
    """
    Download the HellaSwag dataset for commonsense reasoning evaluation.

    HellaSwag is a multiple-choice dataset for commonsense natural language inference.

    Args:
        cache_dir: Optional custom cache directory. Defaults to data/benchmarks/hellaswag

    Returns:
        Path to the downloaded dataset file
    """
    ensure_benchmark_dir()

    if cache_dir is None:
        cache_dir = os.path.join(BENCHMARK_DIR, "hellaswag")
    os.makedirs(cache_dir, exist_ok=True)

    output_file = os.path.join(cache_dir, "hellaswag_val.jsonl")

    if not os.path.exists(output_file):
        # Download from HuggingFace datasets repository
        data_url = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"

        print(f"Downloading HellaSwag dataset from {data_url}...")
        response = requests.get(data_url)
        response.raise_for_status()

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(response.text)

        # Count examples
        num_examples = len(response.text.strip().split("\n"))
        print(f"HellaSwag dataset downloaded: {num_examples} examples")
    else:
        print(f"HellaSwag dataset already cached at {output_file}")

    return output_file


def download_mmlu(cache_dir: Optional[str] = None) -> str:
    """
    Download the MMLU dataset for multitask language understanding evaluation.

    MMLU is a multiple-choice dataset covering 57 subjects for evaluating
    multitask accuracy across diverse knowledge domains.

    Args:
        cache_dir: Optional custom cache directory. Defaults to data/benchmarks/mmlu

    Returns:
        Path to the downloaded dataset directory
    """
    import io
    import tarfile

    ensure_benchmark_dir()

    if cache_dir is None:
        cache_dir = os.path.join(BENCHMARK_DIR, "mmlu")
    os.makedirs(cache_dir, exist_ok=True)

    # Check if data already exists (look for dev directory)
    dev_dir = os.path.join(cache_dir, "dev")

    if not os.path.exists(dev_dir):
        # Download from original Berkeley repository
        data_url = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"

        print(f"Downloading MMLU dataset from {data_url}...")
        response = requests.get(data_url)
        response.raise_for_status()

        print("Extracting MMLU dataset...")
        with tarfile.open(fileobj=io.BytesIO(response.content), mode="r") as tar:
            tar.extractall(path=cache_dir)

        # Count subjects in dev set
        if os.path.exists(dev_dir):
            num_subjects = len([f for f in os.listdir(dev_dir) if f.endswith(".csv")])
            print(f"MMLU dataset extracted: {num_subjects} subjects")
        else:
            print("MMLU dataset extracted")
    else:
        print(f"MMLU dataset already cached at {cache_dir}")

    return cache_dir


def download_piqa(cache_dir: Optional[str] = None) -> Dict[str, str]:
    """
    Download the PIQA dataset for physical commonsense reasoning evaluation.

    PIQA tests physical commonsense reasoning through multiple-choice questions.

    Args:
        cache_dir: Optional custom cache directory. Defaults to data/benchmarks/piqa

    Returns:
        Dictionary with paths to 'questions', 'labels', and 'solutions' files
    """
    ensure_benchmark_dir()

    if cache_dir is None:
        cache_dir = os.path.join(BENCHMARK_DIR, "piqa")
    os.makedirs(cache_dir, exist_ok=True)

    # PIQA has separate files for questions, labels, and solutions
    files = {
        "questions": "valid.jsonl",
        "labels": "valid-labels.lst",
    }

    base_url = "https://yonatanbisk.com/piqa/data"

    output_paths = {}
    for file_type, filename in files.items():
        output_file = os.path.join(cache_dir, filename)
        output_paths[file_type] = output_file

        if not os.path.exists(output_file):
            data_url = f"{base_url}/{filename}"

            print(f"Downloading PIQA {file_type} from {data_url}...")
            response = requests.get(data_url)
            response.raise_for_status()

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(response.text)

            print(f"PIQA {file_type} downloaded")
        else:
            print(f"PIQA {file_type} already cached at {output_file}")

    return output_paths


def download_arc(cache_dir: Optional[str] = None) -> Dict[str, str]:
    """
    Download the ARC dataset for question answering evaluation.

    ARC (AI2 Reasoning Challenge) is a multiple-choice question answering dataset
    with two difficulty levels: Challenge and Easy.

    Args:
        cache_dir: Optional custom cache directory. Defaults to data/benchmarks/arc

    Returns:
        Dictionary with paths to 'challenge' and 'easy' dataset files
    """
    import io
    import zipfile

    ensure_benchmark_dir()

    if cache_dir is None:
        cache_dir = os.path.join(BENCHMARK_DIR, "arc")
    os.makedirs(cache_dir, exist_ok=True)

    # ARC test set files
    challenge_file = os.path.join(cache_dir, "ARC-Challenge-Test.jsonl")
    easy_file = os.path.join(cache_dir, "ARC-Easy-Test.jsonl")

    if not os.path.exists(challenge_file) or not os.path.exists(easy_file):
        # Download and extract from zip file
        zip_url = (
            "https://s3-us-west-2.amazonaws.com/ai2-website/data/ARC-V1-Feb2018.zip"
        )

        print(f"Downloading ARC dataset from {zip_url}...")
        response = requests.get(zip_url)
        response.raise_for_status()

        print("Extracting ARC dataset...")
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
            # Extract Challenge test set
            challenge_path = "ARC-V1-Feb2018-2/ARC-Challenge/ARC-Challenge-Test.jsonl"
            with zip_file.open(challenge_path) as source:
                with open(challenge_file, "wb") as target:
                    target.write(source.read())

            # Extract Easy test set
            easy_path = "ARC-V1-Feb2018-2/ARC-Easy/ARC-Easy-Test.jsonl"
            with zip_file.open(easy_path) as source:
                with open(easy_file, "wb") as target:
                    target.write(source.read())

        # Count examples
        with open(challenge_file, "r", encoding="utf-8") as f:
            challenge_examples = len(f.read().strip().split("\n"))
        with open(easy_file, "r", encoding="utf-8") as f:
            easy_examples = len(f.read().strip().split("\n"))

        print(
            f"ARC dataset extracted: {challenge_examples} challenge examples, {easy_examples} easy examples"
        )
    else:
        print(f"ARC dataset already cached at {cache_dir}")

    return {
        "challenge": challenge_file,
        "easy": easy_file,
    }


def download_winogrande(
    cache_dir: Optional[str] = None, size: str = "xl"
) -> Dict[str, str]:
    """
    Download the WinoGrande dataset for pronoun resolution evaluation.

    WinoGrande is a large-scale dataset for commonsense reasoning focused on
    pronoun resolution.

    Args:
        cache_dir: Optional custom cache directory. Defaults to data/benchmarks/winogrande
        size: Dataset size variant - not used for dev set (always uses full dev set).
               Kept for API compatibility.

    Returns:
        Dictionary with paths to dataset files
    """
    import io
    import zipfile

    ensure_benchmark_dir()

    if cache_dir is None:
        cache_dir = os.path.join(BENCHMARK_DIR, "winogrande")
    os.makedirs(cache_dir, exist_ok=True)

    # WinoGrande validation set (dev set has no size variants)
    output_file = os.path.join(cache_dir, "dev.jsonl")
    labels_file = os.path.join(cache_dir, "dev-labels.lst")

    if not os.path.exists(output_file) or not os.path.exists(labels_file):
        # Download and extract from zip file
        zip_url = "https://storage.googleapis.com/ai2-mosaic/public/winogrande/winogrande_1.1.zip"

        print(f"Downloading WinoGrande dataset from {zip_url}...")
        response = requests.get(zip_url)
        response.raise_for_status()

        print("Extracting WinoGrande dataset...")
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
            # Extract specific files we need from dev set
            questions_path = "winogrande_1.1/dev.jsonl"
            labels_path = "winogrande_1.1/dev-labels.lst"

            # Extract questions
            with zip_file.open(questions_path) as source:
                with open(output_file, "wb") as target:
                    target.write(source.read())

            # Extract labels
            with zip_file.open(labels_path) as source:
                with open(labels_file, "wb") as target:
                    target.write(source.read())

        # Count examples
        with open(output_file, "r", encoding="utf-8") as f:
            num_examples = len(f.read().strip().split("\n"))

        print(f"WinoGrande dataset extracted: {num_examples} examples")
    else:
        print(f"WinoGrande dataset already cached at {output_file}")

    return {
        "questions": output_file,
        "labels": labels_file,
    }


def load_jsonl(file_path: str) -> List[Dict]:
    """
    Load a JSONL file and return list of examples.

    Args:
        file_path: Path to JSONL file

    Returns:
        List of dictionaries, one per line
    """
    examples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def load_labels(file_path: str) -> List[int]:
    """
    Load a labels file (one label per line).

    Args:
        file_path: Path to labels file

    Returns:
        List of integer labels
    """
    labels = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(int(line))
    return labels
