"""Dataset preprocessing and loading for GSM8K and SVAMP."""

import re
from datasets import load_dataset
from typing import Dict, List, Optional


def load_gsm8k(split: str = "test", cache_dir: str = ".cache", max_samples: Optional[int] = None) -> List[Dict]:
    """Load GSM8K dataset.
    
    Args:
        split: Dataset split (train or test)
        cache_dir: Cache directory for downloaded datasets
        max_samples: Maximum number of samples to load (for pilot runs)
        
    Returns:
        List of dictionaries with 'question' and 'answer' keys
    """
    dataset = load_dataset("gsm8k", "main", split=split, cache_dir=cache_dir)
    
    examples = []
    for idx, item in enumerate(dataset):
        if max_samples is not None and idx >= max_samples:
            break
            
        question = item["question"]
        answer_text = item["answer"]
        
        # Extract final numeric answer from "#### X" format
        answer_match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", answer_text)
        if answer_match:
            answer = float(answer_match.group(1))
        else:
            # Fallback: try to find any number at the end
            numbers = re.findall(r"-?\d+(?:\.\d+)?", answer_text)
            answer = float(numbers[-1]) if numbers else None
            
        examples.append({
            "id": f"gsm8k-{idx}",
            "question": question,
            "answer": answer,
            "answer_text": answer_text
        })
    
    return examples


def load_svamp(split: str = "test", cache_dir: str = ".cache", max_samples: Optional[int] = None) -> List[Dict]:
    """Load SVAMP dataset.
    
    Args:
        split: Dataset split (train or test)
        cache_dir: Cache directory for downloaded datasets
        max_samples: Maximum number of samples to load (for pilot runs)
        
    Returns:
        List of dictionaries with 'question' and 'answer' keys
    """
    # SVAMP is available on HuggingFace
    dataset = load_dataset("ChilleD/SVAMP", split=split, cache_dir=cache_dir)
    
    examples = []
    for idx, item in enumerate(dataset):
        if max_samples is not None and idx >= max_samples:
            break
            
        question = item["Body"] + " " + item["Question"]
        answer = float(item["Answer"])
        
        examples.append({
            "id": f"svamp-{idx}",
            "question": question.strip(),
            "answer": answer,
            "answer_text": str(answer)
        })
    
    return examples


def load_dataset_by_name(
    name: str,
    split: str = "test",
    cache_dir: str = ".cache",
    max_samples: Optional[int] = None
) -> List[Dict]:
    """Load dataset by name.
    
    Args:
        name: Dataset name (gsm8k or svamp)
        split: Dataset split
        cache_dir: Cache directory
        max_samples: Maximum samples to load
        
    Returns:
        List of dataset examples
    """
    if name == "gsm8k":
        return load_gsm8k(split, cache_dir, max_samples)
    elif name == "svamp":
        return load_svamp(split, cache_dir, max_samples)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def extract_answer_from_text(text: str) -> Optional[float]:
    """Extract numeric answer from model-generated text.
    
    Looks for patterns like:
    - "The answer is X"
    - "#### X"
    - Last number in the text
    
    Args:
        text: Model-generated text
        
    Returns:
        Extracted numeric answer or None
    """
    text = text.strip()
    
    # Pattern 1: "The answer is X" or "Therefore, X"
    patterns = [
        r"[Tt]he answer is\s*[:\-]?\s*(-?\d+(?:\.\d+)?)",
        r"[Tt]herefore,?\s*[:\-]?\s*(-?\d+(?:\.\d+)?)",
        r"####\s*(-?\d+(?:\.\d+)?)",
        r"[Ff]inal answer:\s*(-?\d+(?:\.\d+)?)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    
    # Pattern 2: Last number in the text
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            pass
    
    return None
