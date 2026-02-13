"""Inference script for FAB-SC and SC methods."""

import os
import sys
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from tqdm import tqdm

from src.preprocess import load_dataset_by_name, extract_answer_from_text
from src.model import LLMModel, create_cot_prompt, create_backsolve_prompt, create_falsification_prompt


def parse_backsolve_response(response: str) -> Tuple[float, str]:
    """Parse backsolve verification response.
    
    Returns:
        (confidence, verdict)
    """
    confidence = 0.0
    verdict = "FAIL"
    
    # Extract confidence
    conf_match = re.search(r"CONFIDENCE:\s*([0-9.]+)", response, re.IGNORECASE)
    if conf_match:
        try:
            confidence = float(conf_match.group(1))
            confidence = max(0.0, min(1.0, confidence))
        except ValueError:
            pass
    
    # Extract verdict
    verdict_match = re.search(r"VERDICT:\s*(PASS|FAIL)", response, re.IGNORECASE)
    if verdict_match:
        verdict = verdict_match.group(1).upper()
    
    return confidence, verdict


def parse_falsification_response(response: str) -> Tuple[float, str]:
    """Parse falsification response.
    
    Returns:
        (error_prob, reason)
    """
    error_prob = 0.5  # Default neutral
    reason = "Unknown"
    
    # Extract error probability
    prob_match = re.search(r"ERROR_PROB:\s*([0-9.]+)", response, re.IGNORECASE)
    if prob_match:
        try:
            error_prob = float(prob_match.group(1))
            error_prob = max(0.0, min(1.0, error_prob))
        except ValueError:
            pass
    
    # Extract reason
    reason_match = re.search(r"REASON:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
    if reason_match:
        reason = reason_match.group(1).strip()
    
    return error_prob, reason


def run_fab_sc(
    question: str,
    ground_truth: float,
    model: LLMModel,
    cfg: DictConfig
) -> Dict:
    """Run FAB-SC method on a single question.
    
    Args:
        question: Question text
        ground_truth: Ground truth answer
        model: LLM model
        cfg: Configuration
        
    Returns:
        Result dictionary with predictions and metrics
    """
    method_cfg = cfg.run.method
    
    # Step 1: Forward solve sampling
    cot_prompt = create_cot_prompt(question)
    forward_samples = model.generate(
        cot_prompt,
        temperature=method_cfg.temperature,
        max_tokens=method_cfg.max_tokens,
        n=method_cfg.forward_samples
    )
    
    # Extract answers from forward samples
    candidates = []
    for idx, sample_text in enumerate(forward_samples):
        answer = extract_answer_from_text(sample_text)
        if answer is None:
            continue
            
        token_count = model.count_tokens(sample_text)
        
        # Initialize weights
        b_score = 1.0  # Default if backsolve disabled
        f_score = 1.0  # Default if falsification disabled
        
        # Step 2a: Backsolve check
        if method_cfg.backsolve_enabled:
            backsolve_prompt = create_backsolve_prompt(question, answer, sample_text)
            backsolve_response = model.generate(
                backsolve_prompt,
                temperature=0.3,  # Lower temperature for verification
                max_tokens=512,
                n=1
            )[0]
            b_score, verdict = parse_backsolve_response(backsolve_response)
        else:
            backsolve_response = None
            verdict = "N/A"
        
        # Step 2b: Falsification ensemble
        falsification_responses = []
        if method_cfg.falsification_enabled:
            error_probs = []
            for k in range(method_cfg.falsification_attempts):
                falsify_prompt = create_falsification_prompt(question, sample_text, k)
                falsify_response = model.generate(
                    falsify_prompt,
                    temperature=0.7,  # Higher temperature for diversity
                    max_tokens=256,
                    n=1
                )[0]
                falsification_responses.append(falsify_response)
                
                error_prob, reason = parse_falsification_response(falsify_response)
                error_probs.append(error_prob)
            
            # F-score = 1 - mean(error_prob)
            f_score = 1.0 - (sum(error_probs) / len(error_probs))
        
        # Step 3: Anti-verbosity regularization
        l_score = math.exp(-method_cfg.lambda_length * token_count / method_cfg.tokens0)
        
        # Step 4: Combined weight
        weight = (b_score ** method_cfg.alpha) * (f_score ** method_cfg.beta) * l_score
        
        candidates.append({
            "answer": answer,
            "rationale": sample_text,
            "token_count": token_count,
            "b_score": b_score,
            "f_score": f_score,
            "l_score": l_score,
            "weight": weight,
            "backsolve_verdict": verdict,
            "backsolve_response": backsolve_response,
            "falsification_responses": falsification_responses
        })
    
    # Step 5: Weighted aggregation
    if not candidates:
        return {
            "predicted": None,
            "ground_truth": ground_truth,
            "correct": False,
            "candidates": []
        }
    
    # Group by answer and sum weights
    answer_weights = {}
    for cand in candidates:
        ans = cand["answer"]
        answer_weights[ans] = answer_weights.get(ans, 0.0) + cand["weight"]
    
    # Select answer with maximum total weight
    predicted = max(answer_weights.items(), key=lambda x: x[1])[0]
    correct = abs(predicted - ground_truth) < 1e-6
    
    return {
        "predicted": predicted,
        "ground_truth": ground_truth,
        "correct": correct,
        "candidates": candidates,
        "answer_weights": answer_weights
    }


def run_sc(
    question: str,
    ground_truth: float,
    model: LLMModel,
    cfg: DictConfig
) -> Dict:
    """Run standard Self-Consistency method.
    
    Args:
        question: Question text
        ground_truth: Ground truth answer
        model: LLM model
        cfg: Configuration
        
    Returns:
        Result dictionary with predictions and metrics
    """
    method_cfg = cfg.run.method
    
    # Forward solve sampling
    cot_prompt = create_cot_prompt(question)
    forward_samples = model.generate(
        cot_prompt,
        temperature=method_cfg.temperature,
        max_tokens=method_cfg.max_tokens,
        n=method_cfg.forward_samples
    )
    
    # Extract answers
    answers = []
    candidates = []
    for sample_text in forward_samples:
        answer = extract_answer_from_text(sample_text)
        if answer is not None:
            answers.append(answer)
            candidates.append({
                "answer": answer,
                "rationale": sample_text
            })
    
    # Majority vote
    if not answers:
        return {
            "predicted": None,
            "ground_truth": ground_truth,
            "correct": False,
            "candidates": []
        }
    
    answer_counts = Counter(answers)
    predicted = answer_counts.most_common(1)[0][0]
    correct = abs(predicted - ground_truth) < 1e-6
    
    return {
        "predicted": predicted,
        "ground_truth": ground_truth,
        "correct": correct,
        "candidates": candidates,
        "answer_counts": dict(answer_counts)
    }


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Main inference script."""
    
    # Determine task type based on mode
    if cfg.mode == "sanity_check":
        # Override for sanity check mode
        cfg.run.dataset.max_samples = 5
        cfg.wandb.project = f"{cfg.wandb.project}-sanity"
    
    # Initialize wandb
    if cfg.wandb.mode != "disabled":
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=cfg.run.run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
            resume="allow",
            tags=cfg.run.wandb.get("tags", [])
        )
        print(f"WandB run URL: {wandb.run.get_url()}")
    
    # Load dataset
    print(f"Loading dataset: {cfg.run.dataset.name}")
    dataset = load_dataset_by_name(
        cfg.run.dataset.name,
        split=cfg.run.dataset.split,
        cache_dir=cfg.cache_dir,
        max_samples=cfg.run.dataset.max_samples
    )
    print(f"Loaded {len(dataset)} examples")
    
    # Initialize model
    print(f"Initializing model: {cfg.run.model.name}")
    model = LLMModel(
        provider=cfg.run.model.provider,
        model_name=cfg.run.model.api_model_name,
        max_context_length=cfg.run.model.max_context_length
    )
    
    # Run inference
    results = []
    correct_count = 0
    total_count = 0
    
    method_name = cfg.run.method.name
    print(f"Running {method_name} on {len(dataset)} questions...")
    
    for example in tqdm(dataset):
        question = example["question"]
        ground_truth = example["answer"]
        
        if ground_truth is None:
            continue
        
        # Run appropriate method
        if method_name == "fab-sc":
            result = run_fab_sc(question, ground_truth, model, cfg)
        elif method_name == "sc":
            result = run_sc(question, ground_truth, model, cfg)
        else:
            raise ValueError(f"Unknown method: {method_name}")
        
        result["question_id"] = example["id"]
        result["question"] = question
        results.append(result)
        
        if result["correct"]:
            correct_count += 1
        total_count += 1
        
        # Log to wandb
        if cfg.wandb.mode != "disabled":
            wandb.log({
                "accuracy": correct_count / total_count,
                "correct": int(result["correct"]),
                "question_idx": total_count - 1
            })
    
    # Compute final metrics
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    
    print(f"\nResults:")
    print(f"  Accuracy: {accuracy:.4f} ({correct_count}/{total_count})")
    
    # Save results
    results_dir = Path(cfg.results_dir) / cfg.run.run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = results_dir / "results.jsonl"
    with open(results_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    
    print(f"Saved results to {results_file}")
    
    # Save metrics summary
    metrics = {
        "accuracy": accuracy,
        "correct_count": correct_count,
        "total_count": total_count,
        "run_id": cfg.run.run_id,
        "method": method_name,
        "dataset": cfg.run.dataset.name
    }
    
    metrics_file = results_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Log to wandb summary
    if cfg.wandb.mode != "disabled":
        wandb.summary["accuracy"] = accuracy
        wandb.summary["correct_count"] = correct_count
        wandb.summary["total_count"] = total_count
        wandb.finish()
    
    # Sanity validation
    if cfg.mode == "sanity_check":
        # Check that we processed at least 5 samples
        samples_processed = total_count
        outputs_valid = all(r["predicted"] is not None for r in results)
        
        # Check for diversity in predictions
        all_predictions = [r["predicted"] for r in results if r["predicted"] is not None]
        outputs_unique = len(set(all_predictions)) > 1 if len(all_predictions) > 1 else True
        
        # All metrics finite
        metrics_finite = not math.isnan(accuracy) and not math.isinf(accuracy)
        
        # Summary
        summary = {
            "samples": samples_processed,
            "outputs_valid": outputs_valid,
            "outputs_unique": outputs_unique,
            "accuracy": accuracy
        }
        print(f"SANITY_VALIDATION_SUMMARY: {json.dumps(summary)}")
        
        # Verdict
        if samples_processed >= 5 and outputs_valid and metrics_finite:
            print("SANITY_VALIDATION: PASS")
        else:
            reasons = []
            if samples_processed < 5:
                reasons.append("insufficient_samples")
            if not outputs_valid:
                reasons.append("invalid_outputs")
            if not metrics_finite:
                reasons.append("invalid_metrics")
            print(f"SANITY_VALIDATION: FAIL reason={','.join(reasons)}")


if __name__ == "__main__":
    main()
