"""LLM model interface for OpenAI and Anthropic APIs."""

import os
import time
from typing import List, Dict, Optional
import tiktoken


class LLMModel:
    """Base class for LLM models."""
    
    def __init__(self, provider: str, model_name: str, max_context_length: int = 128000):
        """Initialize LLM model.
        
        Args:
            provider: Provider name (openai or anthropic)
            model_name: Model identifier
            max_context_length: Maximum context length
        """
        self.provider = provider
        self.model_name = model_name
        self.max_context_length = max_context_length
        
        if provider == "openai":
            from openai import OpenAI
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            self.client = OpenAI(api_key=api_key)
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        elif provider == "anthropic":
            from anthropic import Anthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            self.client = Anthropic(api_key=api_key)
            self.tokenizer = tiktoken.get_encoding("cl100k_base")  # Approximation
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        n: int = 1,
        stop: Optional[List[str]] = None
    ) -> List[str]:
        """Generate completions from prompt.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            n: Number of completions
            stop: Stop sequences
            
        Returns:
            List of generated completions
        """
        if self.provider == "openai":
            return self._generate_openai(prompt, temperature, max_tokens, n, stop)
        elif self.provider == "anthropic":
            return self._generate_anthropic(prompt, temperature, max_tokens, n, stop)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def _generate_openai(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        n: int,
        stop: Optional[List[str]]
    ) -> List[str]:
        """Generate using OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                n=n,
                stop=stop
            )
            return [choice.message.content for choice in response.choices]
        except Exception as e:
            print(f"OpenAI API error: {e}")
            time.sleep(1)  # Brief backoff
            raise
    
    def _generate_anthropic(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        n: int,
        stop: Optional[List[str]]
    ) -> List[str]:
        """Generate using Anthropic API.
        
        Note: Anthropic doesn't support n>1 directly, so we make n sequential calls.
        """
        results = []
        for _ in range(n):
            try:
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                    stop_sequences=stop
                )
                results.append(response.content[0].text)
            except Exception as e:
                print(f"Anthropic API error: {e}")
                time.sleep(1)
                raise
        return results


def create_cot_prompt(question: str) -> str:
    """Create Chain-of-Thought prompt for arithmetic reasoning.
    
    Args:
        question: Math word problem
        
    Returns:
        Formatted prompt with CoT instruction
    """
    prompt = f"""Solve this math word problem step by step. Show your reasoning clearly.

Question: {question}

Let's solve this step by step:"""
    return prompt


def create_backsolve_prompt(question: str, answer: float, rationale: str) -> str:
    """Create backsolve/plug-back verification prompt.
    
    Args:
        question: Original question
        answer: Proposed answer
        rationale: Rationale that led to the answer
        
    Returns:
        Verification prompt
    """
    prompt = f"""Given a proposed answer to a math problem, verify it by checking if it satisfies all constraints.

Question: {question}

Proposed answer: {answer}

Proposed rationale:
{rationale}

Task: Verify this answer by:
1. Identifying all constraints and quantities from the problem
2. Checking if the proposed answer satisfies each constraint
3. Recomputing derived quantities to ensure consistency

Provide your verification in this format:
VERIFICATION:
[Your step-by-step verification]

CONFIDENCE: [A number between 0 and 1, where 1 means completely confident the answer is correct]

VERDICT: [PASS or FAIL]"""
    return prompt


def create_falsification_prompt(question: str, rationale: str, refuter_idx: int) -> str:
    """Create adversarial falsification prompt.
    
    Args:
        question: Original question
        rationale: Rationale to try to refute
        refuter_idx: Index of refuter (for diversity)
        
    Returns:
        Falsification prompt
    """
    focus_aspects = [
        "Check for unit conversion errors and quantity misreads.",
        "Look for algebraic mistakes or arithmetic errors.",
        "Verify that all problem constraints were considered."
    ]
    
    focus = focus_aspects[refuter_idx % len(focus_aspects)]
    
    prompt = f"""You are a critical reviewer. Try to find flaws in this solution to a math problem.

Question: {question}

Proposed solution:
{rationale}

Task: {focus}
Look for errors, invalid assumptions, or logical gaps.

Provide your analysis in this format:
ANALYSIS:
[Your critical analysis]

ERROR_PROB: [A number between 0 and 1 representing the probability this solution is wrong. 0 = definitely correct, 1 = definitely wrong]

REASON: [One sentence explaining the main issue, or "No significant errors found"]"""
    return prompt
