#!/usr/bin/env python3
"""
GSM8K Math Reasoning Benchmark

Simple benchmark to test math reasoning capabilities.
Uses a small subset of GSM8K for quick evaluation.
"""

import sys
import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from qwen3_distill.generation.generator import StudentGenerator, GenerationConfig


# Sample GSM8K problems (easy to verify)
GSM8K_SAMPLES = [
    {
        "question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        "answer": 18,  # 16 - 3 - 4 = 9 eggs, 9 * 2 = $18
    },
    {
        "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
        "answer": 3,  # 2 blue + 1 white = 3
    },
    {
        "question": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
        "answer": 70000,  # Value increased: 80000 * 1.5 = 120000, New value: 80000 + 120000 = 200000, Profit: 200000 - 80000 - 50000 = 70000
    },
    {
        "question": "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?",
        "answer": 540,  # 3 sprints * 3 times * 60 meters = 540
    },
    {
        "question": "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy. She gives the chickens their feed in three separate meals. If she has 20 chickens, how many cups of feed does she need for the final meal of the day?",
        "answer": 20,  # 3 cups per chicken / 3 meals = 1 cup per meal per chicken, 20 chickens * 1 cup = 20 cups
    },
    {
        "question": "Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glasses. How much does he need to pay for them?",
        "answer": 64,  # 8 full price ($5) + 8 discounted ($3) = 40 + 24 = $64
    },
    {
        "question": "A basketball team has 24 players. Half of them are boys and the rest are girls. How many girls are on the team?",
        "answer": 12,  # 24 / 2 = 12
    },
    {
        "question": "Tom has 10 apples. He gives 3 to his friend and eats 2. How many apples does he have left?",
        "answer": 5,  # 10 - 3 - 2 = 5
    },
    {
        "question": "A store sells pencils for $0.25 each. If you buy 12 pencils, how much do you pay?",
        "answer": 3.0,  # 12 * 0.25 = $3.00
    },
    {
        "question": "Sarah reads 15 pages of a book every day. How many pages will she read in a week?",
        "answer": 105,  # 15 * 7 = 105
    },
]


def extract_number(text: str) -> Optional[float]:
    """
    Extract the final numerical answer from generated text.

    Args:
        text: Generated response text

    Returns:
        Extracted number or None
    """
    # Try to find patterns like "The answer is X", "$X", "X dollars", etc.
    patterns = [
        r'(?:answer is|equals?|=)\s*\$?(\d+\.?\d*)',
        r'\$(\d+\.?\d*)',
        r'(\d+\.?\d*)\s*dollars?',
        r'(?:total|result).*?(\d+\.?\d*)',
    ]

    text_lower = text.lower()

    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue

    # Fallback: extract last number in text
    numbers = re.findall(r'\d+\.?\d*', text)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            pass

    return None


def evaluate_on_gsm8k(
    generator: StudentGenerator,
    problems: List[Dict[str, Any]],
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Evaluate model on GSM8K problems.

    Args:
        generator: Model generator
        problems: List of GSM8K problems
        model_name: Name for logging

    Returns:
        Evaluation results
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name} on GSM8K")
    print(f"{'='*60}\n")

    prompts = [
        f"Solve this math problem step by step and give the final answer:\n\n{p['question']}"
        for p in problems
    ]

    # Generate responses
    print(f"Generating responses for {len(prompts)} problems...")
    trajectories = generator.generate_trajectories(
        prompts,
        temperature=0.0,  # Greedy for math
        max_tokens=512
    )

    # Extract answers and compare
    results = []
    correct = 0

    for i, (problem, traj) in enumerate(zip(problems, trajectories)):
        generated_text = traj['generated_text']
        predicted_answer = extract_number(generated_text)
        true_answer = problem['answer']

        is_correct = False
        if predicted_answer is not None:
            # Allow small floating point tolerance
            is_correct = abs(predicted_answer - true_answer) < 0.01
            if is_correct:
                correct += 1

        results.append({
            "problem_id": i,
            "question": problem['question'],
            "true_answer": true_answer,
            "predicted_answer": predicted_answer,
            "is_correct": is_correct,
            "response": generated_text,
        })

        # Print each result
        status = "✅" if is_correct else "❌"
        print(f"{status} Problem {i+1}: Predicted={predicted_answer}, True={true_answer}")

    accuracy = correct / len(problems) if problems else 0

    print(f"\n{'='*60}")
    print(f"Results: {correct}/{len(problems)} correct ({accuracy:.1%})")
    print(f"{'='*60}\n")

    return {
        "model_name": model_name,
        "num_problems": len(problems),
        "correct": correct,
        "accuracy": accuracy,
        "results": results,
    }


def main():
    """Run GSM8K benchmark comparing base model vs LoRA epoch 9."""
    print("="*60)
    print("GSM8K BENCHMARK: BASE vs EPOCH 9 LoRA")
    print("="*60)

    # Clear environment variable to avoid conflicts
    if 'VLLM_API_URL' in os.environ:
        del os.environ['VLLM_API_URL']
    if 'TEACHER_API_URL' in os.environ:
        del os.environ['TEACHER_API_URL']

    # Initialize student generator (will test both base and LoRA)
    print("\nInitializing Student Generator...")
    student_config = GenerationConfig(
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        api_url="http://vllm-student:8000",
        samples_per_prompt=1,
        max_tokens=512,
    )
    student_gen = StudentGenerator(student_config)

    # Evaluate base model
    print("\n" + "="*60)
    print("EVALUATING BASE MODEL (NO LORA)")
    print("="*60)
    base_results = evaluate_on_gsm8k(
        student_gen,
        GSM8K_SAMPLES,
        "Base Model (Qwen3-4B)"
    )

    # Load LoRA checkpoint from epoch 9
    print("\n" + "="*60)
    print("LOADING EPOCH 9 LoRA CHECKPOINT")
    print("="*60)
    lora_path = "./checkpoints/epoch_9_lora"
    print(f"Loading LoRA adapter from: {lora_path}")
    student_gen.reload_with_lora(lora_path)
    print("LoRA loaded successfully!\n")

    # Evaluate trained model with LoRA
    print("\n" + "="*60)
    print("EVALUATING TRAINED MODEL (EPOCH 9 LoRA)")
    print("="*60)
    lora_results = evaluate_on_gsm8k(
        student_gen,
        GSM8K_SAMPLES,
        "Trained Model (Epoch 9 LoRA)"
    )

    # Compare
    print(f"\n{'='*60}")
    print("TRAINING IMPACT ANALYSIS")
    print(f"{'='*60}\n")

    base_acc = base_results['accuracy']
    lora_acc = lora_results['accuracy']
    improvement = lora_acc - base_acc
    relative_improvement = (improvement / base_acc * 100) if base_acc > 0 else 0

    print(f"Base Model Accuracy:    {base_acc:.1%} ({base_results['correct']}/{base_results['num_problems']})")
    print(f"Trained Model Accuracy: {lora_acc:.1%} ({lora_results['correct']}/{lora_results['num_problems']})")
    print(f"\nAbsolute Improvement: {improvement:+.1%}")
    print(f"Relative Improvement: {relative_improvement:+.1f}%")

    if improvement > 0.10:
        print("\n✅ EXCELLENT - Training significantly improved reasoning (+10% absolute)")
    elif improvement > 0.05:
        print("\n✅ GOOD - Training improved reasoning (+5% absolute)")
    elif improvement > 0.00:
        print("\n⚠️  MODEST - Training showed small improvement")
    elif improvement == 0:
        print("\n⚠️  NO CHANGE - Training had no measurable impact")
    else:
        print("\n❌ REGRESSION - Training degraded reasoning capability")

    # Compare problem-by-problem
    print(f"\n{'='*60}")
    print("PROBLEM-BY-PROBLEM COMPARISON")
    print(f"{'='*60}\n")

    improved = 0
    degraded = 0
    unchanged_correct = 0
    unchanged_incorrect = 0

    for i, (base_r, lora_r) in enumerate(zip(base_results['results'], lora_results['results'])):
        base_correct = base_r['is_correct']
        lora_correct = lora_r['is_correct']

        if not base_correct and lora_correct:
            improved += 1
            print(f"✅ Problem {i+1}: FIXED by training (was wrong, now correct)")
        elif base_correct and not lora_correct:
            degraded += 1
            print(f"❌ Problem {i+1}: DEGRADED by training (was correct, now wrong)")
        elif base_correct and lora_correct:
            unchanged_correct += 1
        else:
            unchanged_incorrect += 1

    print(f"\nSummary:")
    print(f"  Problems fixed by training: {improved}")
    print(f"  Problems degraded by training: {degraded}")
    print(f"  Remained correct: {unchanged_correct}")
    print(f"  Remained incorrect: {unchanged_incorrect}")

    # Save results
    output_dir = Path("evaluation/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison = {
        "base_model": {k: v for k, v in base_results.items() if k != 'results'},
        "trained_model": {k: v for k, v in lora_results.items() if k != 'results'},
        "training_impact": {
            "absolute_improvement": improvement,
            "relative_improvement_pct": relative_improvement,
            "problems_fixed": improved,
            "problems_degraded": degraded,
            "remained_correct": unchanged_correct,
            "remained_incorrect": unchanged_incorrect,
        },
        "detailed_results": {
            "base_model": base_results['results'],
            "trained_model": lora_results['results'],
        }
    }

    output_file = output_dir / "gsm8k_base_vs_lora.json"
    with open(output_file, "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"\nResults saved to: {output_file}\n")


if __name__ == "__main__":
    main()
