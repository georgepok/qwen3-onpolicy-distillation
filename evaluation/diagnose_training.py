#!/usr/bin/env python3
"""
Diagnostic script to investigate training degradation.

This script helps identify why the trained model performs worse than base:
1. Verifies LoRA is actually loading
2. Checks training metrics (if available)
3. Evaluates teacher model quality
4. Provides actionable recommendations
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qwen3_distill.generation.generator import StudentGenerator, GenerationConfig
from qwen3_distill.scoring.scorer import TeacherScorer, ScoringConfig


def test_lora_loading():
    """Test if LoRA is actually being loaded."""
    print("\n" + "="*60)
    print("TEST 1: LoRA Loading Verification")
    print("="*60)
    
    config = GenerationConfig(
        model_name="Qwen/Qwen3-4B-Instruct",
        api_url="http://localhost:8000/v1",
        temperature=0.0,  # Deterministic
    )
    
    generator = StudentGenerator(config)
    
    # Test prompt
    test_prompt = "What is 2+2?"
    
    # Generate without LoRA
    print("\n[1/2] Generating WITHOUT LoRA...")
    trajectories_base = generator.generate_trajectories([test_prompt])
    response_base = trajectories_base[0]["generated_text"]
    print(f"Base response: {response_base[:100]}")
    
    # Load LoRA
    lora_path = "./checkpoints/epoch_9_lora"
    if not Path(lora_path).exists():
        print(f"\n❌ LoRA checkpoint not found: {lora_path}")
        print("Make sure training completed and saved checkpoints")
        return False
    
    print(f"\n[2/2] Generating WITH LoRA from {lora_path}...")
    generator.reload_with_lora(lora_path)
    trajectories_lora = generator.generate_trajectories([test_prompt])
    response_lora = trajectories_lora[0]["generated_text"]
    print(f"LoRA response: {response_lora[:100]}")
    
    # Compare
    print("\n" + "-"*60)
    print("COMPARISON:")
    print("-"*60)
    print(f"Base:  {response_base[:200]}")
    print(f"LoRA:  {response_lora[:200]}")
    
    if response_base == response_lora:
        print("\n❌ RESPONSES IDENTICAL - LoRA NOT LOADING!")
        print("\nPossible causes:")
        print("  1. vLLM server not started with --enable-lora flag")
        print("  2. LoRA adapter path not accessible to container")
        print("  3. LoRA adapter corrupted or invalid")
        print("\nFix:")
        print("  Add to docker-compose.yml vllm-student service:")
        print("    command: --enable-lora --max-lora-rank 16")
        return False
    else:
        print("\n✅ RESPONSES DIFFERENT - LoRA IS LOADING!")
        similarity = sum(a == b for a, b in zip(response_base, response_lora)) / max(len(response_base), len(response_lora))
        print(f"   Similarity: {similarity*100:.1f}% (should be < 90%)")
        return True


def check_training_metrics():
    """Check if training metrics show improvement."""
    print("\n" + "="*60)
    print("TEST 2: Training Metrics Analysis")
    print("="*60)
    
    metrics_files = [
        "results/training_metrics.json",
        "logs/training.log",
        "checkpoints/metrics.txt",
    ]
    
    found = False
    for metrics_file in metrics_files:
        if Path(metrics_file).exists():
            print(f"\n✅ Found metrics file: {metrics_file}")
            found = True
            
            # Try to parse and show key metrics
            try:
                if metrics_file.endswith(".json"):
                    import json
                    with open(metrics_file, "r") as f:
                        data = json.load(f)
                    print(f"\nMetrics summary:")
                    print(json.dumps(data, indent=2)[:500])
                else:
                    with open(metrics_file, "r") as f:
                        lines = f.readlines()
                    print(f"\nLast 20 lines:")
                    print("".join(lines[-20:]))
            except Exception as e:
                print(f"   Could not parse: {e}")
    
    if not found:
        print("\n❌ No training metrics found!")
        print("\nThis is concerning - we can't verify if training improved.")
        print("\nExpected files:")
        for f in metrics_files:
            print(f"  - {f}")
        print("\nWithout metrics, we can't tell if:")
        print("  1. Training loss/KL decreased")
        print("  2. Model converged")
        print("  3. Training completed successfully")
        return False
    
    return True


def evaluate_teacher():
    """Evaluate teacher model on sample problems."""
    print("\n" + "="*60)
    print("TEST 3: Teacher Model Quality")
    print("="*60)
    
    # Simple math problems
    test_problems = [
        {"question": "If John has 5 apples and buys 3 more, how many does he have?", "answer": "8"},
        {"question": "What is 7 + 8?", "answer": "15"},
        {"question": "If a book costs $12 and you buy 3 books, how much do you spend?", "answer": "36"},
        {"question": "What is 100 - 37?", "answer": "63"},
        {"question": "If there are 4 boxes with 6 items each, how many items total?", "answer": "24"},
    ]
    
    print(f"\nEvaluating teacher on {len(test_problems)} sample problems...")
    
    config = ScoringConfig(
        model_name="Qwen/Qwen3-32B-Instruct",
        api_url="http://localhost:30000/v1",
    )
    
    try:
        scorer = TeacherScorer(config)
        
        correct = 0
        for i, problem in enumerate(test_problems):
            print(f"\n[{i+1}/{len(test_problems)}] {problem['question']}")
            
            # Generate teacher response
            response = scorer.client.completions.create(
                model=config.model_name,
                prompt=problem['question'],
                max_tokens=100,
                temperature=0.0,
            )
            
            teacher_text = response.choices[0].text
            print(f"  Teacher: {teacher_text[:100]}")
            
            # Extract number from response
            import re
            numbers = re.findall(r'\b\d+\b', teacher_text)
            if numbers:
                predicted = numbers[-1]  # Take last number
                is_correct = predicted == problem["answer"]
                correct += is_correct
                print(f"  Answer: {predicted} {'✅' if is_correct else '❌'} (expected {problem['answer']})")
            else:
                print(f"  Answer: Could not extract ❌")
        
        accuracy = correct / len(test_problems) * 100
        print(f"\n" + "-"*60)
        print(f"Teacher Accuracy: {accuracy:.1f}% ({correct}/{len(test_problems)})")
        print("-"*60)
        
        if accuracy < 70:
            print("\n❌ TEACHER TOO WEAK!")
            print(f"   {accuracy:.1f}% accuracy is insufficient for distillation")
            print("\nRecommendations:")
            print("  1. Use stronger teacher (Qwen3-72B or GPT-4)")
            print("  2. Switch to supervised fine-tuning on GSM8K dataset")
            print("  3. Use outcome-based distillation (only learn from correct examples)")
            return False
        elif accuracy < 85:
            print("\n⚠️  TEACHER MODERATE")
            print(f"   {accuracy:.1f}% accuracy may limit student improvement")
            print("   Consider using stronger teacher for better results")
            return True
        else:
            print("\n✅ TEACHER QUALITY GOOD")
            print(f"   {accuracy:.1f}% accuracy should enable effective distillation")
            return True
            
    except Exception as e:
        print(f"\n❌ Could not evaluate teacher: {e}")
        print("\nMake sure teacher API is running:")
        print("  docker-compose up vllm-teacher")
        return False


def main():
    """Run all diagnostic tests."""
    print("="*60)
    print("TRAINING DEGRADATION DIAGNOSTIC")
    print("="*60)
    print("\nThis script will help identify why trained model performs worse.")
    print("Running 3 diagnostic tests...\n")
    
    results = {}
    
    # Test 1: LoRA loading
    try:
        results["lora_loading"] = test_lora_loading()
    except Exception as e:
        print(f"\n❌ LoRA test failed: {e}")
        results["lora_loading"] = False
    
    # Test 2: Training metrics
    try:
        results["has_metrics"] = check_training_metrics()
    except Exception as e:
        print(f"\n❌ Metrics test failed: {e}")
        results["has_metrics"] = False
    
    # Test 3: Teacher quality
    try:
        results["teacher_quality"] = evaluate_teacher()
    except Exception as e:
        print(f"\n❌ Teacher test failed: {e}")
        results["teacher_quality"] = False
    
    # Summary
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)
    
    all_passed = all(results.values())
    
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test.replace('_', ' ').title()}")
    
    print("\n" + "="*60)
    
    if all_passed:
        print("✅ ALL TESTS PASSED")
        print("\nYour infrastructure seems OK, but model still degraded.")
        print("This suggests the training objective is the problem.")
        print("\nRecommendation:")
        print("  Switch from pure distillation to outcome-based training")
        print("  See TRAINING_DEGRADATION_ANALYSIS.md for details")
    else:
        print("❌ ISSUES DETECTED")
        failed = [k for k, v in results.items() if not v]
        print(f"\nFailed tests: {', '.join(failed)}")
        print("\nFix these issues before continuing training!")
        print("See TRAINING_DEGRADATION_ANALYSIS.md for solutions")
    
    print("="*60)


if __name__ == "__main__":
    main()
