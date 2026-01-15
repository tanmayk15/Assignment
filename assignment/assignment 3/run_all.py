"""
Test runner - Run all demonstrations
"""

import subprocess
import sys
from pathlib import Path


def run_script(script_path: str, description: str):
    """Run a Python script and report results"""
    print("\n" + "=" * 70)
    print(f" {description}")
    print("=" * 70 + "\n")
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=False,
            text=True,
            check=True
        )
        print(f"\n✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed with error code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n✗ {description} failed: {e}")
        return False


def main():
    """Run all demonstration scripts"""
    print("=" * 70)
    print(" RUNNING ALL DEMONSTRATIONS")
    print("=" * 70)
    
    scripts = [
        ("src/model_selection/vlm_comparison.py", "Model Selection Analysis (Part A)"),
        ("src/architecture/custom_vlm.py", "Architecture Implementation (Part B)"),
        ("src/optimization/inference_optimizer.py", "Optimization Pipeline (Part C)"),
        ("src/training/qa_generator.py", "QA Pair Generation (Part E)"),
        ("src/validation/metrics.py", "Validation Framework (Part F)"),
        ("demo.py", "End-to-End Demo"),
    ]
    
    results = []
    
    for script, description in scripts:
        success = run_script(script, description)
        results.append((description, success))
    
    # Print summary
    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for description, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status} - {description}")
    
    print("\n" + "=" * 70)
    print(f" {passed}/{total} demonstrations completed successfully")
    print("=" * 70)
    
    if passed == total:
        print("\n✓✓✓ ALL DEMONSTRATIONS PASSED ✓✓✓")
        return 0
    else:
        print("\n✗✗✗ SOME DEMONSTRATIONS FAILED ✗✗✗")
        return 1


if __name__ == "__main__":
    sys.exit(main())
