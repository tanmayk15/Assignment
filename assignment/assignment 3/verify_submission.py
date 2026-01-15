"""
Verification script - Check if all submission components are present
"""

import os
from pathlib import Path
from typing import List, Tuple


def check_file_exists(filepath: str) -> bool:
    """Check if file exists"""
    return Path(filepath).exists()


def check_directory_exists(dirpath: str) -> bool:
    """Check if directory exists"""
    return Path(dirpath).is_dir()


def verify_submission() -> Tuple[List[str], List[str]]:
    """Verify all submission components"""
    
    print("=" * 70)
    print(" SUBMISSION VERIFICATION")
    print("=" * 70)
    
    passed = []
    failed = []
    
    # Documentation files
    print("\n[1/6] Checking documentation files...")
    docs = [
        "README.md",
        "SOLUTION.md",
        "QUICKSTART.md",
        "INDEX.md",
        "GITHUB_SETUP.md",
        "SUBMISSION.md"
    ]
    
    for doc in docs:
        if check_file_exists(doc):
            print(f"  ✓ {doc}")
            passed.append(doc)
        else:
            print(f"  ✗ {doc} - MISSING")
            failed.append(doc)
    
    # Implementation files
    print("\n[2/6] Checking implementation files...")
    implementations = [
        "src/model_selection/vlm_comparison.py",
        "src/architecture/custom_vlm.py",
        "src/optimization/inference_optimizer.py",
        "src/training/qa_generator.py",
        "src/validation/metrics.py"
    ]
    
    for impl in implementations:
        if check_file_exists(impl):
            print(f"  ✓ {impl}")
            passed.append(impl)
        else:
            print(f"  ✗ {impl} - MISSING")
            failed.append(impl)
    
    # Executable files
    print("\n[3/6] Checking executable files...")
    executables = [
        "demo.py",
        "run_all.py",
        "setup.sh",
        "setup.ps1"
    ]
    
    for exe in executables:
        if check_file_exists(exe):
            print(f"  ✓ {exe}")
            passed.append(exe)
        else:
            print(f"  ✗ {exe} - MISSING")
            failed.append(exe)
    
    # Configuration files
    print("\n[4/6] Checking configuration files...")
    configs = [
        "requirements.txt",
        ".gitignore"
    ]
    
    for cfg in configs:
        if check_file_exists(cfg):
            print(f"  ✓ {cfg}")
            passed.append(cfg)
        else:
            print(f"  ✗ {cfg} - MISSING")
            failed.append(cfg)
    
    # Directories
    print("\n[5/6] Checking directories...")
    directories = [
        "src",
        "src/model_selection",
        "src/architecture",
        "src/optimization",
        "src/training",
        "src/validation",
        "recordings"
    ]
    
    for directory in directories:
        if check_directory_exists(directory):
            print(f"  ✓ {directory}/")
            passed.append(directory)
        else:
            print(f"  ✗ {directory}/ - MISSING")
            failed.append(directory)
    
    # Recording instructions
    print("\n[6/6] Checking recording setup...")
    if check_file_exists("recordings/README.md"):
        print("  ✓ recordings/README.md")
        passed.append("recordings/README.md")
    else:
        print("  ✗ recordings/README.md - MISSING")
        failed.append("recordings/README.md")
    
    # Check for actual recordings (warning if missing)
    recording_files = list(Path("recordings").glob("*.mp4")) if Path("recordings").exists() else []
    if recording_files:
        print(f"  ✓ Found {len(recording_files)} recording file(s)")
        for rec in recording_files:
            print(f"    - {rec.name}")
    else:
        print("  ⚠ No recording files found (remember to add before submission)")
    
    return passed, failed


def check_file_sizes():
    """Check file sizes to ensure content is present"""
    print("\n" + "=" * 70)
    print(" FILE SIZE CHECK")
    print("=" * 70)
    
    important_files = [
        "SOLUTION.md",
        "README.md",
        "src/architecture/custom_vlm.py",
        "src/optimization/inference_optimizer.py",
        "src/validation/metrics.py"
    ]
    
    for filepath in important_files:
        if Path(filepath).exists():
            size = Path(filepath).stat().st_size
            size_kb = size / 1024
            
            # Check if file has reasonable content
            if size_kb < 1:
                print(f"  ⚠ {filepath}: {size_kb:.1f}KB (might be empty)")
            elif size_kb < 10:
                print(f"  ⚠ {filepath}: {size_kb:.1f}KB (might be incomplete)")
            else:
                print(f"  ✓ {filepath}: {size_kb:.1f}KB")
        else:
            print(f"  ✗ {filepath}: NOT FOUND")


def print_summary(passed: List[str], failed: List[str]):
    """Print verification summary"""
    print("\n" + "=" * 70)
    print(" VERIFICATION SUMMARY")
    print("=" * 70)
    
    total = len(passed) + len(failed)
    pass_rate = (len(passed) / total * 100) if total > 0 else 0
    
    print(f"\nTotal items checked: {total}")
    print(f"Passed: {len(passed)} ({pass_rate:.1f}%)")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print("\nMissing items:")
        for item in failed:
            print(f"  ✗ {item}")
    
    print("\n" + "=" * 70)
    
    if not failed:
        print("✓✓✓ ALL SUBMISSION COMPONENTS PRESENT ✓✓✓")
        print("\nNext steps:")
        print("  1. Add screen recordings to recordings/ directory")
        print("  2. Set up GitHub repository (see GITHUB_SETUP.md)")
        print("  3. Push all files to GitHub")
        print("  4. Verify repository is public")
        print("  5. Submit GitHub link")
    else:
        print("✗✗✗ SOME COMPONENTS MISSING ✗✗✗")
        print("\nPlease ensure all files are present before submission.")
    
    print("=" * 70)


def verify_python_syntax():
    """Verify Python files have valid syntax"""
    print("\n" + "=" * 70)
    print(" PYTHON SYNTAX CHECK")
    print("=" * 70)
    
    import py_compile
    
    python_files = [
        "demo.py",
        "run_all.py",
        "src/model_selection/vlm_comparison.py",
        "src/architecture/custom_vlm.py",
        "src/optimization/inference_optimizer.py",
        "src/training/qa_generator.py",
        "src/validation/metrics.py"
    ]
    
    all_valid = True
    
    for py_file in python_files:
        if Path(py_file).exists():
            try:
                py_compile.compile(py_file, doraise=True)
                print(f"  ✓ {py_file} - Valid syntax")
            except py_compile.PyCompileError as e:
                print(f"  ✗ {py_file} - Syntax error: {e}")
                all_valid = False
        else:
            print(f"  ⚠ {py_file} - File not found")
    
    if all_valid:
        print("\n✓ All Python files have valid syntax")
    else:
        print("\n✗ Some Python files have syntax errors")


def main():
    """Main verification function"""
    print("Starting submission verification...\n")
    
    # Check components
    passed, failed = verify_submission()
    
    # Check file sizes
    check_file_sizes()
    
    # Check Python syntax
    verify_python_syntax()
    
    # Print summary
    print_summary(passed, failed)
    
    return 0 if not failed else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
