"""
Quick start script to set up and test the quality inspection system.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and print the result."""
    print(f"\n{'='*70}")
    print(f"Step: {description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✓ {description} completed successfully!")
        if result.stdout:
            print(result.stdout)
    else:
        print(f"✗ Error in {description}")
        if result.stderr:
            print(result.stderr)
        return False
    
    return True


def main():
    """Run the complete demonstration workflow."""
    
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║   AUTOMATED QUALITY INSPECTION SYSTEM - QUICK START               ║
    ║   Defect Detection & Classification for Manufacturing            ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Create directories
    Path('sample_images').mkdir(exist_ok=True)
    Path('results').mkdir(exist_ok=True)
    
    steps = [
        {
            'cmd': [sys.executable, 'generate_samples.py', '--output', 'sample_images', '--num-samples', '8'],
            'desc': 'Generate synthetic PCB sample images'
        },
        {
            'cmd': [sys.executable, 'demo.py', '--mode', 'batch', '--dir', 'sample_images', '--output', 'results'],
            'desc': 'Run batch inspection on all samples'
        }
    ]
    
    for i, step in enumerate(steps, 1):
        print(f"\n\n[STEP {i}/{len(steps)}]")
        success = run_command(step['cmd'], step['desc'])
        if not success:
            print(f"\n✗ Workflow stopped due to error in step {i}")
            return
    
    print(f"""
    
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                    🎉 DEMONSTRATION COMPLETE! 🎉                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    
    📁 Check the following directories:
    
    1. sample_images/
       - Generated PCB images with various defects
       - Perfect (defect-free) reference image
       
    2. results/
       - Annotated images showing detected defects
       - JSON reports for each image
       - Summary report with statistics
       - Visualization charts
    
    🔍 To inspect a specific image:
       python defect_inspector.py --image sample_images/pcb_defect_01.jpg --output results
    
    📊 To view summary:
       - Open results/summary_report.json
       - Open results/summary_visualization.png
    
    🎥 Record a screen demo:
       - Run the inspection on various images
       - Show the annotated output
       - Display the JSON reports
    
    ✅ Next steps:
       1. Review the generated images in sample_images/
       2. Check annotated results in results/
       3. Read the summary report
       4. Test with your own images!
    
    """)


if __name__ == '__main__':
    main()
