"""
Demo script to test the defect inspection system on sample images.
Runs batch processing and generates comprehensive results.
"""

import cv2
import numpy as np
from pathlib import Path
from defect_inspector import DefectInspector
import json
import matplotlib.pyplot as plt
from typing import List, Dict


def batch_process(image_dir: str, output_dir: str = 'results'):
    """
    Process all images in a directory and generate reports.
    
    Args:
        image_dir: Directory containing input images
        output_dir: Directory to save results
    """
    image_path = Path(image_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Get all image files
    image_files = list(image_path.glob('*.jpg')) + \
                  list(image_path.glob('*.png')) + \
                  list(image_path.glob('*.jpeg'))
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return
    
    print(f"\n{'='*70}")
    print(f"BATCH PROCESSING: {len(image_files)} images")
    print(f"{'='*70}\n")
    
    # Initialize inspector
    inspector = DefectInspector(sensitivity=0.5)
    
    all_reports = []
    
    # Process each image
    for i, img_file in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {img_file.name}")
        print("-" * 70)
        
        try:
            # Analyze image
            report = inspector.analyze_image(str(img_file))
            all_reports.append(report)
            
            # Save annotated image
            annotated_path = output_path / f"{img_file.stem}_annotated.jpg"
            inspector.visualize_defects(str(img_file), report, str(annotated_path))
            
            # Save individual report
            report_path = output_path / f"{img_file.stem}_report.json"
            inspector.save_report(report, str(report_path))
            
        except Exception as e:
            print(f"Error processing {img_file.name}: {e}")
    
    # Generate summary report
    generate_summary_report(all_reports, output_path)
    
    print(f"\n{'='*70}")
    print(f"✓ Batch processing complete!")
    print(f"✓ Results saved to: {output_path}")
    print(f"{'='*70}\n")


def generate_summary_report(reports: List[Dict], output_dir: Path):
    """Generate a summary report for all processed images."""
    
    summary = {
        'total_images': len(reports),
        'passed': sum(1 for r in reports if r['quality_status'] == 'PASS'),
        'failed': sum(1 for r in reports if r['quality_status'] == 'FAIL'),
        'total_defects': sum(len(r['defects']) for r in reports),
        'defect_breakdown': {}
    }
    
    # Count defects by type
    defect_counts = {}
    severity_counts = {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0}
    
    for report in reports:
        for defect in report['defects']:
            dtype = defect['type']
            defect_counts[dtype] = defect_counts.get(dtype, 0) + 1
            severity_counts[defect['severity']] += 1
    
    summary['defect_breakdown'] = defect_counts
    summary['severity_breakdown'] = severity_counts
    summary['pass_rate'] = f"{(summary['passed'] / summary['total_images'] * 100):.1f}%"
    
    # Save summary
    summary_path = output_dir / 'summary_report.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY REPORT")
    print(f"{'='*70}")
    print(f"Total images processed: {summary['total_images']}")
    print(f"Passed: {summary['passed']} | Failed: {summary['failed']}")
    print(f"Pass rate: {summary['pass_rate']}")
    print(f"Total defects found: {summary['total_defects']}")
    
    if defect_counts:
        print(f"\nDefect type breakdown:")
        for dtype, count in defect_counts.items():
            print(f"  - {dtype}: {count}")
        
        print(f"\nSeverity breakdown:")
        for severity, count in severity_counts.items():
            print(f"  - {severity}: {count}")
    
    print(f"\n✓ Summary saved to: {summary_path}")
    
    # Create visualization
    create_summary_visualization(summary, output_dir)


def create_summary_visualization(summary: Dict, output_dir: Path):
    """Create visualization charts for the summary."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Quality Inspection Summary', fontsize=16, fontweight='bold')
    
    # 1. Pass/Fail pie chart
    ax1 = axes[0, 0]
    labels = ['Pass', 'Fail']
    sizes = [summary['passed'], summary['failed']]
    colors = ['#2ecc71', '#e74c3c']
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax1.set_title('Pass/Fail Distribution')
    
    # 2. Defect type bar chart
    ax2 = axes[0, 1]
    if summary['defect_breakdown']:
        defect_types = list(summary['defect_breakdown'].keys())
        defect_counts = list(summary['defect_breakdown'].values())
        bars = ax2.bar(defect_types, defect_counts, color='#3498db')
        ax2.set_xlabel('Defect Type')
        ax2.set_ylabel('Count')
        ax2.set_title('Defect Type Distribution')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
    
    # 3. Severity bar chart
    ax3 = axes[1, 0]
    severities = list(summary['severity_breakdown'].keys())
    severity_counts = list(summary['severity_breakdown'].values())
    colors_sev = ['#f39c12', '#e67e22', '#c0392b']
    bars = ax3.bar(severities, severity_counts, color=colors_sev)
    ax3.set_xlabel('Severity Level')
    ax3.set_ylabel('Count')
    ax3.set_title('Severity Distribution')
    
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    # 4. Statistics text
    ax4 = axes[1, 1]
    ax4.axis('off')
    stats_text = f"""
    INSPECTION STATISTICS
    
    Total Images: {summary['total_images']}
    Passed: {summary['passed']}
    Failed: {summary['failed']}
    Pass Rate: {summary['pass_rate']}
    
    Total Defects: {summary['total_defects']}
    Avg Defects/Image: {summary['total_defects']/summary['total_images']:.2f}
    """
    ax4.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure
    viz_path = output_dir / 'summary_visualization.png'
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"✓ Summary visualization saved to: {viz_path}")
    plt.close()


def test_single_image(image_path: str, output_dir: str = 'results'):
    """Test the inspector on a single image."""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Initialize inspector
    inspector = DefectInspector(sensitivity=0.5)
    
    # Analyze
    report = inspector.analyze_image(image_path)
    
    # Visualize
    img_name = Path(image_path).stem
    annotated_path = output_path / f"{img_name}_annotated.jpg"
    inspector.visualize_defects(image_path, report, str(annotated_path))
    
    # Save report
    report_path = output_path / f"{img_name}_report.json"
    inspector.save_report(report, str(report_path))


def main():
    """Main demo function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Demo: Automated Quality Inspection System'
    )
    parser.add_argument('--mode', type=str, choices=['single', 'batch'], 
                       default='batch',
                       help='Processing mode: single image or batch')
    parser.add_argument('--image', type=str,
                       help='Path to single image (for single mode)')
    parser.add_argument('--dir', type=str, default='sample_images',
                       help='Directory containing images (for batch mode)')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        if not args.image:
            print("Error: --image required for single mode")
            return
        test_single_image(args.image, args.output)
    else:
        batch_process(args.dir, args.output)


if __name__ == '__main__':
    main()
