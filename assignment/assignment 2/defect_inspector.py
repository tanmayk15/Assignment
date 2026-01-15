"""
Automated Quality Inspection System for Manufacturing
Detects and classifies defects in PCB (Printed Circuit Board) images
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple
import json
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime


class DefectInspector:
    """
    A class for detecting and classifying defects in manufactured items (PCBs).
    Supports multiple defect types: scratches, missing components, misalignment.
    """
    
    def __init__(self, sensitivity: float = 0.5):
        """
        Initialize the defect inspector.
        
        Args:
            sensitivity: Detection sensitivity (0-1), higher = more sensitive
        """
        self.sensitivity = sensitivity
        self.defect_types = {
            'scratch': {'color': (0, 0, 255), 'severity_threshold': 100},
            'missing_component': {'color': (255, 0, 0), 'severity_threshold': 500},
            'misalignment': {'color': (0, 255, 255), 'severity_threshold': 200},
            'discoloration': {'color': (255, 165, 0), 'severity_threshold': 150}
        }
        
    def analyze_image(self, image_path: str) -> Dict:
        """
        Analyze an input image for defects.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing defect information
        """
        print(f"\n{'='*60}")
        print(f"Analyzing image: {Path(image_path).name}")
        print(f"{'='*60}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        original = image.copy()
        
        # Detect defects
        defects = self.detect_defects(image)
        
        # Generate report
        report = {
            'image_path': image_path,
            'timestamp': datetime.now().isoformat(),
            'total_defects': len(defects),
            'defects': defects,
            'quality_status': 'PASS' if len(defects) == 0 else 'FAIL'
        }
        
        # Print summary
        self._print_report(report)
        
        return report
    
    def detect_defects(self, image: np.ndarray) -> List[Dict]:
        """
        Detect all types of defects in the image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detected defects with their properties
        """
        defects = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect scratches (linear defects)
        scratch_defects = self._detect_scratches(image, gray)
        defects.extend(scratch_defects)
        
        # Detect missing components (holes/voids)
        missing_defects = self._detect_missing_components(image, gray)
        defects.extend(missing_defects)
        
        # Detect misalignment (shape irregularities)
        misalign_defects = self._detect_misalignment(image, gray)
        defects.extend(misalign_defects)
        
        # Detect discoloration
        discolor_defects = self._detect_discoloration(image)
        defects.extend(discolor_defects)
        
        return defects
    
    def _detect_scratches(self, image: np.ndarray, gray: np.ndarray) -> List[Dict]:
        """Detect scratch defects using edge detection and line detection."""
        defects = []
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                               threshold=int(50 * self.sensitivity),
                               minLineLength=20, maxLineGap=10)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                # Filter for scratch-like lines
                if length > 30:
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    severity = self._calculate_severity(length, 'scratch')
                    confidence = min(0.95, length / 200 + 0.5)
                    
                    defects.append({
                        'type': 'scratch',
                        'center': (center_x, center_y),
                        'bbox': (min(x1, x2), min(y1, y2), 
                                abs(x2-x1), abs(y2-y1)),
                        'severity': severity,
                        'confidence': round(confidence, 3),
                        'area': length,
                        'description': f'Linear scratch detected, length: {int(length)}px'
                    })
        
        return defects
    
    def _detect_missing_components(self, image: np.ndarray, gray: np.ndarray) -> List[Dict]:
        """Detect missing component defects using blob detection."""
        defects = []
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours (potential missing components)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter based on area
            if 100 < area < 5000:
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    severity = self._calculate_severity(area, 'missing_component')
                    confidence = min(0.92, area / 2000 + 0.6)
                    
                    defects.append({
                        'type': 'missing_component',
                        'center': (cx, cy),
                        'bbox': (x, y, w, h),
                        'severity': severity,
                        'confidence': round(confidence, 3),
                        'area': area,
                        'description': f'Missing/damaged component, area: {int(area)}px²'
                    })
        
        return defects
    
    def _detect_misalignment(self, image: np.ndarray, gray: np.ndarray) -> List[Dict]:
        """Detect misalignment defects using shape analysis."""
        defects = []
        
        # Apply adaptive threshold
        adaptive_thresh = cv2.adaptiveThreshold(gray, 255, 
                                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area > 200:
                # Check shape irregularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    # Irregular shapes might indicate misalignment
                    if circularity < 0.4:
                        M = cv2.moments(contour)
                        if M['m00'] != 0:
                            cx = int(M['m10'] / M['m00'])
                            cy = int(M['m01'] / M['m00'])
                            
                            x, y, w, h = cv2.boundingRect(contour)
                            aspect_ratio = float(w) / h if h > 0 else 0
                            
                            # Check for extreme aspect ratios (misalignment indicator)
                            if aspect_ratio > 3 or aspect_ratio < 0.3:
                                severity = self._calculate_severity(area, 'misalignment')
                                confidence = round(1 - circularity, 3)
                                
                                defects.append({
                                    'type': 'misalignment',
                                    'center': (cx, cy),
                                    'bbox': (x, y, w, h),
                                    'severity': severity,
                                    'confidence': confidence,
                                    'area': area,
                                    'description': f'Component misalignment, circularity: {circularity:.2f}'
                                })
        
        return defects
    
    def _detect_discoloration(self, image: np.ndarray) -> List[Dict]:
        """Detect discoloration defects using color analysis."""
        defects = []
        
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for abnormal discoloration
        # Looking for unusual hues or saturation
        lower_bound = np.array([0, 50, 50])
        upper_bound = np.array([180, 255, 255])
        
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if 150 < area < 3000:
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    severity = self._calculate_severity(area, 'discoloration')
                    confidence = min(0.88, area / 1500 + 0.5)
                    
                    defects.append({
                        'type': 'discoloration',
                        'center': (cx, cy),
                        'bbox': (x, y, w, h),
                        'severity': severity,
                        'confidence': round(confidence, 3),
                        'area': area,
                        'description': f'Surface discoloration, area: {int(area)}px²'
                    })
        
        return defects
    
    def _calculate_severity(self, metric: float, defect_type: str) -> str:
        """
        Calculate severity level based on defect size/metric.
        
        Args:
            metric: Size metric (length, area, etc.)
            defect_type: Type of defect
            
        Returns:
            Severity level: 'LOW', 'MEDIUM', or 'HIGH'
        """
        threshold = self.defect_types[defect_type]['severity_threshold']
        
        if metric < threshold * 0.5:
            return 'LOW'
        elif metric < threshold * 1.5:
            return 'MEDIUM'
        else:
            return 'HIGH'
    
    def _print_report(self, report: Dict):
        """Print formatted inspection report."""
        print(f"\nInspection Report:")
        print(f"Status: {report['quality_status']}")
        print(f"Total defects found: {report['total_defects']}")
        
        if report['defects']:
            print(f"\nDefect Details:")
            print(f"{'-'*60}")
            
            for i, defect in enumerate(report['defects'], 1):
                print(f"\nDefect #{i}:")
                print(f"  Type: {defect['type']}")
                print(f"  Center coordinates (x, y): {defect['center']}")
                print(f"  Bounding box (x, y, w, h): {defect['bbox']}")
                print(f"  Severity: {defect['severity']}")
                print(f"  Confidence: {defect['confidence']:.1%}")
                print(f"  Description: {defect['description']}")
        
        print(f"\n{'='*60}\n")
    
    def visualize_defects(self, image_path: str, report: Dict, output_path: str = None):
        """
        Visualize defects on the image with bounding boxes and labels.
        
        Args:
            image_path: Path to input image
            report: Inspection report from analyze_image()
            output_path: Path to save annotated image (optional)
        """
        image = cv2.imread(image_path)
        annotated = image.copy()
        
        # Draw defects
        for defect in report['defects']:
            x, y, w, h = defect['bbox']
            center = defect['center']
            defect_type = defect['type']
            color = self.defect_types[defect_type]['color']
            
            # Draw bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            
            # Draw center point
            cv2.circle(annotated, center, 5, color, -1)
            
            # Add label
            label = f"{defect_type}: {defect['severity']}"
            cv2.putText(annotated, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Add center coordinates
            coord_text = f"({center[0]}, {center[1]})"
            cv2.putText(annotated, coord_text, (center[0] + 10, center[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Add status banner
        status_color = (0, 255, 0) if report['quality_status'] == 'PASS' else (0, 0, 255)
        cv2.rectangle(annotated, (0, 0), (300, 40), status_color, -1)
        cv2.putText(annotated, f"Status: {report['quality_status']}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save if output path provided
        if output_path:
            cv2.imwrite(output_path, annotated)
            print(f"Annotated image saved to: {output_path}")
        
        # Display
        plt.figure(figsize=(15, 10))
        
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        plt.title(f'Detected Defects: {len(report["defects"])}')
        plt.axis('off')
        
        plt.tight_layout()
        
        if output_path:
            viz_path = str(Path(output_path).with_suffix('')) + '_comparison.png'
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            print(f"Comparison saved to: {viz_path}")
        
        plt.show()
    
    def save_report(self, report: Dict, output_path: str):
        """
        Save inspection report to JSON file.
        
        Args:
            report: Inspection report
            output_path: Path to save JSON file
        """
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to: {output_path}")


def main():
    """Main function to demonstrate the defect inspection system."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Automated Quality Inspection System')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--output', type=str, default='output',
                       help='Output directory for results')
    parser.add_argument('--sensitivity', type=float, default=0.5,
                       help='Detection sensitivity (0-1)')
    parser.add_argument('--visualize', action='store_true',
                       help='Display visualization of results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize inspector
    inspector = DefectInspector(sensitivity=args.sensitivity)
    
    # Analyze image
    report = inspector.analyze_image(args.image)
    
    # Generate output paths
    image_name = Path(args.image).stem
    annotated_path = output_dir / f"{image_name}_annotated.jpg"
    report_path = output_dir / f"{image_name}_report.json"
    
    # Visualize and save results
    inspector.visualize_defects(args.image, report, str(annotated_path))
    inspector.save_report(report, str(report_path))
    
    print(f"\n✓ Inspection complete!")
    print(f"✓ Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
