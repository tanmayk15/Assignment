"""
Script to generate synthetic defective PCB images for testing
the quality inspection system.
"""

import cv2
import numpy as np
from pathlib import Path
import random


class DefectGenerator:
    """Generate synthetic defects on PCB images for testing."""
    
    @staticmethod
    def create_base_pcb(width=800, height=600):
        """Create a base PCB-like image."""
        # Create green PCB base
        pcb = np.zeros((height, width, 3), dtype=np.uint8)
        pcb[:, :] = (34, 139, 34)  # Green color
        
        # Add some texture
        noise = np.random.randint(-20, 20, (height, width, 3), dtype=np.int16)
        pcb = np.clip(pcb.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Add circuit traces
        for _ in range(15):
            x1, y1 = random.randint(0, width), random.randint(0, height)
            x2, y2 = random.randint(0, width), random.randint(0, height)
            cv2.line(pcb, (x1, y1), (x2, y2), (200, 200, 200), 2)
        
        # Add solder pads (circles)
        for _ in range(30):
            x, y = random.randint(50, width-50), random.randint(50, height-50)
            radius = random.randint(5, 15)
            cv2.circle(pcb, (x, y), radius, (192, 192, 192), -1)
        
        # Add components (rectangles)
        for _ in range(10):
            x, y = random.randint(50, width-100), random.randint(50, height-80)
            w, h = random.randint(30, 80), random.randint(20, 50)
            cv2.rectangle(pcb, (x, y), (x+w, y+h), (50, 50, 50), -1)
            # Add pins
            for i in range(3):
                cv2.circle(pcb, (x + i*15 + 10, y + h//2), 3, (192, 192, 192), -1)
        
        return pcb
    
    @staticmethod
    def add_scratch(image, num_scratches=3):
        """Add scratch defects to the image."""
        img = image.copy()
        for _ in range(num_scratches):
            x1, y1 = random.randint(0, img.shape[1]), random.randint(0, img.shape[0])
            length = random.randint(50, 200)
            angle = random.uniform(0, 2*np.pi)
            x2 = int(x1 + length * np.cos(angle))
            y2 = int(y1 + length * np.sin(angle))
            
            # Draw scratch
            cv2.line(img, (x1, y1), (x2, y2), (100, 100, 100), 
                    random.randint(1, 3))
        return img
    
    @staticmethod
    def add_missing_component(image, num_missing=2):
        """Add missing component defects."""
        img = image.copy()
        for _ in range(num_missing):
            x = random.randint(100, img.shape[1]-100)
            y = random.randint(100, img.shape[0]-100)
            size = random.randint(20, 60)
            
            # Create irregular hole/void
            cv2.circle(img, (x, y), size, (0, 0, 0), -1)
            # Add darker border to emphasize
            cv2.circle(img, (x, y), size, (20, 20, 20), 2)
        return img
    
    @staticmethod
    def add_misalignment(image, num_misalign=2):
        """Add misalignment defects."""
        img = image.copy()
        for _ in range(num_misalign):
            x = random.randint(100, img.shape[1]-150)
            y = random.randint(100, img.shape[0]-100)
            
            # Draw misaligned component (rotated rectangle)
            w, h = random.randint(40, 80), random.randint(15, 30)
            angle = random.uniform(-30, 30)
            
            # Create rotated rectangle
            rect = ((x, y), (w, h), angle)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img, [box], 0, (60, 60, 60), -1)
        return img
    
    @staticmethod
    def add_discoloration(image, num_spots=3):
        """Add discoloration defects."""
        img = image.copy()
        for _ in range(num_spots):
            x = random.randint(50, img.shape[1]-50)
            y = random.randint(50, img.shape[0]-50)
            radius = random.randint(20, 50)
            
            # Create discolored spot with gradient
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            cv2.circle(mask, (x, y), radius, 255, -1)
            mask = cv2.GaussianBlur(mask, (51, 51), 0)
            
            # Apply brownish discoloration
            color = np.array([20, 80, 120])  # BGR brownish
            for c in range(3):
                img[:, :, c] = np.where(mask > 0,
                                       np.clip(img[:, :, c].astype(np.int16) + 
                                              (color[c] * mask / 255).astype(np.int16), 0, 255),
                                       img[:, :, c])
        return img.astype(np.uint8)
    
    @staticmethod
    def generate_dataset(output_dir: Path, num_samples: int = 10):
        """Generate a complete dataset with various defects."""
        output_dir.mkdir(exist_ok=True, parents=True)
        
        defect_functions = {
            'scratch': DefectGenerator.add_scratch,
            'missing': DefectGenerator.add_missing_component,
            'misalign': DefectGenerator.add_misalignment,
            'discolor': DefectGenerator.add_discoloration
        }
        
        # Generate perfect sample
        print("Generating perfect (defect-free) PCB...")
        perfect_pcb = DefectGenerator.create_base_pcb()
        cv2.imwrite(str(output_dir / "pcb_perfect.jpg"), perfect_pcb)
        
        # Generate defective samples
        for i in range(num_samples):
            print(f"Generating defective sample {i+1}/{num_samples}...")
            
            base = DefectGenerator.create_base_pcb()
            
            # Randomly select defect types
            num_defect_types = random.randint(1, 3)
            selected_defects = random.sample(list(defect_functions.keys()), 
                                           num_defect_types)
            
            # Apply defects
            for defect_type in selected_defects:
                base = defect_functions[defect_type](base)
            
            # Save with descriptive filename
            defect_names = '_'.join(selected_defects)
            filename = f"pcb_defect_{i+1:02d}_{defect_names}.jpg"
            cv2.imwrite(str(output_dir / filename), base)
        
        # Generate samples with single defect types for testing
        print("\nGenerating single-defect samples for validation...")
        for defect_type, func in defect_functions.items():
            base = DefectGenerator.create_base_pcb()
            defective = func(base)
            cv2.imwrite(str(output_dir / f"pcb_{defect_type}_only.jpg"), defective)
        
        print(f"\n✓ Dataset generated successfully in: {output_dir}")
        print(f"✓ Total images: {num_samples + 5} (1 perfect + {num_samples} mixed + 4 single-type)")


def main():
    """Generate sample PCB images."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate synthetic PCB defect dataset')
    parser.add_argument('--output', type=str, default='sample_images',
                       help='Output directory for generated images')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of defective samples to generate')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    DefectGenerator.generate_dataset(output_dir, args.num_samples)


if __name__ == '__main__':
    main()
