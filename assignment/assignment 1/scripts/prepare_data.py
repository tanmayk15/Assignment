"""
Prepare and organize dataset
Convert to required format and create train/val/test splits
"""

import os
import argparse
import shutil
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare dataset')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input directory with raw data')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for processed data')
    parser.add_argument('--train_split', type=float, default=0.7,
                       help='Training set ratio')
    parser.add_argument('--val_split', type=float, default=0.15,
                       help='Validation set ratio')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    return parser.parse_args()


def organize_voc_dataset(input_dir, output_dir, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    Organize PASCAL VOC format dataset into train/val/test splits
    """
    random.seed(seed)
    
    print(f"Organizing dataset from {input_dir} to {output_dir}...")
    
    # Expected structure:
    # input_dir/
    #   JPEGImages/
    #   Annotations/
    
    images_dir = os.path.join(input_dir, 'JPEGImages')
    annotations_dir = os.path.join(input_dir, 'Annotations')
    
    if not os.path.exists(images_dir) or not os.path.exists(annotations_dir):
        print(f"Error: Expected JPEGImages/ and Annotations/ in {input_dir}")
        return
    
    # Get all image files
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(image_files)} images")
    
    # Shuffle
    random.shuffle(image_files)
    
    # Split
    n = len(image_files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train+n_val]
    test_files = image_files[n_train+n_val:]
    
    print(f"Split: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
    
    # Create output structure
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'annotations', split), exist_ok=True)
    
    # Copy files
    def copy_split(files, split):
        print(f"Copying {split} split...")
        for img_file in files:
            # Image
            src_img = os.path.join(images_dir, img_file)
            dst_img = os.path.join(output_dir, 'images', split, img_file)
            shutil.copy2(src_img, dst_img)
            
            # Annotation
            xml_file = img_file.rsplit('.', 1)[0] + '.xml'
            src_xml = os.path.join(annotations_dir, xml_file)
            
            if os.path.exists(src_xml):
                dst_xml = os.path.join(output_dir, 'annotations', split, xml_file)
                shutil.copy2(src_xml, dst_xml)
    
    copy_split(train_files, 'train')
    copy_split(val_files, 'val')
    copy_split(test_files, 'test')
    
    print(f"\n✓ Dataset organized successfully!")
    print(f"Output directory: {output_dir}")
    print("\nStructure:")
    print(f"  {output_dir}/")
    print(f"    images/")
    print(f"      train/  ({len(train_files)} images)")
    print(f"      val/    ({len(val_files)} images)")
    print(f"      test/   ({len(test_files)} images)")
    print(f"    annotations/")
    print(f"      train/  ({len(train_files)} annotations)")
    print(f"      val/    ({len(val_files)} annotations)")
    print(f"      test/   ({len(test_files)} annotations)")


def main():
    args = parse_args()
    
    # Calculate test split
    test_ratio = 1.0 - args.train_split - args.val_split
    
    if test_ratio < 0:
        print("Error: train_split + val_split must be <= 1.0")
        return
    
    print("Dataset preparation")
    print("=" * 60)
    print(f"Train split: {args.train_split:.2%}")
    print(f"Val split: {args.val_split:.2%}")
    print(f"Test split: {test_ratio:.2%}")
    print(f"Random seed: {args.seed}")
    print("=" * 60)
    print()
    
    organize_voc_dataset(
        args.input_dir,
        args.output_dir,
        args.train_split,
        args.val_split,
        args.seed
    )


if __name__ == '__main__':
    main()
