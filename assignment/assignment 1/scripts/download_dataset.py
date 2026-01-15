"""
Download and prepare PASCAL VOC dataset
"""

import os
import argparse
import tarfile
import urllib.request
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Download file with progress bar"""
    with DownloadProgressBar(unit='B', unit_scale=True,
                            miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def parse_args():
    parser = argparse.ArgumentParser(description='Download dataset')
    parser.add_argument('--output_dir', type=str, default='dataset/',
                       help='Output directory for dataset')
    parser.add_argument('--dataset', type=str, default='voc2012',
                       choices=['voc2007', 'voc2012', 'coco'],
                       help='Dataset to download')
    
    return parser.parse_args()


def download_voc2012(output_dir):
    """Download PASCAL VOC 2012 dataset"""
    print("Downloading PASCAL VOC 2012 dataset...")
    
    url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Download
    tar_path = os.path.join(output_dir, "VOCtrainval_11-May-2012.tar")
    
    if not os.path.exists(tar_path):
        print(f"Downloading from {url}...")
        download_url(url, tar_path)
    else:
        print(f"Archive already exists: {tar_path}")
    
    # Extract
    print("Extracting...")
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(output_dir)
    
    print(f"Dataset downloaded and extracted to {output_dir}")
    print("\nNext steps:")
    print("1. Organize the dataset into train/val/test splits")
    print("2. Run: python scripts/prepare_data.py")


def main():
    args = parse_args()
    
    if args.dataset == 'voc2012':
        download_voc2012(args.output_dir)
    elif args.dataset == 'voc2007':
        print("VOC2007 download not implemented yet")
    elif args.dataset == 'coco':
        print("COCO download not implemented yet")
        print("Please manually download COCO from: https://cocodataset.org/")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
