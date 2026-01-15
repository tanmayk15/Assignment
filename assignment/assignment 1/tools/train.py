"""
Training Script
Main script to train the object detection model
"""

import os
import argparse
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from models import build_model
from data import VOCDetectionDataset, collate_fn
from engine import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train Faster R-CNN from scratch')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Path to output directory')
    parser.add_argument('--num_epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line args
    if args.num_epochs is not None:
        config['training']['num_epochs'] = args.num_epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    
    # Update data directory
    config['dataset']['data_dir'] = args.data_dir
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = VOCDetectionDataset(
        data_dir=args.data_dir,
        split='train',
        classes=config['dataset']['classes'],
        config=config,
    )
    
    val_dataset = VOCDetectionDataset(
        data_dir=args.data_dir,
        split='val',
        classes=config['dataset']['classes'],
        config=config,
    )
    
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")
    
    # Print class distribution
    print("\nTrain set class distribution:")
    dist = train_dataset.get_class_distribution()
    for cls, count in dist.items():
        print(f"  {cls}: {count}")
    
    # Create data loaders
    train_cfg = config['training']
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        num_workers=train_cfg.get('num_workers', 4),
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=False,
        num_workers=train_cfg.get('num_workers', 4),
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    # Build model
    print("\nBuilding model...")
    model = build_model(config)
    model.to(device)
    
    # Print model info
    params = model.count_parameters()
    size = model.get_model_size()
    print(f"Model: {config['model']['name']}")
    print(f"Backbone: {config['model']['backbone']}")
    print(f"Total parameters: {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")
    print(f"Model size: {size:.2f} MB")
    
    # Create optimizer
    opt_cfg = train_cfg['optimizer']
    if opt_cfg['type'] == 'SGD':
        optimizer = optim.SGD(
            model.parameters(),
            lr=opt_cfg['lr'],
            momentum=opt_cfg['momentum'],
            weight_decay=opt_cfg['weight_decay'],
        )
    elif opt_cfg['type'] == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=opt_cfg['lr'],
            weight_decay=opt_cfg['weight_decay'],
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_cfg['type']}")
    
    # Create learning rate scheduler
    lr_cfg = train_cfg['lr_schedule']
    if lr_cfg['type'] == 'multi_step':
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=lr_cfg['milestones'],
            gamma=lr_cfg['gamma'],
        )
    elif lr_cfg['type'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=train_cfg['num_epochs'],
        )
    else:
        raise ValueError(f"Unknown scheduler: {lr_cfg['type']}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
        output_dir=args.output_dir,
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Start training
    trainer.train()


if __name__ == '__main__':
    main()
