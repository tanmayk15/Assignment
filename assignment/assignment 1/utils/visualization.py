"""
Visualization utilities
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Dict
import torch


class Colors:
    """Color palette for visualization"""
    
    def __init__(self):
        self.palette = [
            (255, 0, 0),      # Red
            (0, 255, 0),      # Green
            (0, 0, 255),      # Blue
            (255, 255, 0),    # Yellow
            (255, 0, 255),    # Magenta
            (0, 255, 255),    # Cyan
            (128, 0, 0),      # Maroon
            (0, 128, 0),      # Dark Green
            (0, 0, 128),      # Navy
            (128, 128, 0),    # Olive
        ]
    
    def __call__(self, idx: int) -> Tuple[int, int, int]:
        return self.palette[idx % len(self.palette)]


def draw_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    class_names: List[str],
    thickness: int = 2,
    font_scale: float = 0.5,
) -> np.ndarray:
    """
    Draw bounding boxes on image
    
    Args:
        image: [H, W, 3] RGB image
        boxes: [N, 4] boxes in (x1, y1, x2, y2) format
        labels: [N] class indices
        scores: [N] confidence scores
        class_names: List of class names
        thickness: Box line thickness
        font_scale: Font scale for labels
        
    Returns:
        Annotated image
    """
    image = image.copy()
    colors = Colors()
    
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.astype(int)
        
        # Get color
        color = colors(label)
        
        # Draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # Prepare label text
        class_name = class_names[label] if label < len(class_names) else f'class_{label}'
        label_text = f'{class_name}: {score:.2f}'
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            1
        )
        
        # Draw label background
        cv2.rectangle(
            image,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            image,
            label_text,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1
        )
    
    return image


def plot_training_curves(
    log_file: str,
    output_path: str = None,
    metrics: List[str] = ['loss', 'mAP'],
):
    """
    Plot training curves from log file
    
    Args:
        log_file: Path to training log
        output_path: Where to save plot
        metrics: List of metrics to plot
    """
    # This is a placeholder - actual implementation would parse log file
    print(f"Plotting training curves from {log_file}")
    print(f"Metrics: {metrics}")
    
    # Create dummy plot
    fig, axes = plt.subplots(1, len(metrics), figsize=(6*len(metrics), 5))
    
    if len(metrics) == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics):
        ax.set_title(f'{metric} vs Epoch')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric)
        ax.grid(True)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


def create_confusion_matrix_plot(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    output_path: str = None,
):
    """
    Plot confusion matrix
    
    Args:
        confusion_matrix: [N, N] confusion matrix
        class_names: List of class names
        output_path: Where to save plot
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    im = ax.imshow(confusion_matrix, cmap='Blues')
    
    # Set ticks
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add colorbar
    plt.colorbar(im, ax=ax)
    
    # Add values
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = ax.text(j, i, f'{confusion_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black")
    
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved confusion matrix to {output_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    # Test visualization
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    boxes = np.array([[100, 100, 200, 200], [300, 300, 450, 450]])
    labels = np.array([0, 1])
    scores = np.array([0.95, 0.87])
    class_names = ['person', 'vehicle']
    
    annotated = draw_boxes(image, boxes, labels, scores, class_names)
    
    print(f"Original image shape: {image.shape}")
    print(f"Annotated image shape: {annotated.shape}")
