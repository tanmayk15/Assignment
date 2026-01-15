"""
Inference Engine
Handles model inference and post-processing
"""

import time
import torch
import torch.nn as nn
import cv2
import numpy as np
from typing import Dict, List, Tuple, Union
from PIL import Image


class ObjectDetector:
    """
    Inference engine for object detection
    """
    
    def __init__(
        self,
        model: nn.Module,
        classes: List[str],
        config: Dict,
        device: Union[str, torch.device] = 'cuda',
    ):
        self.model = model
        self.classes = classes
        self.config = config
        
        # Device
        if isinstance(device, str):
            self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model.to(self.device)
        self.model.eval()
        
        # Inference config
        inf_cfg = config.get('inference', {})
        self.confidence_threshold = inf_cfg.get('confidence_threshold', 0.5)
        self.nms_threshold = inf_cfg.get('nms_threshold', 0.5)
        
        # Image size
        self.image_size = tuple(config['dataset']['image_size'])
        
        # Normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)
    
    def predict(
        self,
        image: Union[str, np.ndarray, Image.Image, torch.Tensor],
        confidence_threshold: float = None,
    ) -> Dict:
        """
        Run inference on a single image
        
        Args:
            image: Image (path, numpy array, PIL Image, or tensor)
            confidence_threshold: Optional override for confidence threshold
            
        Returns:
            Dictionary with 'boxes', 'labels', 'scores', 'inference_time'
        """
        start_time = time.time()
        
        # Load and preprocess image
        image_tensor, original_image = self._preprocess(image)
        
        # Run inference
        with torch.no_grad():
            predictions = self.model.predict(
                image_tensor,
                confidence_threshold or self.confidence_threshold
            )
        
        inference_time = time.time() - start_time
        
        # Post-process predictions
        if len(predictions) > 0 and isinstance(predictions[0], dict):
            result = predictions[0]
            result['inference_time'] = inference_time
            result['fps'] = 1.0 / inference_time
        else:
            result = {
                'boxes': torch.tensor([]),
                'labels': torch.tensor([]),
                'scores': torch.tensor([]),
                'inference_time': inference_time,
                'fps': 1.0 / inference_time,
            }
        
        return result
    
    def predict_batch(
        self,
        images: List[Union[str, np.ndarray, Image.Image]],
        confidence_threshold: float = None,
    ) -> List[Dict]:
        """
        Run inference on a batch of images
        
        Args:
            images: List of images
            confidence_threshold: Optional override for confidence threshold
            
        Returns:
            List of prediction dictionaries
        """
        # Preprocess all images
        image_tensors = []
        for image in images:
            tensor, _ = self._preprocess(image)
            image_tensors.append(tensor)
        
        # Stack into batch
        batch = torch.cat(image_tensors, dim=0)
        
        # Run inference
        start_time = time.time()
        with torch.no_grad():
            predictions = self.model.predict(
                batch,
                confidence_threshold or self.confidence_threshold
            )
        inference_time = time.time() - start_time
        
        # Add timing info
        for pred in predictions:
            if isinstance(pred, dict):
                pred['inference_time'] = inference_time / len(images)
                pred['fps'] = len(images) / inference_time
        
        return predictions
    
    def _preprocess(
        self,
        image: Union[str, np.ndarray, Image.Image, torch.Tensor]
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Preprocess image for inference
        
        Returns:
            Preprocessed tensor and original image
        """
        # Load image
        if isinstance(image, str):
            original_image = cv2.imread(image)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            original_image = np.array(image)
        elif isinstance(image, np.ndarray):
            original_image = image.copy()
        elif isinstance(image, torch.Tensor):
            return image.unsqueeze(0).to(self.device), None
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Resize
        image_resized = cv2.resize(original_image, 
                                   (self.image_size[1], self.image_size[0]))
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float()
        image_tensor = image_tensor / 255.0
        
        # Normalize
        image_tensor = image_tensor.to(self.device)
        image_tensor = (image_tensor - self.mean) / self.std
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor, original_image
    
    def visualize(
        self,
        image: Union[str, np.ndarray, Image.Image],
        predictions: Dict,
        save_path: str = None,
        show: bool = False,
    ) -> np.ndarray:
        """
        Visualize predictions on image
        
        Args:
            image: Input image
            predictions: Prediction dictionary
            save_path: Optional path to save visualization
            show: Whether to display the image
            
        Returns:
            Annotated image as numpy array
        """
        # Load image
        if isinstance(image, str):
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            img = np.array(image)
        elif isinstance(image, np.ndarray):
            img = image.copy()
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Get predictions
        boxes = predictions['boxes'].cpu().numpy() if isinstance(predictions['boxes'], torch.Tensor) else predictions['boxes']
        labels = predictions['labels'].cpu().numpy() if isinstance(predictions['labels'], torch.Tensor) else predictions['labels']
        scores = predictions['scores'].cpu().numpy() if isinstance(predictions['scores'], torch.Tensor) else predictions['scores']
        
        # Scale boxes if image was resized
        orig_h, orig_w = img.shape[:2]
        scale_x = orig_w / self.image_size[1]
        scale_y = orig_h / self.image_size[0]
        
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
        
        # Colors for each class
        colors = self._generate_colors(len(self.classes))
        
        # Draw boxes
        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box.astype(int)
            
            # Adjust label index (account for background class)
            class_idx = label - 1 if label > 0 else 0
            class_name = self.classes[class_idx] if class_idx < len(self.classes) else 'unknown'
            color = colors[class_idx % len(colors)]
            
            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label_text = f'{class_name}: {score:.2f}'
            (label_w, label_h), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            )
            
            cv2.rectangle(img, (x1, y1 - label_h - 10), 
                         (x1 + label_w, y1), color, -1)
            cv2.putText(img, label_text, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Add FPS info
        if 'fps' in predictions:
            fps_text = f"FPS: {predictions['fps']:.1f}"
            cv2.putText(img, fps_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # Save or show
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print(f"Saved visualization to {save_path}")
        
        if show:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 8))
            plt.imshow(img)
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        
        return img
    
    def _generate_colors(self, n: int) -> List[Tuple[int, int, int]]:
        """Generate distinct colors for visualization"""
        colors = []
        for i in range(n):
            hue = i / n
            rgb = self._hsv_to_rgb(hue, 0.8, 0.9)
            colors.append(tuple(int(c * 255) for c in rgb))
        return colors
    
    def _hsv_to_rgb(self, h: float, s: float, v: float) -> Tuple[float, float, float]:
        """Convert HSV to RGB"""
        import colorsys
        return colorsys.hsv_to_rgb(h, s, v)


if __name__ == "__main__":
    print("Inference module - use via inference.py script")
