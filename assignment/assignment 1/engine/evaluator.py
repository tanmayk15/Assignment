"""
Evaluation Engine
Computes mAP and other detection metrics
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
from tqdm import tqdm


class Evaluator:
    """
    Evaluator for object detection
    Computes mAP (mean Average Precision) and other metrics
    """
    
    def __init__(
        self,
        classes: List[str],
        iou_thresholds: List[float] = [0.5, 0.75],
        device: torch.device = None,
    ):
        self.classes = classes
        self.num_classes = len(classes)
        self.iou_thresholds = iou_thresholds
        self.device = device or torch.device('cpu')
    
    def evaluate(
        self,
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        confidence_threshold: float = 0.05,
    ) -> Dict:
        """
        Evaluate model on dataset
        
        Args:
            model: Detection model
            data_loader: DataLoader for evaluation data
            confidence_threshold: Minimum confidence score
            
        Returns:
            Dictionary with evaluation metrics
        """
        model.eval()
        
        # Collect all predictions and ground truths
        all_predictions = []
        all_ground_truths = []
        
        print("Running inference...")
        with torch.no_grad():
            for images, targets in tqdm(data_loader):
                images = images.to(self.device)
                
                # Get predictions
                predictions = model.predict(images, confidence_threshold)
                
                # Store predictions and ground truths
                for pred, target in zip(predictions, targets):
                    all_predictions.append(pred)
                    all_ground_truths.append(target)
        
        # Compute metrics
        print("\nComputing mAP...")
        results = {}
        
        for iou_thresh in self.iou_thresholds:
            ap_per_class = self._compute_ap(
                all_predictions,
                all_ground_truths,
                iou_thresh
            )
            
            mean_ap = np.mean(list(ap_per_class.values()))
            results[f'mAP@{iou_thresh}'] = mean_ap
            results[f'AP@{iou_thresh}'] = ap_per_class
        
        # Print results
        self._print_results(results)
        
        return results
    
    def _compute_ap(
        self,
        predictions: List[Dict],
        ground_truths: List[Dict],
        iou_threshold: float,
    ) -> Dict[str, float]:
        """
        Compute Average Precision for each class
        
        Args:
            predictions: List of prediction dicts
            ground_truths: List of ground truth dicts
            iou_threshold: IoU threshold for positive detection
            
        Returns:
            Dictionary mapping class name to AP
        """
        ap_per_class = {}
        
        for class_idx, class_name in enumerate(self.classes):
            # Collect all predictions and ground truths for this class
            class_preds = []
            class_gts = []
            
            for img_idx, (pred, gt) in enumerate(zip(predictions, ground_truths)):
                # Predictions for this class (labels are 1-indexed with 0 as background)
                if isinstance(pred, dict) and 'labels' in pred:
                    mask = pred['labels'] == (class_idx + 1)  # +1 for background
                    
                    if mask.sum() > 0:
                        boxes = pred['boxes'][mask]
                        scores = pred['scores'][mask]
                        
                        for box, score in zip(boxes, scores):
                            class_preds.append({
                                'image_id': img_idx,
                                'box': box.cpu().numpy(),
                                'score': score.item(),
                            })
                
                # Ground truths for this class (labels are 0-indexed)
                gt_labels = gt['labels'].cpu().numpy() if isinstance(gt['labels'], torch.Tensor) else gt['labels']
                gt_boxes = gt['boxes'].cpu().numpy() if isinstance(gt['boxes'], torch.Tensor) else gt['boxes']
                
                mask = gt_labels == class_idx  # Ground truth uses 0-based indexing
                if mask.sum() > 0:
                    for box in gt_boxes[mask]:
                        class_gts.append({
                            'image_id': img_idx,
                            'box': box,
                            'detected': False,
                        })
            
            # Compute AP for this class
            if len(class_preds) == 0:
                ap_per_class[class_name] = 0.0
                continue
            
            if len(class_gts) == 0:
                ap_per_class[class_name] = 0.0
                continue
            
            # Sort predictions by confidence
            class_preds = sorted(class_preds, key=lambda x: x['score'], reverse=True)
            
            # Compute precision and recall
            tp = np.zeros(len(class_preds))
            fp = np.zeros(len(class_preds))
            
            # Group ground truths by image
            gt_by_image = defaultdict(list)
            for gt in class_gts:
                gt_by_image[gt['image_id']].append(gt)
            
            for pred_idx, pred in enumerate(class_preds):
                img_id = pred['image_id']
                pred_box = pred['box']
                
                # Find best matching ground truth
                best_iou = 0.0
                best_gt_idx = -1
                
                for gt_idx, gt in enumerate(gt_by_image[img_id]):
                    if gt['detected']:
                        continue
                    
                    iou = self._compute_iou_single(pred_box, gt['box'])
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                # Check if it's a true positive
                if best_iou >= iou_threshold and best_gt_idx >= 0:
                    if not gt_by_image[img_id][best_gt_idx]['detected']:
                        tp[pred_idx] = 1
                        gt_by_image[img_id][best_gt_idx]['detected'] = True
                    else:
                        fp[pred_idx] = 1
                else:
                    fp[pred_idx] = 1
            
            # Compute precision and recall
            cumulative_tp = np.cumsum(tp)
            cumulative_fp = np.cumsum(fp)
            
            recalls = cumulative_tp / len(class_gts)
            precisions = cumulative_tp / (cumulative_tp + cumulative_fp + 1e-6)
            
            # Compute AP using 11-point interpolation
            ap = self._compute_ap_11point(precisions, recalls)
            ap_per_class[class_name] = ap
        
        return ap_per_class
    
    def _compute_iou_single(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two boxes"""
        # Compute intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Compute union
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        iou = intersection / (union + 1e-6)
        return iou
    
    def _compute_ap_11point(self, precisions: np.ndarray, recalls: np.ndarray) -> float:
        """Compute AP using 11-point interpolation"""
        ap = 0.0
        
        for recall_thresh in np.linspace(0, 1, 11):
            # Get maximum precision for recall >= recall_thresh
            prec_at_recall = precisions[recalls >= recall_thresh]
            
            if len(prec_at_recall) > 0:
                ap += np.max(prec_at_recall)
        
        ap /= 11.0
        return ap
    
    def _print_results(self, results: Dict):
        """Print evaluation results"""
        print("\n" + "=" * 60)
        print("Evaluation Results")
        print("=" * 60)
        
        for iou_thresh in self.iou_thresholds:
            print(f"\nmAP@{iou_thresh}: {results[f'mAP@{iou_thresh}']:.4f}")
            print("\nPer-class AP:")
            
            ap_dict = results[f'AP@{iou_thresh}']
            for class_name, ap in ap_dict.items():
                print(f"  {class_name:20s}: {ap:.4f}")
        
        print("\n" + "=" * 60)


if __name__ == "__main__":
    print("Evaluator module - use via eval.py script")
