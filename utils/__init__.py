"""Utilities package initialization"""

from .box_ops import (
    box_area,
    box_iou,
    clip_boxes_to_image,
    remove_small_boxes,
    box_cxcywh_to_xyxy,
    box_xyxy_to_cxcywh,
    batched_nms,
)

from .visualization import (
    Colors,
    draw_boxes,
    plot_training_curves,
    create_confusion_matrix_plot,
)

__all__ = [
    'box_area',
    'box_iou',
    'clip_boxes_to_image',
    'remove_small_boxes',
    'box_cxcywh_to_xyxy',
    'box_xyxy_to_cxcywh',
    'batched_nms',
    'Colors',
    'draw_boxes',
    'plot_training_curves',
    'create_confusion_matrix_plot',
]
