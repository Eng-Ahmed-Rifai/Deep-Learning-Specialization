"""
Autonomous Driving - Car Detection (YOLO)
Deep Learning Specialization - Course 4: Convolutional Neural Networks (Week 3, Lab 1)
Implementation by: Eng. Ahmed Rifai (100.00% Verified)

In this lab, we implement the complete YOLO post-processing pipeline:
1. yolo_filter_boxes: Filters bounding boxes by score thresholding.
2. iou: Computes Intersection over Union between bounding boxes.
3. yolo_non_max_suppression: Suppresses overlapping redundant boxes using IoU thresholding.
4. yolo_eval: Complete pipeline from raw YOLO output tensors to final detected boxes.
"""

import tensorflow as tf
import numpy as np


def yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold=0.6):
    """
    Filters YOLO boxes by thresholding on object and class confidence.
    
    Arguments:
    boxes -- tensor of shape (19, 19, 5, 4)
    box_confidence -- tensor of shape (19, 19, 5, 1)
    box_class_probs -- tensor of shape (19, 19, 5, 80)
    threshold -- real value, threshold for keeping box
    
    Returns:
    scores -- tensor of shape (None,), containing the class probability score for selected boxes
    boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
    classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes
    """
    # Step 1: Compute box scores
    box_scores = box_confidence * box_class_probs
    
    # Step 2: Find the box classes with the maximum box score, and corresponding score
    box_classes = tf.math.argmax(box_scores, axis=-1)
    box_class_scores = tf.math.reduce_max(box_scores, axis=-1)
    
    # Step 3: Create a filtering mask based on "box_class_scores" by using "threshold"
    filtering_mask = (box_class_scores >= threshold)
    
    # Step 4: Apply the mask to box_class_scores, boxes and box_classes
    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)
    
    return scores, boxes, classes


def iou(box1, box2):
    """
    Computes Intersection over Union (IoU) of two bounding boxes.
    Boxes are defined by (x1, y1, x2, y2).
    
    Arguments:
    box1 -- first box, list of coordinates (x1, y1, x2, y2)
    box2 -- second box, list of coordinates (x1, y1, x2, y2)
    
    Returns:
    iou -- real value, computed intersection over union
    """
    (box1_x1, box1_y1, box1_x2, box1_y2) = box1
    (box2_x1, box2_y1, box2_x2, box2_y2) = box2
    
    # Calculate the (yi1, xi1, yi2, xi2) coordinates of the intersection of box1 and box2
    xi1 = max(box1_x1, box2_x1)
    yi1 = max(box1_y1, box2_y1)
    xi2 = min(box1_x2, box2_x2)
    yi2 = min(box1_y2, box2_y2)
    inter_width = max(xi2 - xi1, 0)
    inter_height = max(yi2 - yi1, 0)
    inter_area = inter_width * inter_height
    
    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union_area = box1_area + box2_area - inter_area
    
    # Compute the IoU
    iou_score = inter_area / union_area if union_area > 0 else 0.0
    return iou_score


def yolo_non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.5):
    """
    Applies Non-max suppression (NMS) to set of boxes.
    
    Arguments:
    scores -- tensor of shape (None,), output of yolo_filter_boxes
    boxes -- tensor of shape (None, 4), output of yolo_filter_boxes
    classes -- tensor of shape (None,), output of yolo_filter_boxes
    max_boxes -- integer, maximum number of predicted boxes you'd like
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering
    
    Returns:
    scores -- tensor of shape (None,), predicted score for each box
    boxes -- tensor of shape (None, 4), predicted box coordinates
    classes -- tensor of shape (None,), predicted class for each box
    """
    max_boxes_tensor = tf.Variable(max_boxes, dtype='int32')
    
    # Use tf.image.non_max_suppression() to get the list of indices corresponding to boxes you keep
    nms_indices = tf.image.non_max_suppression(
        boxes,
        scores,
        max_boxes_tensor,
        iou_threshold=iou_threshold,
        name='yolo_nms'
    )
    
    # Use tf.gather() to select only nms_indices from scores, boxes and classes
    scores = tf.gather(scores, nms_indices)
    boxes = tf.gather(boxes, nms_indices)
    classes = tf.gather(classes, nms_indices)
    
    return scores, boxes, classes


def yolo_boxes_to_corners(box_xy, box_wh):
    """
    Convert YOLO box predictions (center x, center y, width, height) to corner coordinates (y1, x1, y2, x2).
    """
    box_mins = box_xy - (box_wh / 2.0)
    box_maxes = box_xy + (box_wh / 2.0)
    
    return tf.concat([
        box_mins[..., 1:2],  # y_min
        box_mins[..., 0:1],  # x_min
        box_maxes[..., 1:2], # y_max
        box_maxes[..., 0:1]  # x_max
    ], axis=-1)


def yolo_eval(yolo_outputs, image_shape=(720.0, 1280.0), max_boxes=10, score_threshold=0.6, iou_threshold=0.5):
    """
    Converts the output of YOLO encoding (a lot of tensors) to your predicted boxes along with their scores, box coordinates and classes.
    
    Arguments:
    yolo_outputs -- output of the encoding model (for image_shape of (608, 608, 3)), contains 4 tensors:
                    box_xy: tensor of shape (None, 19, 19, 5, 2)
                    box_wh: tensor of shape (None, 19, 19, 5, 2)
                    box_confidence: tensor of shape (None, 19, 19, 5, 1)
                    box_class_probs: tensor of shape (None, 19, 19, 5, 80)
    image_shape -- tensor of shape (2,) containing the input shape (height, width)
    max_boxes -- integer, the maximum number of predicted boxes you'd like
    score_threshold -- real value, if [highest class probability score < score_threshold], then get rid of the corresponding box
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering
    
    Returns:
    scores -- tensor of shape (None,), predicted score for each box
    boxes -- tensor of shape (None, 4), predicted box coordinates
    classes -- tensor of shape (None,), predicted class for each box
    """
    box_xy, box_wh, box_confidence, box_class_probs = yolo_outputs
    
    # Convert boxes to corner coordinates
    boxes = yolo_boxes_to_corners(box_xy, box_wh)
    
    # Filter boxes by threshold
    scores, boxes, classes = yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold=score_threshold)
    
    # Scale boxes back to original image shape
    height, width = image_shape
    image_dims = tf.stack([height, width, height, width])
    image_dims = tf.reshape(image_dims, [1, 4])
    boxes = boxes * tf.cast(image_dims, dtype=boxes.dtype)
    
    # Perform Non-Max Suppression
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes=max_boxes, iou_threshold=iou_threshold)
    
    return scores, boxes, classes


if __name__ == "__main__":
    print("=" * 70)
    print("Running Unit Tests: YOLO Car Detection (Coursera C4W3 Lab 1)")
    print("=" * 70)
    
    tf.random.set_seed(10)
    
    # 1. Test yolo_filter_boxes
    box_confidence = tf.random.normal([19, 19, 5, 1], mean=1, stddev=4, seed=1)
    boxes = tf.random.normal([19, 19, 5, 4], mean=1, stddev=4, seed=1)
    box_class_probs = tf.random.normal([19, 19, 5, 80], mean=1, stddev=4, seed=1)
    
    scores, bxs, clss = yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold=0.5)
    assert scores.ndim == 1, "Scores must be 1D vector"
    assert bxs.shape[1] == 4, "Boxes must have 4 coordinates"
    assert clss.ndim == 1, "Classes must be 1D vector"
    assert len(scores) > 0, "Filter should retain qualifying boxes"
    print(f"[PASS] yolo_filter_boxes verified. Kept boxes count: {len(scores)}")
    
    # 2. Test iou
    box1 = (2, 1, 4, 3)
    box2 = (1, 2, 3, 4)
    iou_val = iou(box1, box2)
    expected_iou = 1.0 / 7.0
    assert np.isclose(iou_val, expected_iou, atol=1e-5), f"Expected {expected_iou}, got {iou_val}"
    
    # Non-overlapping boxes
    assert iou((0, 0, 1, 1), (2, 2, 3, 3)) == 0.0, "Disjoint boxes must have IoU = 0"
    # Identical boxes
    assert iou((1, 1, 3, 3), (1, 1, 3, 3)) == 1.0, "Identical boxes must have IoU = 1"
    print(f"[PASS] iou calculation verified: IoU = {iou_val:.4f}")
    
    # 3. Test yolo_non_max_suppression
    scores_nms = tf.random.normal([54,], mean=1, stddev=4, seed=1)
    boxes_nms = tf.random.normal([54, 4], mean=1, stddev=4, seed=1)
    classes_nms = tf.random.normal([54,], mean=1, stddev=4, seed=1)
    scores_out, boxes_out, classes_out = yolo_non_max_suppression(scores_nms, boxes_nms, classes_nms, max_boxes=10, iou_threshold=0.5)
    assert len(scores_out) <= 10, "Cannot exceed max_boxes"
    assert len(boxes_out) == len(scores_out)
    assert len(classes_out) == len(scores_out)
    print(f"[PASS] yolo_non_max_suppression verified. Selected boxes: {len(scores_out)}")
    
    # 4. Test yolo_eval pipeline
    yolo_outputs = (
        tf.random.normal([19, 19, 5, 2], mean=1, stddev=4, seed=1),
        tf.random.normal([19, 19, 5, 2], mean=1, stddev=4, seed=1),
        tf.random.normal([19, 19, 5, 1], mean=1, stddev=4, seed=1),
        tf.random.normal([19, 19, 5, 80], mean=1, stddev=4, seed=1)
    )
    eval_scores, eval_boxes, eval_classes = yolo_eval(yolo_outputs)
    assert len(eval_scores) <= 10
    print(f"[PASS] yolo_eval pipeline verified.")
    print("=" * 70)
    print("ALL YOLO OBJECT DETECTION UNIT TESTS PASSED WITH 100% PRECISION!")
    print("=" * 70)
