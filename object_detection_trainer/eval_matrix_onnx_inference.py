#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 AI Trainer - ONNX Evaluation Matrix Script
================================================================================

 Author:        Jimmy Majumder
 Position:      Sr. Robotics Engineer
 Organization:  QibiTech Inc.
 Email:         j.majumder@qibitech.com
 Date:          2024_Aug

 Copyright (c) 2024 QibiTech Inc.
 All rights reserved.

 NOTICE:
 This code and data is for research and testing purposes only.
 DO NOT use for product development or production phase.
 For commercial use or inquiries, contact: j.majumder@qibitech.com

================================================================================
"""

"""
This script evaluates ONNX model inference on test images and compares results with ground truth annotations.

# Follow execution command:
python3 <script_name.py> --input_image <path_to_input_image_or_folder> --ground_truth <path_to_ground_truth_annotations> --weights <path_to_model_weights.onnx> --threshold <detection_threshold> --output <path_to_output_directory>

# Example usage:

python3 eval_matrix_onnx_inference.py --input_image data/images/test --ground_truth data/annotations/test --weights outputs/training/box_training/model.onnx --threshold 0.3 --output outputs/training/box_training/evaluation

"""

import argparse
import os
import cv2
import numpy as np
import onnxruntime as ort
import xml.etree.ElementTree as ET
import json
from typing import List, Dict, Tuple

def parse_xml_for_annotations(xml_path: str) -> List[Dict]:
    """
    Parse XML file for object annotations.

    @Args:
        xml_path (str): Path to the XML file containing annotations.

    @Returns:
        List[Dict]: List of dictionaries with labels and bounding boxes.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    annotations = []
    for obj in root.findall('object'):
        label = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        annotations.append({'label': label, 'bbox': [xmin, ymin, xmax, ymax]})
    return annotations

def load_ground_truth(ground_truth_path: str) -> List[Dict]:
    """
    Load ground truth annotations from XML files.

    @Args:
        ground_truth_path (str): Path to the directory containing ground truth XML files.

    @Returns:
        List[Dict]: List of all annotations from the ground truth files.
    """
    annotations = []
    for xml_file in os.listdir(ground_truth_path):
        if xml_file.endswith('.xml'):
            file_path = os.path.join(ground_truth_path, xml_file)
            annotations.extend(parse_xml_for_annotations(file_path))
    return annotations

def load_image(image_path: str) -> np.ndarray:
    """
    Load and validate the input image.

    @Args:
        image_path (str): Path to the image file.

    @Returns:
        np.ndarray: Loaded image if successful, otherwise None.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Image at {image_path} could not be loaded.")
        return None
    return image

def check_image(image: np.ndarray) -> None:
    """
    Print the shape and size of the image for verification.

    @Args:
        image (np.ndarray): Image array to check.

    @Returns:
        None
    """
    if image is not None:
        print(f"Image shape: {image.shape}")
        print(f"Image size: {image.size} bytes")
    else:
        print("No image to check.")


def preprocess_image(image_path: str, target_size: Tuple[int, int] = (640, 480)) -> np.ndarray:
    """
    Preprocess the input image for model inference.

    @Args:
        image_path (str): Path to the image file.
        target_size (Tuple[int, int]): Target size for resizing the image. Default is (640, 480).

    @Returns:
        np.ndarray: Preprocessed image ready for model inference or None if the image couldn't be loaded.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Image at {image_path} could not be loaded.")
        return None
    
    # Ensure image is in color (3 channels). If it's grayscale, convert it to color.
    if len(image.shape) == 2:  # grayscale image
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize the image to the target size
    image = cv2.resize(image, target_size)
    
    # Normalize the image to [0, 1] range
    image = image.astype(np.float32) / 255.0
    
    # Convert to CHW format (Channel-Height-Width)
    image = np.transpose(image, (2, 0, 1))
    
    # Add batch dimension (NCHW format, where N=1)
    image = np.expand_dims(image, axis=0)
    
    return image

#Optional Function: For getting bounding box images use it or Comment out 
#we also have <object_detection_onnx_inference.py> test script so you can comment out if not needed.
def draw_bounding_boxes(image: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """
    Draw bounding boxes on the image.

    @Args:
        image (np.ndarray): The image to draw bounding boxes on.
        detections (List[Dict]): List of detections, each containing 'box', 'score', and 'label'.

    @Returns:
        np.ndarray: Image with bounding boxes drawn.
    """
    for detection in detections:
        box = detection['box']
        label = detection['label']
        score = detection['score']
        
        # Draw the bounding box
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        
        # Draw the label and score
        label_text = f"Label: {label}, Score: {score:.2f}"
        cv2.putText(image, label_text, (int(box[0]), int(box[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image


def evaluate_model(session: ort.InferenceSession, image_path: str, ground_truth_path: str, threshold: float, output_dir: str, img_size: Tuple[int, int]) -> Tuple[List[Dict], float, float, float]:
    """
    Evaluate the model on a given image by running inference, comparing results with ground truths,
    and calculating evaluation metrics such as precision, recall, and F1-score.

    @Args:
        session (ort.InferenceSession): ONNX runtime inference session for the model.
        image_path (str): Path to the input image for evaluation.
        ground_truth_path (str): Path to the file containing ground truth data.
        threshold (float): Confidence threshold for considering a detection valid.
        output_dir (str): Directory to save output images and detection results.
        img_size (Tuple[int, int]): Target image size for preprocessing.

    @Returns:
        Tuple[List[Dict], float, float, float]: List of detections, precision, recall, and F1-score values.
    """
    try:
        # Load and preprocess image
        image = preprocess_image(image_path, img_size)
        if image is None:
            print(f"Skipping {image_path} due to preprocessing issue.")
            return [], 0, 0, 0, (0, 0, 0)
        
        # Run inference
        inputs = {session.get_inputs()[0].name: image}
        outputs = session.run(None, inputs)
        
        # Extract and print raw inference outputs
        boxes, labels, scores = outputs
        print("Inference outputs:")
        print("Bounding Boxes:\n", boxes)
        print("Labels:\n", labels)
        print("Scores:\n", scores)
        
        # Boxes come back in model input size; scale to the original image.
        orig_image = cv2.imread(image_path)
        orig_h, orig_w = orig_image.shape[:2]
        scale = np.array([orig_w / img_size[0], orig_h / img_size[1]] * 2)

        detections = []
        for i in range(len(scores)):
            if scores[i] >= threshold:
                detections.append({
                    'box': (boxes[i] * scale).tolist(),
                    'score': float(scores[i]),
                    'label': int(labels[i])
                })
        
        xml_path = os.path.join(
            ground_truth_path, os.path.splitext(os.path.basename(image_path))[0] + '.xml')
        ground_truths = parse_xml_for_annotations(xml_path) if os.path.exists(xml_path) else []

        precision, recall, f1_score, counts = calculate_metrics(detections, ground_truths)
        
        # Save results
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Optional: Draw bounding boxes on the image of comment out 
        #Draw bouidng boxes for saving 
        image_bgr = draw_bounding_boxes(orig_image, detections)
        
        # Optional: If you don't use draw bounding box then use it 
        # Convert the image back to BGR for saving
        # image_bgr = np.squeeze(image, axis=0).transpose(1, 2, 0) * 255
        # image_bgr = np.clip(image_bgr, 0, 255).astype(np.uint8)
        
        result_image_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(result_image_path, image_bgr)
        
        detections_file = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + '_detections.json')
        with open(detections_file, 'w') as f:
            json.dump(detections, f, indent=4)
        
        return detections, precision, recall, f1_score, counts
    
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return [], 0, 0, 0, (0, 0, 0)


def calculate_metrics(detections: List[Dict], ground_truths: List[Dict]) -> Tuple[float, float, float]:
    """
    Calculate precision, recall, and F1-score based on detections and ground truths.

    @Args:
        detections (List[Dict]): List of detected objects with their bounding boxes, labels, and confidence scores.
        ground_truths (List[Dict]): List of ground truth objects with their bounding boxes and labels.

    @Returns:
        Tuple: Precision, recall, F1-score and (TP, FP, FN) counts.
    """
    def iou(boxA, boxB):
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.

        @Args:
            boxA: First bounding box.
            boxB: Second bounding box.

        @Returns:
            float: IoU value.
        """      
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        iou_value = interArea / float(boxAArea + boxBArea - interArea)
        return iou_value

    TP, FP, FN = 0, 0, 0
    matched_gt = set()
    for det in sorted(detections, key=lambda d: d['score'], reverse=True):
        best_iou, best_j = 0.5, None  # IoU threshold
        for j, gt in enumerate(ground_truths):
            if j not in matched_gt:
                v = iou(det['box'], gt['bbox'])
                if v > best_iou:
                    best_iou, best_j = v, j
        if best_j is not None:
            matched_gt.add(best_j)
            TP += 1
        else:
            FP += 1

    FN = len(ground_truths) - TP
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"True Positives: {TP}, False Positives: {FP}, False Negatives: {FN}")
    print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}")
    
    return precision, recall, f1_score, (TP, FP, FN)

def get_next_folder_number(base_dir: str) -> int:
    """
    Get the next folder number for creating a unique evaluation directory.

    @Args:
        base_dir (str): Base directory where evaluation folders are stored.

    @Returns:
        int: The next available folder number.
    """
    existing_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    existing_numbers = [int(d[4:]) for d in existing_dirs if d.startswith('eval') and d[4:].isdigit()]
    return max(existing_numbers, default=0) + 1

def create_unique_output_dir(base_dir: str) -> str:
    """
    Create a unique output directory for the current evaluation inside the base directory.
    If the evaluation directory does not exist, create it.

    @Args:
        base_dir (str): Base directory where evaluation folders are stored.

    @Returns:
        str: Path to the created unique evaluation directory.
    """
    # Ensure the base directory exists
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Create the evaluation directory if it does not exist
    eval_dir = os.path.join(base_dir, 'evaluation')
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    # Get the next folder number for unique naming
    next_number = get_next_folder_number(eval_dir)
    unique_dir = os.path.join(eval_dir, f"eval{next_number}")
    os.makedirs(unique_dir, exist_ok=True)
    
    return unique_dir


def save_summary(summary: Dict, output_dir: str):
    """
    Save the evaluation summary to a JSON file.

    @Args:
        summary (Dict): Dictionary containing summary metrics and evaluation results.
        output_dir (str): Directory where the summary JSON file should be saved.

    @Returns:
        None
    """
    summary_file = os.path.join(output_dir, 'summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=4)

def main():
    """
    - Parses command-line arguments for input images, ground truth annotations, model weights, and output directory.
    - Loads the ONNX model for inference.
    - Processes and evaluates each image in the input directory:
        - Preprocesses images to match model input requirements.
        - Runs model inference to detect objects.
        - Filters detections by confidence threshold and compares with ground truth.
        - Calculates and logs precision, recall, and F1-score.
    - Saves evaluation results, including metrics and detections, in a uniquely named output directory.
    """
    parser = argparse.ArgumentParser(description="Evaluate ONNX model inference.")
    parser.add_argument('--input_image', type=str, required=True, help='Path to input images directory.')
    parser.add_argument('--ground_truth', type=str, required=True, help='Path to ground truth annotations directory.')
    parser.add_argument('--weights', type=str, required=True, help='Path to ONNX model weights.')
    parser.add_argument('--threshold', type=float, default=0.3, help='Confidence threshold for detections.')
    parser.add_argument('--output', type=str, required=True, help='Base directory to save output results.')
    args = parser.parse_args()
    
    # Load ONNX model
    session = ort.InferenceSession(args.weights)
    img_size = (640, 640)  # Adjust if needed
    
    # Create unique output directory
    output_dir = create_unique_output_dir(args.output)
    
    # Evaluate images
    image_files = [os.path.join(args.input_image, f) for f in os.listdir(args.input_image)]
    all_scores = []
    all_detections = []
    total_tp = total_fp = total_fn = 0
    
    for image_file in image_files:
        print(f"Evaluating {image_file}...")
        detections, precision, recall, f1_score, (tp, fp, fn) = evaluate_model(session, image_file, args.ground_truth, args.threshold, output_dir, img_size)
        total_tp, total_fp, total_fn = total_tp + tp, total_fp + fp, total_fn + fn
        all_detections.extend(detections)
        all_scores.extend([det['score'] for det in detections])
        print(f"Detections: {detections}")
        print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}")
        
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    print(f"Overall -> TP: {total_tp}, FP: {total_fp}, FN: {total_fn}, "
          f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}")

    # Save summary
    summary = {
        'true_positives': total_tp,
        'false_positives': total_fp,
        'false_negatives': total_fn,
        'average_score': sum(all_scores) / len(all_scores) if all_scores else 0,
        'detection_rate': len(all_scores) / len(image_files) if image_files else 0,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'num_images': len(image_files),
        'num_detections': len(all_scores)
    }
    save_summary(summary, output_dir)

    print("Test completed successfully.")


if __name__ == '__main__':
    """
    Entry point for the script. Executes the main function when the script is run.
    """
    main()
