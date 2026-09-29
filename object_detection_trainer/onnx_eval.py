#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 AI Trainer - ONNX Evaluation Script
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
@This script is a test script for the evaluation of ONNX inference with inference images.

# Follow execution command:
Python3 <script_name.py> --input_image <path_to_input_image_or_folder> --weights <path_to_model_weights.onnx> --threshold <detection_threshold> --output <path_to_output_directory>

# Use it for current setup:

python3 onnx_eval.py --input_image ../inference_data/ --weights outputs/training/box_training/model.onnx --threshold 0.3 --output outputs/training/box_training/evaluation

"""

import argparse
import json
import os
import cv2
import numpy as np
import onnxruntime

def load_onnx_model(model_path):
    """
    Load the ONNX model from the specified path.
    
    Args:
        model_path (str): Path to the ONNX model file.
    
    Returns:
        onnxruntime.InferenceSession: The loaded ONNX model session.
    
    Raises:
        RuntimeError: If the model fails to load.
    """
    try:
        session = onnxruntime.InferenceSession(model_path)
        print("ONNX model loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to load ONNX model: {e}")
    return session

def verify_onnx_input_shape(session, expected_shape):
    """
    Verify that the ONNX model's input shape matches the expected shape.
    
    Args:
        session (onnxruntime.InferenceSession): The ONNX model session.
        expected_shape (tuple): Expected input shape of the model.
    
    Returns:
        tuple: The verified input shape.
    
    Raises:
        ValueError: If the input shape does not match the expected shape.
    """
    input_shape = session.get_inputs()[0].shape
    if input_shape != expected_shape:
        raise ValueError(f"Model input shape {input_shape} does not match expected shape {expected_shape}.")
    print(f"ONNX model input shape verified: {input_shape}")
    return input_shape

def print_onnx_io_shapes(session):
    """
    Print the input shape and output shapes of the ONNX model.
    
    Args:
        session (onnxruntime.InferenceSession): The ONNX model session.
    """
    input_shape = session.get_inputs()[0].shape
    output_shapes = [output.shape for output in session.get_outputs()]
    
    print(f"ONNX model input shape: {input_shape}")
    print(f"ONNX model output shapes: {output_shapes}")

def prepare_input(image, img_size):
    """
    Prepare the input image for the ONNX model.
    
    Args:
        image (numpy.ndarray): The input image.
        img_size (int): The size to which the image should be resized.
    
    Returns:
        numpy.ndarray: The prepared image for the model.
    """
    image = cv2.resize(image, (img_size, img_size))
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image

def load_image(image_path):
    """
    Load an image from the specified path.
    
    Args:
        image_path (str): Path to the image file.
    
    Returns:
        numpy.ndarray: The loaded image.
    
    Raises:
        ValueError: If the image cannot be loaded.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found: {image_path}")
    return image

def process_outputs(outputs, threshold):
    """
    Process the model outputs to extract detections based on the threshold.
    
    Args:
        outputs (list): The raw outputs from the model.
        threshold (float): The detection threshold.
    
    Returns:
        list: A list of detections where each detection is a dictionary containing 'box', 'score', and 'label'.
    """
    detections = []
    boxes, labels, scores = outputs  # Unpack the outputs

    # Print the raw outputs for debugging
    print(f"Outputs: {outputs}")

    for i in range(len(scores)):
        if scores[i] >= threshold:
            detection = {
                'box': boxes[i].tolist(),
                'score': float(scores[i]),
                'label': int(labels[i])
            }
            detections.append(detection)

            # Print each detection's details
            print(f"Detection: {json.dumps(detection, indent=4)}")
    
    return detections

def get_next_folder_number(base_dir):
    """
    Determine the next folder number to create a unique output directory.
    
    Args:
        base_dir (str): The base directory where the new folder will be created.
    
    Returns:
        int: The next folder number.
    """
    os.makedirs(base_dir, exist_ok=True)
    existing_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    existing_numbers = [int(d[4:]) for d in existing_dirs if d.startswith('eval') and d[4:].isdigit()]
    
    return max(existing_numbers, default=0) + 1

def create_unique_output_dir(base_dir):
    """
    Create a unique output directory based on the next available folder number.
    
    Args:
        base_dir (str): The base directory where the new folder will be created.
    
    Returns:
        str: The path to the newly created unique directory.
    """
    next_number = get_next_folder_number(base_dir)
    unique_dir = os.path.join(base_dir, f"eval{next_number}")
    
    os.makedirs(unique_dir, exist_ok=True)
    return unique_dir

def evaluate_model(session, image_path, threshold, output_dir, img_size):
    """
    Evaluate the model on a single image and save the results.
    
    Args:
        session (onnxruntime.InferenceSession): The ONNX model session.
        image_path (str): Path to the input image.
        threshold (float): The detection threshold.
        output_dir (str): Directory to save the evaluation results.
        img_size (int): The size to which the image should be resized.
    
    Returns:
        list: A list of scores for the detections in the image.
    """
    image = load_image(image_path)
    inputs = prepare_input(image, img_size)
    outputs = session.run(None, {session.get_inputs()[0].name: inputs})

    detections = process_outputs(outputs, threshold)

    result_file = os.path.join(output_dir, f'{os.path.splitext(os.path.basename(image_path))[0]}_results.json')
    
    with open(result_file, 'w') as f:
        json.dump(detections, f, indent=4)

    print(f"Evaluation results for {os.path.basename(image_path)} saved to {result_file}")
    
    return [det['score'] for det in detections]

def evaluate_images(session, image_paths, threshold, output_dir, img_size):
    """
    Evaluate the model on a list of images and generate a summary of results.
    
    Args:
        session (onnxruntime.InferenceSession): The ONNX model session.
        image_paths (list): List of paths to the input images.
        threshold (float): The detection threshold.
        output_dir (str): Directory to save the evaluation results.
        img_size (int): The size to which the image should be resized.
    """
    all_scores = []
    for image_path in image_paths:
        scores = evaluate_model(session, image_path, threshold, output_dir, img_size)
        all_scores.extend(scores)

    average_score = sum(all_scores) / len(all_scores) if all_scores else 0

    # Simple evaluation metrics (for demonstration purposes)
    num_images = len(image_paths)
    num_detections = len(all_scores)
    detection_rate = num_detections / num_images if num_images else 0

    summary = {
        'average_score': average_score,
        'detection_rate': detection_rate,
        'num_images': num_images,
        'num_detections': num_detections
    }
    summary_file = os.path.join(output_dir, 'summary.json')
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=4)

    print(f"Summary saved to {summary_file}")

def get_image_size_from_shape(shape):
    """
    Extract image height and width from the model's input shape.
    
    Args:
        shape (tuple): The shape of the model's input tensor.
    
    Returns:
        tuple: Height and width of the image.
    """
    # Assuming the shape is (batch_size, channels, height, width)
    return shape[2], shape[3]

def main():
    """
    Main function to parse arguments, load the model, and evaluate images.
    """
    parser = argparse.ArgumentParser(description='Run evaluation with an ONNX model')
    parser.add_argument('--input_image', type=str, required=True, help='Path to the input image or directory of images')
    parser.add_argument('--weights', type=str, required=True, help='Path to the ONNX weights file')
    parser.add_argument('--threshold', type=float, default=0.3, help='Detection threshold')
    parser.add_argument('--output', type=str, required=True, help='Base directory to save evaluation results')
    args = parser.parse_args()

    # Load the ONNX model
    session = load_onnx_model(args.weights)
    
    # Print ONNX model input and output shapes
    print_onnx_io_shapes(session)

    # Verify model input shape
    expected_shape = session.get_inputs()[0].shape
    img_height, img_width = get_image_size_from_shape(expected_shape)
    
    # Create a unique output directory
    output_dir = create_unique_output_dir(args.output)

    # Determine image files to process
    if os.path.isdir(args.input_image):
        image_files = [os.path.join(args.input_image, f) for f in os.listdir(args.input_image) if os.path.isfile(os.path.join(args.input_image, f))]
    else:
        image_files = [args.input_image]

    # Evaluate images and generate results
    evaluate_images(session, image_files, args.threshold, output_dir, img_height)

    print("Test completed successfully.")

if __name__ == "__main__":
    main()
