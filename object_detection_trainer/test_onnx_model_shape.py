#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 AI Trainer - ONNX Model Shape Test Script
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

import argparse
import json
import os
import cv2
import numpy as np
import onnxruntime

def load_onnx_model(model_path):
    try:
        session = onnxruntime.InferenceSession(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load ONNX model: {e}")
    return session

def verify_onnx_input_shape(session, expected_shape):
    input_shape = session.get_inputs()[0].shape
    if input_shape != expected_shape:
        raise ValueError(f"Model input shape {input_shape} does not match expected shape {expected_shape}.")
    print(f"ONNX model input shape verified: {input_shape}")
    return input_shape

def prepare_input(image, img_size):
    image = cv2.resize(image, (img_size, img_size))
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found: {image_path}")
    return image

def process_outputs(outputs, threshold):
    detections = []
    boxes, labels, scores = outputs  # Unpack the outputs

    for i in range(len(scores)):
        if scores[i] >= threshold:
            detections.append({
                'box': boxes[i].tolist(),
                'score': float(scores[i]),
                'label': int(labels[i])
            })
    
    return detections

def get_next_folder_number(base_dir):
    os.makedirs(base_dir, exist_ok=True)
    existing_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    existing_numbers = [int(d[4:]) for d in existing_dirs if d.startswith('eval') and d[4:].isdigit()]
    
    return max(existing_numbers, default=0) + 1

def create_unique_output_dir(base_dir):
    next_number = get_next_folder_number(base_dir)
    unique_dir = os.path.join(base_dir, f"eval{next_number}")
    
    os.makedirs(unique_dir, exist_ok=True)
    return unique_dir

def evaluate_model(session, image_path, threshold, output_dir, img_size):
    image = load_image(image_path)
    inputs = prepare_input(image, img_size)
    outputs = session.run(None, {session.get_inputs()[0].name: inputs})

    detections = process_outputs(outputs, threshold)

    # Print the evaluation results (Score and Box) before saving
    print(f"Evaluation results for {image_path}:")
    for det in detections:
        print(f"Label: {det['label']}, Score: {det['score']:.4f}, Box: {det['box']}")

    result_file = os.path.join(output_dir, f'{os.path.splitext(os.path.basename(image_path))[0]}_results.json')
    
    with open(result_file, 'w') as f:
        json.dump(detections, f, indent=4)

    print(f"Results saved to {result_file}")
    
    return [det['score'] for det in detections]

def evaluate_images(session, image_paths, threshold, output_dir, img_size):
    all_scores = []
    for image_path in image_paths:
        scores = evaluate_model(session, image_path, threshold, output_dir, img_size)
        all_scores.extend(scores)

    average_score = sum(all_scores) / len(all_scores) if all_scores else 0
    
    summary = {'average_score': average_score}
    summary_file = os.path.join(output_dir, 'summary.json')
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=4)

    print(f"Summary saved to {summary_file}")

def get_image_size_from_shape(shape):
    # Assuming the shape is (batch_size, channels, height, width)
    return shape[2], shape[3]

def main():
    parser = argparse.ArgumentParser(description='Run evaluation with an ONNX model')
    parser.add_argument('--input_image', type=str, required=True, help='Path to the input image or directory of images')
    parser.add_argument('--weights', type=str, required=True, help='Path to the ONNX weights file')
    parser.add_argument('--threshold', type=float, default=0.3, help='Detection threshold')
    parser.add_argument('--output', type=str, required=True, help='Base directory to save evaluation results')
    args = parser.parse_args()

    session = load_onnx_model(args.weights)

    # Verify model input shape
    expected_shape = session.get_inputs()[0].shape
    img_height, img_width = get_image_size_from_shape(expected_shape)

    output_dir = create_unique_output_dir(args.output)

    if os.path.isdir(args.input_image):
        image_files = [os.path.join(args.input_image, f) for f in os.listdir(args.input_image) if os.path.isfile(os.path.join(args.input_image, f))]
    else:
        image_files = [args.input_image]

    evaluate_images(session, image_files, args.threshold, output_dir, img_height)

if __name__ == "__main__":
    main()
