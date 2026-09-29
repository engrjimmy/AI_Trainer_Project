#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 AI Trainer - ONNX Object Detection Script
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
@This script performs object detection using ONNX models for inference.

@param input: An image or a directory containing images for detection.

@Execution Command:
If you generate ONNX model using export.py, use it:
python3 object_detection_onnx_inference.py --input ../inference_data/ --weights weights/model.onnx --data data_configs/box.yaml --show --imgsz 640

Or, if you generate model.onnx from train.py main function, use it:
python3 object_detection_onnx_inference.py --input ../inference_data/ --weights outputs/training/box_training/model.onnx --data data_configs/box.yaml --show --imgsz 640
"""

import argparse
import cv2
import numpy as np
import onnxruntime
import os
import glob
import time
import json
import matplotlib.pyplot as plt
import yaml
import torch

from utils.transforms import infer_transforms, resize
from utils.general import set_infer_dir
from utils.annotations import inference_annotations, convert_detections
from utils.logging import LogJSON

def parse_opt():
    """
    Parse command line arguments for the script.

    @param None
    @stg: Uses argparse to handle command line arguments and returns them as a dictionary.
    @msgs: Command-line argument parsing setup.
    @types: Dictionary
    @returns: Dictionary of parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input', 
        help='Folder path to input image (one image or a folder path)',
    )
    parser.add_argument(
        '--data', 
        default=None,
        help='Path to the data config file'
    )
    parser.add_argument(
        '-w', '--weights', 
        default=None,
        help='Path to trained checkpoint weights'
    )
    parser.add_argument(
        '-th', '--threshold', 
        default=0.3, 
        type=float,
        help='Detection threshold'
    )
    parser.add_argument(
        '-si', '--show',  
        action='store_true',
        help='Visualize output only if this argument is passed'
    )
    parser.add_argument(
        '-mpl', '--mpl-show', 
        dest='mpl_show', 
        action='store_true',
        help='Visualize using matplotlib, helpful in notebooks'
    )
    parser.add_argument(
        '-ims', '--imgsz', 
        default=640,
        type=int,
        help='Resize image to, by default use the original frame/image size'
    )
    parser.add_argument(
        '-nlb', '--no-labels',
        dest='no_labels',
        action='store_true',
        help='Do not show labels on top of bounding boxes'
    )
    parser.add_argument(
        '--classes',
        nargs='+',
        type=int,
        default=None,
        help='Filter classes by visualization, e.g., --classes 1 2 3'
    )
    parser.add_argument(
        '--track',
        action='store_true',
        help='Enable tracking if applicable'
    )
    parser.add_argument(
        '--log-json',
        dest='log_json',
        action='store_true',
        help='Store a JSON log file in the output directory'
    )
    args = vars(parser.parse_args())
    return args

def collect_all_images(dir_test):
    """
    Collect all image file paths from a directory or a single image path.

    @param dir_test: Directory containing images or single image path.
    @stg: Scans the specified directory for image files or adds a single image path to the list.
    @msgs: None
    @types: List of strings
    @returns: List containing all image paths.
    """
    test_images = []
    if os.path.isdir(dir_test):
        image_file_types = ['*.jpg', '*.jpeg', '*.png', '*.ppm']
        for file_type in image_file_types:
            test_images.extend(glob.glob(f"{dir_test}/{file_type}"))
    else:
        test_images.append(dir_test)
    return test_images

def to_numpy(tensor):
    """
    Convert a PyTorch tensor to a NumPy array.

    @param tensor: Input PyTorch tensor.
    @stg: Ensures that the tensor is moved to the CPU and converted to a NumPy array.
    @msgs: None
    @types: numpy.ndarray
    @returns: Converted NumPy array.
    """
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def save_detection_log(out_dir, log_data):
    """
    Save detection log data to a JSON file.

    @param out_dir: Output directory where the log file will be saved.
    @param log_data: List of dictionaries containing detection information.
    @stg: Creates or updates a JSON log file with detection results.
    @msgs: Logs success or error messages.
    @types: None
    @returns: None
    """
    log_file = os.path.join(out_dir, 'detection_log.json')
    try:
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=4)
        print("Log file is also saved.")
    except Exception as e:
        print(f"Error saving log file: {e}")

def main(args):
    """
    Main function to perform object detection using an ONNX model.

    @param args: Dictionary of command line arguments.
    @stg: Loads the ONNX model, processes images from the specified input directory or file,
          performs inference, visualizes and saves results, and optionally logs the detection data in JSON format.
    @msgs: Various print statements to show the current status and setup.
    @types: None
    @returns: None
    """
    np.random.seed(42)
    
    # Load the ONNX model.
    ort_session = onnxruntime.InferenceSession(
        args['weights'], providers=['CPUExecutionProvider']
    )
    with open(args['data']) as file:
        data_configs = yaml.safe_load(file)
        NUM_CLASSES = data_configs['NC']
        CLASSES = data_configs['CLASSES']

    # Set output directory and initialize colors for bounding boxes.
    OUT_DIR = set_infer_dir()
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    DIR_TEST = args['input'] or data_configs['image_path']
    test_images = collect_all_images(DIR_TEST)
    print(f"Number of test images: {len(test_images)}")

    detection_threshold = args['threshold']

    # Initialize JSON logging if required.
    if args['log_json']:
        log_json = LogJSON(os.path.join(OUT_DIR, 'log.json'))

    detection_log_data = []
    frame_count = 0
    total_fps = 0

    for i, image_path in enumerate(test_images):
        image_name = os.path.basename(image_path).split('.')[0]
        orig_image = cv2.imread(image_path)
        frame_height, frame_width, _ = orig_image.shape
        RESIZE_TO = args['imgsz'] or frame_width
        image_resized = resize(orig_image, RESIZE_TO, square=True)
        image = image_resized.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = infer_transforms(image)
        image = torch.unsqueeze(image, 0)
        
        # Perform inference and measure FPS.
        start_time = time.time()
        preds = ort_session.run(
            None, {ort_session.get_inputs()[0].name: to_numpy(image)}
        )
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        total_fps += fps
        frame_count += 1

        outputs = {
            'boxes': torch.tensor(preds[0]),
            'labels': torch.tensor(preds[1]),
            'scores': torch.tensor(preds[2])
        }
        outputs = [outputs]

        if args['log_json']:
            log_json.update(orig_image, image_name, outputs[0], CLASSES)

        detection_info = {
            'image_name': image_name,
            'boxes': outputs[0]['boxes'].tolist(),
            'labels': outputs[0]['labels'].tolist(),
            'scores': outputs[0]['scores'].tolist()
        }
        detection_log_data.append(detection_info)

        if len(outputs[0]['boxes']) != 0:
            # Convert detections and visualize results.
            draw_boxes, pred_classes, scores = convert_detections(
                outputs, detection_threshold, CLASSES, args
            )
            orig_image = inference_annotations(
                draw_boxes, 
                pred_classes, 
                scores,
                CLASSES,
                COLORS, 
                orig_image, 
                image_resized,
                args
            )
            if args['show']:
                cv2.imshow('Prediction', orig_image)
                cv2.waitKey(1)
            if args['mpl_show']:
                plt.imshow(orig_image[:, :, ::-1])
                plt.axis('off')
                plt.show()
        
        # Save results to output directory.
        cv2.imwrite(f"{OUT_DIR}/{image_name}.jpg", orig_image)
        print(f"Processed image {i+1}...")
        print('-'*50)

    print('Test predictions complete.')
    if args['show']:
        cv2.destroyAllWindows()

    if args['log_json']:
        try:
            log_json.save(os.path.join(OUT_DIR, 'log.json'))
            print("Log file is also saved.")
        except Exception as e:
            print(f"Error saving log file: {e}")

    save_detection_log(OUT_DIR, detection_log_data)
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.2f}")

if __name__ == '__main__':
    args = parse_opt()
    main(args)
