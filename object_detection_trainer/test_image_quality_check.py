#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 AI Trainer - Image Quality Check Script
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
import cv2
import os

def check_image_quality(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False, None, None
        height, width, channels = img.shape
        return True, width, height
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return False, None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/images/test', help='image folder to check')
    image_folder = parser.parse_args().input
    valid_images = 0
    invalid_images = 0
    
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, filename)
            valid, width, height = check_image_quality(image_path)
            if valid:
                valid_images += 1
                print(f"Valid Image: {filename} | Size: {width}x{height} | Shape: ({height}, {width})")
            else:
                invalid_images += 1
                print(f"Invalid Image: {filename}")
    
    print(f"Valid images: {valid_images}")
    print(f"Invalid images: {invalid_images}")

if __name__ == "__main__":
    main()
