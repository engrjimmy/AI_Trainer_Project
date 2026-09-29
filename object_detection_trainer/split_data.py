#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 AI Trainer - Dataset Split Tool
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
Split a Pascal VOC dataset into train/val/test (80/10/10).

USAGE:
python3 split_data.py --data-path <dir containing images/ and Annotations/>
"""

import argparse
import os
import random
import shutil

parser = argparse.ArgumentParser()
parser.add_argument(
    '--data-path', required=True,
    help='directory containing images/ and Annotations/'
)
args = parser.parse_args()

data_path = args.data_path
image_path = os.path.join(data_path, 'images')
label_path = os.path.join(data_path, 'Annotations')

# Path to destination folders
train_folder = os.path.join(image_path, 'train')
val_folder = os.path.join(image_path, 'val')
test_folder = os.path.join(image_path, 'test')

label_train_folder = os.path.join(label_path, 'train')
label_val_folder = os.path.join(label_path, 'val')
label_test_folder = os.path.join(label_path, 'test')

# Define a list of image extensions
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

# Create a list of image filenames in 'image_path'
imgs_list = [
    filename for filename in os.listdir(
        image_path) if os.path.splitext(filename)[-1] in image_extensions]

# Set the random seed 
random.seed(42)

# Shuffle the list of image filenames
random.shuffle(imgs_list)

# Determine the number of images for each set
train_size = int(len(imgs_list) * 0.8)
val_size = int(len(imgs_list) * 0.1)
test_size = int(len(imgs_list) * 0.1)

# Create destination folders if they don't exist
for folder_path in [train_folder, val_folder, test_folder, label_train_folder, label_test_folder, label_val_folder]:
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# Copy image files to destination folders
for i, f in enumerate(imgs_list):
    if i < train_size:
        dest_folder = train_folder
        label_dest_folder = label_train_folder
    elif i < train_size + val_size:
        dest_folder = val_folder
        label_dest_folder = label_val_folder
    else:
        dest_folder = test_folder
        label_dest_folder = label_test_folder

    label_file = f'{os.path.splitext(f)[0]}.xml'
    shutil.copy(os.path.join(image_path, f), os.path.join(dest_folder, f))
    shutil.copy(os.path.join(label_path, label_file), os.path.join(
        label_dest_folder, label_file))

print("Data split completed successfully.")
