"""
This script organizes the LISA Traffic Light dataset into 
images with traffic lights and images without traffic lights.
"""

import os
import shutil
import pandas as pd
import random

# Paths to datasets
lisa_dataset_path = r'C:\\Users\\jnb20\\Desktop\\Code\\Datasets\\LISA_traffic_lights'
annotations_path = os.path.join(lisa_dataset_path, 'Annotations', 'Annotations')
processed_dataset_path = r'C:\\Users\\jnb20\\Desktop\\Code\\Datasets\\processed\\traffic_light_detection'

os.makedirs(processed_dataset_path, exist_ok=True)

# Subdirectories in processed dataset
splits = ['training', 'validation', 'tests']
classes = ['traffic_lights', 'no_traffic_lights']

for split in splits:
    for cls in classes:
        dir_path = os.path.join(processed_dataset_path, split, cls)
        os.makedirs(dir_path, exist_ok=True)

total_images_copied = 0

def process_images_in_directory(images_path, annotations_path, dataset_name='', clip_name=''):
    images_with_tl = []
    images_without_tl = []

    if not os.path.exists(images_path):
        print(f"No images found at {images_path}")
        return [], []
    if not os.path.exists(annotations_path):
        print(f"No annotations found at {annotations_path}")
        return [], []

    # Assume annotation file is 'frameAnnotationsBOX.csv'
    annotation_csv = os.path.join(annotations_path, 'frameAnnotationsBOX.csv')
    if not os.path.exists(annotation_csv):
        print(f"No annotation file found at {annotations_path}")
        return [], []

    # LISA frameAnnotationsBOX.csv file is delimited by ';'
    annotations = pd.read_csv(annotation_csv, delimiter=';')

    annotations['Filename'] = annotations['Filename'].apply(lambda x: os.path.basename(x).lower())

    # Collect all images and normalize filenames
    images_in_dir = [os.path.join(images_path, f) for f in os.listdir(images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    images_in_dir_filenames = [os.path.basename(img).lower() for img in images_in_dir]
    images_in_dir_path_map = {os.path.basename(img).lower(): img for img in images_in_dir}

    # Create a set of images with traffic lights from annotations
    images_with_tl_set = set(annotations['Filename'])

    # Identify images with and without traffic lights
    images_with_tl = [images_in_dir_path_map[img_name] for img_name in images_in_dir_filenames if img_name in images_with_tl_set]
    images_without_tl = [images_in_dir_path_map[img_name] for img_name in images_in_dir_filenames if img_name not in images_with_tl_set]

    total_images = len(images_in_dir)
    dataset_info = f"{dataset_name}/{clip_name}" if clip_name else dataset_name
    print(f"Total images in {dataset_info}: {total_images}")
    print(f"Collected {len(images_with_tl)} images with traffic lights")
    print(f"Collected {len(images_without_tl)} images without traffic lights")

    return images_with_tl, images_without_tl

def process_dataset(dataset_name):
    dataset_images_path = os.path.join(lisa_dataset_path, dataset_name, dataset_name)
    dataset_annotations_path = os.path.join(annotations_path, dataset_name)

    images_with_tl = []
    images_without_tl = []

    if not os.path.exists(dataset_images_path) or not os.path.exists(dataset_annotations_path):
        print(f"Dataset {dataset_name} is missing images or annotations.")
        return [], []

    # Check if 'frames' directory exists directly under 'dataset_images_path'
    frames_dir = os.path.join(dataset_images_path, 'frames')
    if os.path.exists(frames_dir):
        # Process images directly under 'frames' directory (for daySequence and nightSequence datasets)
        with_tl, without_tl = process_images_in_directory(frames_dir, dataset_annotations_path, dataset_name)
        images_with_tl.extend(with_tl)
        images_without_tl.extend(without_tl)
    else:
        # Process clips (for dayTrain and nightTrain datasets)
        clips = [d for d in os.listdir(dataset_images_path) if os.path.isdir(os.path.join(dataset_images_path, d))]
        if not clips:
            print(f"No clips found in dataset {dataset_name}")
            return [], []
        for clip in clips:
            clip_images_path = os.path.join(dataset_images_path, clip, 'frames')
            clip_annotations_path = os.path.join(dataset_annotations_path, clip)
            with_tl, without_tl = process_images_in_directory(clip_images_path, clip_annotations_path, dataset_name, clip)
            images_with_tl.extend(with_tl)
            images_without_tl.extend(without_tl)

    return images_with_tl, images_without_tl

# Collect all images
images_with_tl_all = []
images_without_tl_all = []

# Datasets to process (include all desired datasets)
datasets = ['dayTrain', 'nightTrain']

for dataset_name in datasets:
    with_tl, without_tl = process_dataset(dataset_name)
    images_with_tl_all.extend(with_tl)
    images_without_tl_all.extend(without_tl)

# After collecting all images
print(f"Total images with traffic lights: {len(images_with_tl_all)}")
print(f"Total images without traffic lights: {len(images_without_tl_all)}")

random.shuffle(images_with_tl_all)
random.shuffle(images_without_tl_all)

# Split ratios
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Helper function to split data
def split_data(data_list):
    total = len(data_list)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    return data_list[:train_end], data_list[train_end:val_end], data_list[val_end:]

with_tl_train, with_tl_val, with_tl_test = split_data(images_with_tl_all)
without_tl_train, without_tl_val, without_tl_test = split_data(images_without_tl_all)

# Data split summary
print("Data split:")
print(f"Training set - Traffic Lights: {len(with_tl_train)}, No Traffic Lights: {len(without_tl_train)}")
print(f"Validation set - Traffic Lights: {len(with_tl_val)}, No Traffic Lights: {len(without_tl_val)}")
print(f"Test set - Traffic Lights: {len(with_tl_test)}, No Traffic Lights: {len(without_tl_test)}")

def copy_images(image_list, split, cls):
    global total_images_copied  # Access the global counter
    print(f"Copying {len(image_list)} images to {split}/{cls}")
    for src_path in image_list:
        if not os.path.exists(src_path):
            print(f"Warning: Source image not found {src_path}")
            continue
        filename = os.path.basename(src_path)
        # To avoid filename conflicts, include relative path in filename
        parent_dirs = os.path.relpath(os.path.dirname(src_path), lisa_dataset_path).replace('\\', '_').replace('/', '_')
        new_filename = f"{parent_dirs}_{filename}"
        dst_path = os.path.join(processed_dataset_path, split, cls, new_filename)
        shutil.copyfile(src_path, dst_path)
        total_images_copied += 1  # Increment the counter

copy_images(with_tl_train, 'training', 'traffic_lights')
copy_images(with_tl_val, 'validation', 'traffic_lights')
copy_images(with_tl_test, 'tests', 'traffic_lights')

copy_images(without_tl_train, 'training', 'no_traffic_lights')
copy_images(without_tl_val, 'validation', 'no_traffic_lights')
copy_images(without_tl_test, 'tests', 'no_traffic_lights')

print(f"Copied {total_images_copied} total images")

print("Data copying complete.")
