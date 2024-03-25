#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 14:47:21 2023
Randomly splits the data into training and test sets folders
@author: jaffeux
"""

import os
import random
import shutil
split=0.8 # Trainig proportion
data_path = "All probes_v2/"

for classfolder in os.listdir(data_path):

    
    # path to destination folders
    originfolder= os.path.join(data_path,classfolder)
    train_folder = os.path.join(data_path, 'train',classfolder)
    #val_folder = os.path.join(data_path, 'eval',classfolder)
    test_folder = os.path.join(data_path, 'test',classfolder)
    
    # Define a list of image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    # Create a list of image filenames in 'data_path'
    imgs_list = [filename for filename in os.listdir(originfolder) if os.path.splitext(filename)[-1] in image_extensions]
    # Sets the random seed 
    random.seed(42)
    
    # Shuffle the list of image filenames
    random.shuffle(imgs_list)
    
    # determine the number of images for each set
    train_size = int(len(imgs_list) * split)
    test_size = int(len(imgs_list) * (1-split))
    
    # Create destination folders if they don't exist
    for folder_path in [train_folder, test_folder]:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    
    # Copy image files to destination folders
    for i, f in enumerate(imgs_list):
        if i < train_size:
            dest_folder = train_folder
        else:
            dest_folder = test_folder
        shutil.copy(os.path.join(originfolder, f), os.path.join(dest_folder, f))