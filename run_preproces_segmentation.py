# -*- coding: utf-8 -*-

"""
Pass arguments that are needed to process files
argument 1: Source path for chest X-rays images 
argument 2: Source path for lung masks
argument 3: Path for storing merged left and right lung masks
argument 4: Path for storing train, val, and test data splits
argument 5: Path for saving augmented images
"""

from preprocess.image_preprocessing import ImagePreprocessing
import argparse


parser = argparse.ArgumentParser(description='Run Preprocessing for Segmentation')

parser.add_argument("-sp", "--source_path", 
                    required=True, 
                    help="Raw Images Path eg:- mention source dataset path you want to provide")
parser.add_argument("-mp", "--mask_path", required=True,
                    help="Masks Path eg:- mention lung mask path you want to provide")
parser.add_argument("-mmp", "--merged_mask_path", required=True,
                    help="Merged Mask Path eg:- mention merged mask path to save left and right lung masks in single image")
parser.add_argument("-dp", "--dest_path", required=True,
                    help="Destination Path eg:- mention destination path for storing train, val, and test data splits")
parser.add_argument("-ap", "--aug_path", required=True,
                    help="Augmented Data Path eg:- mention augmented data path to save augmented train data set")

args = vars(parser.parse_args())


ip = ImagePreprocessing(args["source_path"], args["mask_path"], args["merged_mask_path"] , args["dest_path"], args["aug_path"]) 
ip.run_preprocessing()