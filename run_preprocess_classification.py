# -*- coding: utf-8 -*-

"""
Pass arguments that are needed to process files
argument 1: Source path for chest X-rays images 
argument 2: Dest path for storing preprocessed images (Segmented + Bounding Box) for classification problem statement
argument 3: Model path for saved segmentation model
argument 4: Model name of the saved segmentation model
"""

from preprocess.create_lung_box import LungBox
import argparse


parser = argparse.ArgumentParser(description='Run preprocessing for classification using saved segmentation model')

parser.add_argument("-sp", "--source_path", 
                    required=True, 
                    help="Raw Images Path - mention source dataset path for the classification models")
parser.add_argument("-dp", "--dest_path", required=True,
                    help="Destination Path - mention destination path for storing train, val, and test data splits")
parser.add_argument("-mp", "--model_path", required=True,
                    help="Model Path - mention model path for saved segmentation model")
parser.add_argument("-mn", "--model_name", required=True,
                    help="Model name - mention model name of the saved segmentation model")

args = vars(parser.parse_args())

lb = LungBox(args["source_path"], args["dest_path"], args["model_path"], args["model_name"]) 
lb.create_bounding_box()