# -*- coding: utf-8 -*-

"""
Pass arguments that are needed to process files
argument 1: Source path for chest X-rays images 
argument 2: Dest path for storing preprocessed images (Segmented + Bounding Box) for classification problem statement
"""

from preprocess.create_lung_box import LungBox
import argparse


parser = argparse.ArgumentParser(description='Run Preprocessing for Classification')

parser.add_argument("-sp", "--source_path", 
                    required=True, 
                    help="Raw Images Path eg:- mention source dataset path you want to provide")
parser.add_argument("-dp", "--dest_path", required=True,
                    help="Destination Path eg:- mention destination path for storing train, val, and test data splits")


args = vars(parser.parse_args())


lb = LungBox(args["source_path"], args["dest_path"]) 
lb.create_bounding_box()