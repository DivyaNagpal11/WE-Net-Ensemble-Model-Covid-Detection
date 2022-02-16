# -*- coding: utf-8 -*-

"""
Pass arguments that are needed to process files
argument 1: Source path consisting of train, val and test folders for segmentation
"""

from segmentation.train_unet_model import TrainSegmentation
import argparse


parser = argparse.ArgumentParser(description='Train UNet model for Segmentation')

parser.add_argument("-sp", "--source_path", 
                    required=True, 
                    help="Raw Images Path eg:- mention source dataset path you want to provide")

args = vars(parser.parse_args())

ts = TrainSegmentation(args["source_path"]) 
ts.training()