# -*- coding: utf-8 -*-

"""
Pass arguments that are needed to process files
argument 1: Source path for input images
argument 2: Destination path for saving models
argument 3: Model to run
"""

from classification.train_transfer_learning_model import TrainClassification
import argparse


parser = argparse.ArgumentParser(description='Train transfer learning architectures for classification')

parser.add_argument("-sp", "--source_path", 
                    required=True, 
                    help="Raw Images Path eg:- mention source dataset path you want to provide")
parser.add_argument("-dp", "--dest_path", required=True,
                    help="Destination Path eg:- mention destination path for storing trained models")
parser.add_argument("-mn", "--model_name", required=True,
                    help="Model name to be trained eg:- mention vgg16, resnet50, inceptV3, densenet201 or xception")

args = vars(parser.parse_args())


tc = TrainClassification(args["source_path"], args["dest_path"], args["model_name"]) 
tc.train_function()