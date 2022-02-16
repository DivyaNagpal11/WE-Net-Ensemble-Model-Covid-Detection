# -*- coding: utf-8 -*-

"""
Pass arguments that are needed to process files
argument 1: Test images path
argument 2: Saved models path
"""

from ensemble.ensembled_techniques import EnsembleModels
import argparse

parser = argparse.ArgumentParser(description='Build ensmeble models')

parser.add_argument("-tp", "--test_path", 
                    required=True, 
                    help="Test images Path eg:- mention test image dataset path you want to provide")
parser.add_argument("-sm", "--saved_model", required=True,
                    help="Saved Models Path eg:- provide path for saved classification models")

args = vars(parser.parse_args())

em = EnsembleModels(args["test_path"], args["saved_model"]) 
em.trigger_functions()