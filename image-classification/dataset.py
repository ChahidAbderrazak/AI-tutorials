
# structure inspired from with modification and adaptations:
# - https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/master/train.py

import os
import sys
import yaml
import numpy as np
from argparse import ArgumentParser
from lib.utils import *

def prepare_parser():
  parser = ArgumentParser(description='Model training')
  parser.add_argument(
      "--cfg",
      default="config/config.yml",
      metavar="FILE",
      help="path to config file",
      type=str,
  )
  return parser

def run_build_dataset_workspace(config):
    print('###########################################################')
    print('#        Building the Dataset workspace from Raw data     #')
    print('###########################################################')

    #------------------------------ Raw Data  --------------------------------------
    build_wrkspce = 'Yes'
    WORKSPACE_folder = config['DATASET']['workspace']
    RAW_DATA_ROOT =  config['DATASET']['raw_dataset']
    ext_list = config['DATASET']['ext_list']   # extention of raw data
    size = (int ( config['DATASET']['imgSizes_X'] ), int( config['DATASET']['imgSizes_Y'] ) )     
    #----------------------------- Output model  ----------------------------------
    DIR_WORKSPACE, _  =  get_workspace_path(RAW_DATA_ROOT, WORKSPACE_folder)
    DIR_TRAIN, DIR_TEST, DIR_DEPLOY = get_workspace_folders(DIR_WORKSPACE)
    # Build the work space for training
    build_dataset_workspace(build_wrkspce, RAW_DATA_ROOT, ext_list, size, DIR_TRAIN, DIR_TEST, DIR_DEPLOY)
    
def main():
  parser = prepare_parser()
  args =  parser.parse_args()
  # Read config parameters from the sYAML file
  with open(args.cfg, 'r') as stream:
      config = yaml.safe_load(stream)
  # Build the workspace
  run_build_dataset_workspace(config)

if __name__ == '__main__':
    main()
