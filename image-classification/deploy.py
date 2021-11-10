import os
import numpy as np
from argparse import ArgumentParser
from lib.Autils_classification import predict_image

def prepare_parser():
  parser = ArgumentParser(description='Model training')
  parser.add_argument(
      "--model",
      default='model/data_NVS-proj/2D_CNN-MClss_cpu.pth',
      metavar="FILE",
      help="path to trainde model",
      type=str,
  )
  parser.add_argument(
      "--data",
      default="data/deploy",
      metavar="DIRECTORY",
      help="Directory where input images are stored."
  )
  return parser

def main():
    parser = prepare_parser()
    args =  parser.parse_args()
    model_path = args.model
    from glob import glob
    import random
    list_images_paths = glob(os.path.join(args.data , "*.jpg" ) ) 
    # load classes
    import json
    with open('config/classes.json') as json_file:
        classes = json.load(json_file)
    if list_images_paths==[]:
        print('No data was found in the selected directory:', args.data )
        return 
    # deploy the classification model 
    prediction_df = predict_image(args.model, classes, list_images_paths, plotting=True)
    print('\n prediction results : \n', prediction_df)
    # save resut tables
    prediction_df.to_csv( 'output/predictions_ ' + str(os.path.basename(model_path) ) + '.csv', sep=',')

if __name__ == '__main__':
    main()
