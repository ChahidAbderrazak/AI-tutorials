import numpy as np
import os
import sys
import yaml
import argparse
from argparse import ArgumentParser
from torchvision import datasets, transforms, utils
from tqdm import tqdm, trange
from lib.Autils_classification import train_model, test_model, classification_performance
from lib.networks import get_model_instance
from lib.utils import get_workspace_path, get_workspace_folders, parse_devices

def prepare_parser():
    parser = ArgumentParser(description='Model training')
    parser.add_argument(
        "--cfg",
        default="config/config.yml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser

def run_model_training(config):
    WORKSPACE_folder = config['DATASET']['workspace']
    model_name = config['MODEL']['model_name'] 
    RAW_DATA_ROOT =  config['DATASET']['raw_dataset']
    DIR_WORKSPACE, data_TAG  =  get_workspace_path(RAW_DATA_ROOT, WORKSPACE_folder)
    DIR_TRAIN, DIR_TEST, _ = get_workspace_folders(DIR_WORKSPACE)
    classes = sorted(os.listdir(DIR_TRAIN))
    # # output model 
    import torch
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model_folder = os.path.join(config['MODEL']['model_dst'], data_TAG )
    model_path = os.path.join(model_folder ,  '2D_' + model_name + '_' + str(device)+ '.pth')
    # model raining parameters
    nb_folds = int( config['TRAIN']['nb_folds'] )                        # number of folds
    num_epoch = int( config['TRAIN']['num_epoch'] )                      # number of num_epoch 
    split_size = float( config['TRAIN']['split_size'] )                  # train/val split
    batch_size= int( config['TRAIN']['batch_size'] )                     # batch size
    num_workers= int( config['TRAIN']['num_workers'] )                   # num workers
    loss_critereon = config['TRAIN']['loss_critereon']                    # loss criteria
    optimizer = config['TRAIN']['optimizer']                              # optimizer to adjust thr Network weights
    lr = float( config['TRAIN']['lr'] )                                   # learning rate
    transfer_learning = config['TRAIN']['transfer_learning']              # enable  transfer learning
    es_patience_ratio  = config['TRAIN']['es_patience_ratio']             # ~10 Ratio of epochs to be used as early stopping critereon  (num_epoch/es_patience_ratio)
    es_patience = int(num_epoch/es_patience_ratio)+1                      #This is required for early stopping, the number of num_epoch we will wait with no improvement before stopping

    # model instanciation 
    clf_model = get_model_instance(model_name, classes)
    print('###########################################################')
    print('#        Training the model %s  using the following parameters: '%(model_name))
    print('#  device=%s  ,  nb_folds=%d  ,  batch_size=%d  ,  num_epoch=%d  ,  num_workers=%d   '%(device, nb_folds, batch_size, num_epoch, num_workers ))
    print('#  loss_critereon=%s  ,  optimizer=%s  ,  lr=%d  ,  Early stop after=%d epochs'%(loss_critereon, optimizer, lr, es_patience ))
    print('#  Dataset=%s  \n#  model will be saved in =%s   '%(DIR_WORKSPACE, model_path ))
    print('###########################################################')

    # Run the training algorithm
    model, classes, epoch_nums, training_loss, validation_loss = train_model(DIR_TRAIN, clf_model, model_path, nb_folds=nb_folds, num_epoch=num_epoch, lr=lr, optimizer=optimizer,\
                                                                            loss_criteria=loss_critereon, split_size=split_size, batch_size=batch_size, es_patience=es_patience,\
                                                                            num_workers=num_workers, transfer_learning=transfer_learning)
    # Test the trained model
    truelabels, predictions, TS_sz = test_model(model_path, DIR_TEST, classes)
    # show performance
    ACC, recall, Precision, F1_score = classification_performance(classes, truelabels, predictions, TS_sz= TS_sz)#, TR_sz= len(train_loader.dataset) + len(val_loader.dataset))

    return model , classes

def run_model_dev(config):
    WORKSPACE_folder = config['DATASET']['workspace']
    model_name = config['MODEL']['model_name'] 
    RAW_DATA_ROOT =  config['DATASET']['raw_dataset']
    DIR_WORKSPACE, data_TAG  =  get_workspace_path(RAW_DATA_ROOT, WORKSPACE_folder)
    DIR_TRAIN, DIR_TEST, _ = get_workspace_folders(DIR_WORKSPACE)
    classes = sorted(os.listdir(DIR_TRAIN))
    # # output model 
    import torch
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model_folder = os.path.join(config['MODEL']['model_dst'], data_TAG )
    model_path = os.path.join(model_folder ,  '2D_' + model_name + '_' + str(device)+ '.pth')
    # model raining parameters
    nb_folds = int( config['TRAIN']['nb_folds'] )                                       # number of folds
    num_epoch = int( config['TRAIN']['num_epoch'] )                                      # number of num_epoch 
    split_size = float( config['TRAIN']['split_size'] )                                    # train/val split
    batch_size= int( config['TRAIN']['batch_size'] )                                     # batch size
    num_workers= int( config['TRAIN']['num_workers'] )                                      # num workers
    loss_critereon = config['TRAIN']['loss_critereon']                    # loss criteria
    optimizer = config['TRAIN']['optimizer']                                  # optimizer to adjust thr Network weights
    lr = float( config['TRAIN']['lr'] )                                        # learning rate
    transfer_learning = config['TRAIN']['transfer_learning']                    # enable  transfer learning
    es_patience = int(num_epoch/10)+1                                     #This is required for early stopping, the number of num_epoch we will wait with no improvement before stopping

    # model instanciation 
    clf_model = get_model_instance(model_name, classes)
    print('###########################################################')
    print('#        Training the model %s  using the following parameters: '%(model_name))
    print('#  device=%s  ,  nb_folds=%d  ,  batch_size=%d  ,  num_epoch=%d  ,  num_workers=%d   '%(device, nb_folds, batch_size, num_epoch, num_workers ))
    print('#  loss_critereon=%s  ,  optimizer=%s  ,  lr=%d  ,  Early stop after=%d epochs'%(loss_critereon, optimizer, lr, es_patience ))
    print('#  Dataset=%s  \n#  model will be saved in =%s   '%(DIR_WORKSPACE, model_path ))
    print('###########################################################')

    # Run the training algorithm
    model, classes, epoch_nums, training_loss, validation_loss = train_model(DIR_TRAIN, clf_model, model_path, nb_folds=nb_folds, num_epoch=num_epoch, lr=lr, optimizer=optimizer,\
                                                                            loss_criteria=loss_critereon, split_size=split_size, batch_size=batch_size, es_patience=es_patience,\
                                                                            num_workers=num_workers, transfer_learning=transfer_learning)
    # Test the trained model
    truelabels, predictions, TS_sz = test_model(model_path, DIR_TEST, classes)
    # show performance
    classification_performance(classes, truelabels, predictions, TS_sz= TS_sz)#, TR_sz= len(train_loader.dataset) + len(val_loader.dataset))

    return model , classes


def main():
    parser = prepare_parser()
    args =  parser.parse_args()
    # Read config parameters from the sYAML file
    with open(args.cfg, 'r') as stream:
        config = yaml.safe_load(stream)

    # run the training
    run_model_training(config)

if __name__ == '__main__':
    main()
