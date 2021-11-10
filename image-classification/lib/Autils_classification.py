import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# Import PyTorch libraries
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from lib.utils import *
######################## FUNCTIONS ######################################
 
def load_dataset(data_path, split_size=0.7, batch_size=50, num_workers=0):
    import torch
    import torchvision
    import torchvision.transforms as transforms
    # Load all the images
    transformation = transforms.Compose([
        # Randomly augment the image data
            # Random horizontal flip
        transforms.RandomHorizontalFlip(0.5),
            # Random vertical flip
        transforms.RandomVerticalFlip(0.3),
        # transform to tensors
        transforms.ToTensor(),
        # Normalize the pixel values (in R, G, and B channels)
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load all of the images, transforming them
    full_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=transformation
    )
    
    
    # Split into training (70% and testing (30%) datasets)
    train_size = int(split_size * len(full_dataset))
    test_size = len(full_dataset) - train_size
    
    # use torch.utils.data.random_split for training/test split
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
        
    # define a loader for the training data we can iterate through in 50-image batches
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )
    
    # define a loader for the testing data we can iterate through in 50-image batches
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )
        
    return train_loader, test_loader

def save_classes_json(classes):
  import json 
  data= {i:j for i,j in enumerate(classes)}
  with open('config/classes.json', 'w') as outfile:
    json.dump(data, outfile)

def train(model, device, train_loader, optimizer, loss_criteria):
    # Set the model to training mode
    model.train()
    train_loss = 0
    # Process the images in batches
    for batch_idx, (data, target) in enumerate(train_loader):

        # # display data specification
        # if batch_idx==1:
        #     batch_sample_details(data, target)
            
        # Recall that GPU is optimized for the operations we are dealing with
        data, target = data.to(device), target.to(device)
        
        # Reset the optimizer
        optimizer.zero_grad()
        
        # Push the data forward through the model layers
        output = model(data)
        
        # Get the loss
        loss = loss_criteria(output, target)

        # Keep a running total
        train_loss += loss.item()
        
        # Backpropagate
        loss.backward()
        optimizer.step()       
    # return average loss for the epoch
    avg_loss = train_loss / (batch_idx+1)
    return avg_loss

def test(model, device, loss_criteria, test_loader, display=True):
    # Switch the model to evaluation mode (so we don't backpropagate or drop)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        batch_count = 0
        for data, target in test_loader:
            # # display data specification
            # if batch_count==1:
            #     batch_sample_details(data, target)
            # load the batch
            batch_count += 1
            data, target = data.to(device), target.to(device)
            
            # Get the predicted classes for this batch
            output = model(data)
            
            # Calculate the loss for this batch
            test_loss += loss_criteria(output, target).item()
            
            # Calculate the accuracy for this batch
            _, predicted = torch.max(output.data, 1)
            correct += torch.sum(target==predicted).item()

    # Calculate the average loss and total accuracy for this epoch
    avg_loss = test_loss / batch_count
    if display:
      print('Testing set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
          avg_loss, correct, len(test_loader.dataset),
          100. * correct / len(test_loader.dataset)))
    
    # return average loss for the epoch
    return avg_loss

def define_config_optimizer_loss(transfer_learning, model_path, clf_model, lr, optimizer, loss_criteria):

      # Load the previously trained model
    if os.path.exists(model_path) and  transfer_learning == 'Yes' :
      print('\n - Load the pretained model!')
      model = torch.load(model_path)
    else:
      print('\n - Instanciate a new model structure!')
      # Create an instance of the model class and allocate it to the device
      model = clf_model
      transfer_learning = 'Yes'


    # define the optimizer 
    if optimizer == 'adam':
      optimizer_ = optim.Adam(model.parameters(), lr=lr)   # "Adam" optimizer to adjust weights
    else:
      print('Error: The used optimizer=%s is not found'%(optimizer) )
      return 0
    # define the optimizer 
    if loss_criteria == 'crossEntropy':
      loss_criteria_ =nn.CrossEntropyLoss()   # loss
    else:
      print('Error: The used optimizer=%s is not found'%(optimizer) )
      return 0

    return model, optimizer_, loss_criteria_, transfer_learning

def tracking_model_learning_history(model_path):
  
  model_vars = model_path.replace('.pth', '.pkl')
  if not os.path.exists(model_vars) : 
      print('yes')
      epoch_vect = []
      training_loss = []
      validation_loss = []
      old_epoch=0
  else:
      epoch_vect, training_loss, validation_loss, _, _ = load_variables(model_vars)
      old_epoch=len(epoch_vect)

  return old_epoch, epoch_vect, training_loss, validation_loss

def train_model(DIR_TRAIN, clf_model, model_path, nb_folds=5, num_epoch=100, lr=0.001, \
                  optimizer = 'adam', loss_criteria='crossEntropy', split_size=0.7, batch_size=50, \
                    es_patience=50, num_workers=0, transfer_learning='Yes'):
  # define the devide
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  print('device = ', device)
  create_new_folder(os.path.dirname(model_path))
  print("\n--> Training on the folder %s with %d folds \n --> #num_epoch = %d , batch size= %d "%(DIR_TRAIN, nb_folds,num_epoch, batch_size) )
  # Get/save class in json file for deployment 
  classes = sorted(os.listdir(DIR_TRAIN))
  save_classes_json(classes)
  print("--> Classes size = %d \n --> Classes names = %s"%(len(classes), classes ))
  # Start multiple folds training
  for fold in range(1,nb_folds):
    patience = es_patience
    # Get the iterative dataloaders for test and training data
    train_loader, val_loader = load_dataset(DIR_TRAIN, split_size=split_size, batch_size=batch_size, num_workers=num_workers)
    print('\n\n ################# Training fold%d #################'%(fold) )
    print("--> Training size = %d , Validation size = %d"%(len(train_loader.dataset), len(val_loader.dataset) ))
    # define the torch model
    model, optimizer_, loss_criteria_, transfer_learning = define_config_optimizer_loss(transfer_learning, model_path, clf_model, lr, optimizer, loss_criteria)
    # assign the model to the device type
    model.to(device) ; print(model)
    # Track metrics in these arrays
    old_epoch, epoch_vect, training_loss, validation_loss = tracking_model_learning_history(model_path)
    # Train the model 
    loss_min = np.inf
    print('Training on', device)
    for epoch in range(1, num_epoch + 1):
            train_loss = train(model, device, train_loader, optimizer_, loss_criteria_)
            test_loss = test(model, device, loss_criteria_, val_loader, display=False)
            epoch_vect.append(epoch+old_epoch)
            training_loss.append(train_loss)
            validation_loss.append(test_loss)
            if loss_min > validation_loss[-1]:
              # Save the model every epoch
              print(f"Saving a new optimal Model...")
              # torch.save(model.state_dict(), model_path)
              torch.save(model, model_path)
              save_variables(model_path[:-4] + '.pkl', [epoch_vect, training_loss, validation_loss, classes, device])
              loss_min = validation_loss[-1]
              patience = es_patience  # Resetting patience since we have new best validation accuracy
            else:
                patience -= 1
                if patience == 0:
                    print('Early stopping. Best Validation accuracy is: {:.3f}'.format(loss_min))
                    test_loss = test(model, device, loss_criteria_, val_loader)
                    break
            # display performace
            if epoch%10 == 0:
              print("Fold: %d/%d, Epoch: %d/%d"%(fold, nb_folds, epoch,num_epoch))
              print('Training set: Average loss: {:.6f}'.format(train_loss))
              test_loss = test(model, device, loss_criteria_, val_loader)
    # View Loss History
    plt.figure(figsize=(15,15))
    plt.plot(epoch_vect, training_loss)
    plt.plot(epoch_vect, validation_loss)
    plt.xlabel('epoch', fontsize = 15)
    plt.ylabel('loss', fontsize = 15)
    plt.legend(['training', 'validation'], loc='upper right')
    plt.show()
  return model, classes, epoch_vect, training_loss, validation_loss

def get_machine_ressources(model_path):
  # get the appropriate device
  device_model = model_path.split('_')[-1].split('.')[0]
  if   device_model =='cuda':
    if  torch.cuda.is_available():
      device = torch.device('cuda')
    else:
      print('Error: The model selected is compatible with CUDA GPU. No GPU was found!!!')
      return -1
  elif  device_model =='cpu': 
    device = torch.device('cpu')

  else: 
    print('Error: The model processor is not defined [' +  device_model + ']!!!')
    return -1

  return device

def test_model(model_path, DIR_TEST, classes):
  # get the appropriate device
  device = get_machine_ressources(model_path)
  if device == -1:
    return -1, -1, -1

  # load the model
  model_trained = torch.load(model_path)
  model_trained.eval()
  # classes_, model_ = load_variables(model_path[:-4] + '.pkl')

  # load data
  test_loader, _ = load_dataset(DIR_TEST, split_size=0.99)
  # Defining Labels and Predictions
  truelabels = []
  predictions = []
  print("\n--> Getting predictions using the testing set...")
  for data, target in test_loader:

      if device == torch.device('cuda'):
        data, target = data.cuda(), target.cuda()
        for label in target.cpu():
            truelabels.append(label)
        for prediction in model_trained(data).cpu().argmax(1):
            predictions.append(prediction) 
      else:
        for label in target.data.numpy():
            truelabels.append(label)
        for prediction in model_trained(data).data.numpy().argmax(1):
            predictions.append(prediction) 

  return truelabels, predictions, len(test_loader.dataset)

def classification_performance(classes, truelabels, predictions,  TS_sz=0, TR_sz=0):
  #Plot the confusion matrix
  from sklearn.metrics import confusion_matrix
  import seaborn as sns

  cm = confusion_matrix(truelabels, predictions)

  tick_marks = np.arange(len(classes))

  # Normalization
  cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
  cm=np.round(cm, 4)*100
  print("\n--> Normalized confusion matrix")
  print(cm)
  if cm.shape == (1, 1):
      v = cm[0, 0]
      cm = np.array([[v, 0], [0, v]])
  df_cm = pd.DataFrame(cm, index = classes, columns = classes)
  plt.figure(figsize = (7,7))
  sns.heatmap(df_cm, annot=True, cmap=plt.cm.Blues, fmt='g')
  plt.xlabel("Predicted class", fontsize = 10)
  plt.ylabel("True class", fontsize = 10)
  title_msg = ''
  if not TR_sz==0:
     title_msg = title_msg + " -  Training size = " + str(TR_sz ) 
  if not TS_sz==0:
     title_msg = title_msg + " -  Testing size = " + str(TS_sz ) 

  plt.title(title_msg, fontsize = 10) 
  plt.show()

def load_resize_convert_image(file_path, size):
  from PIL import Image
  img =  Image.open(file_path)  # Load image as PIL.Image
  if file_path[-4:]==".tif" or file_path[-5:]==".tiff" :
      img = img.point(lambda i:i*(1./256)).convert('L')  
      folder = str(os.path.join('data', 'workspace')); create_folder_set(folder)
      # Create a resized image
      img = resize_image(img, size)
  return img

def create_datafrane(file_paths, pred_classes):
  dict = {'file':file_paths,'prediction':pred_classes}
  return pd.DataFrame(dict)

def predict_image( model_path, classes, file_paths, size=(128,128), plotting=False):
  # get the appropriate device
  device = get_machine_ressources(model_path)
  if device == -1:
    return -1

  if isinstance(file_paths, str):
      file_paths=[file_paths]
  # Make a prediction
  model_trained = torch.load(model_path)
  model_trained.eval()  # Set to eval mode to change behavior of Dropout, BatchNorm

  transform = transforms.Compose([
      transforms.ToTensor(),
      # Normalize the pixel values (in R, G, and B channels)
      transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
  ])

  # run predictions
  pred_classes = []
  for file_path in file_paths:
    img =  load_resize_convert_image(file_path, size)
    print('image size =', img)
    x = transform(img)  # Preprocess image
    x = x.unsqueeze(0)  # Add batch dimension

    if device == torch.device('cuda'):
          x = x.cuda()
          # model_trained(data).cpu().argmax(1):
          output = model_trained(x).cpu()  # Forward pass
    else:
          output = model_trained(x)  # Forward pass
         
    pred = torch.argmax(output, 1)  # Get predicted class if multi-class classification
    predicted_label = pred[0].cpu().numpy()
    pred_classes.append(classes[ str( predicted_label ) ] )
    if plotting:
      plot_image(img, 'Predicted class = ' + classes[ str( predicted_label ) ] , filename=file_path)

    # create an pandas
  prediction_df = create_datafrane(file_paths, pred_classes)
  return prediction_df


