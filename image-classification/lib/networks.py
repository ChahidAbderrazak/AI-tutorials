
# define the model instanciation
def get_model_instance(model_name, classes):
    try : 
        if model_name == 'CNN-MClss':
            clf_model = Net(num_classes=len(classes))

        return clf_model 

    except:
        msg = "The instanciation  of model architecture named <" +  model_name + ">  is not found"
        raise ModelInstanceError(msg)

# Import PyTorch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
  
class ModelInstanceError(Exception):
    pass

class Net(nn.Module):
    # source: https://www.kaggle.com/yomnamabdulwahab/imagenet-pytorch
    def __init__(self, num_classes=3):
        super(Net, self).__init__()
        # We will apply 12 filters in the first convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        # A second convolutional layer takes 12 input channels, and generates 24 outputs
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        # We in the end apply max pooling with a kernel size of 2
        self.pool = nn.MaxPool2d(kernel_size=2)
        # A drop layer deletes 20% of the features to help prevent overfitting
        self.drop = nn.Dropout2d(p=0.2)
        # Our 128x128 image tensors will be pooled twice with a kernel size of 2. 128/2/2 is 32.
        # This means that our feature tensors are now 32 x 32, and we've generated 24 of them
        # We need to flatten these in order to feed them to a fully-connected layer
        self.fc = nn.Linear(in_features=32 * 32 * 24, out_features=num_classes)

    def forward(self, x):
        # In the forward function, pass the data through the layers we defined in the init function
        # Use a ReLU activation function after layer 1 (convolution 1 and pool)
        x = F.relu(self.pool(self.conv1(x))) 
        # Use a ReLU activation function after layer 2
        x = F.relu(self.pool(self.conv2(x)))  
        # Select some features to drop to prevent overfitting (only drop during training)
        x = F.dropout(self.drop(x), training=self.training)
        # Flatten
        x = x.view(-1, 32 * 32 * 24)
        # Feed to fully-connected layer to predict class
        x = self.fc(x)
        # Return class probabilities via a log_softmax function 
        return torch.log_softmax(x, dim=1)
    