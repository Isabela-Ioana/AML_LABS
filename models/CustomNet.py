import torch
from torch import nn

# Define the custom neural network
class CustomNet(nn.Module):
    def __init__(self):     #in innit we declare the layers of nn
        super(CustomNet, self).__init__()

        #Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3= nn.Conv2d(128,256, kernel_size=3,padding=1)     #input=128 channels/ output=256 channels

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)       #this cuts the image resolution in 2


#After 3 convolutions+3 poolings, the tensor is [b,256,28,28]


        #Fully connected layers
        self.flatten = nn.Flatten()   #because fully connected layers cannot operate on 3d tensors,
                                      #flatten transforms them in 1d (vector)   b x 12983204129

        self.fc1 = nn.Linear(256 * 28 * 28, 512) # Intermediate layer =>
        self.fc2 = nn.Linear(512, 200)           # Final layer -> 200 classes TinyImageNet

# After the final layer, each image from the training batch gets a vector with 200 numbers.
#Each position in the vector is a class : poz0=cat, poz1=dog,...... and for each class, a logit
#which represents "how sure the network is about that class" (NOT PROBABILITIES!! ->can be negative, positive or 0)

    def forward(self, x):
        # We have [B,3 (rgb channels),244, 244]            244x244 is resolution of img
        # B x 3 x 224 x 224

        x = self.pool(self.conv1(x).relu()) # B x 64 x 112 x 112
        # 64 is the output of the first convolutional layer and 112x112 because of pooling
        x= self.pool(self.conv2(x).relu())       # B x 128 x 56 x 56
        x= self.pool(self.conv3(x).relu())       # B x 256 x 28 x 28

        # Flatten step:
        x= self.flatten(x)

        #Fully connected layers step:
        x= self.fc1(x).relu()       # B x 512
        x= self.fc2(x)              # B x 200     FINAL LAYER MUST BE CLEAR OF ACTIV FUNCTION


        return x