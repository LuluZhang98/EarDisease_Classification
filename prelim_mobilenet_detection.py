# -*- coding: utf-8 -*-
"""Prelim_MobileNet_Detection.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1zAdCF0p4wxEVN64bQsMyFQx4mwC9fziN
"""

#import libraries
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
from torchvision.models import mobilenet_v2
import torch.nn as nn

import os
import shutil


# Provide the path to the saved model file
MobileNet_model_path = './MobileNet_model.pth'   

class CustomMobileNetV2(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomMobileNetV2, self).__init__()
        self.model = mobilenet_v2(num_classes=num_classes)

    def forward(self, x):
        x = self.model(x)
        return x

# Instantiate the model
mobileNet_model = CustomMobileNetV2(num_classes=2)

mobileNet_model.load_state_dict(torch.load(MobileNet_model_path, map_location = 'cpu')) 

mobileNet_model.eval()

"""###**Single Image Prediction**"""

import matplotlib.pyplot as plt


def mobilenet_predict(img_path):
    # Load the image
    image = Image.open(img_path)

    # Define the transformations for inference
    transform = transforms.Compose([
        transforms.CenterCrop(1500),
        transforms.RandomRotation(180),
        transforms.RandomResizedCrop(224, (0.7, 1)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor()
    ])

    # Apply the same preprocessing steps as during training
    image_transformed = transform(image)

    # Add a batch dimension to the image tensor
    image_tensor = torch.unsqueeze(image_transformed, 0)

    # Forward pass through the model
    with torch.no_grad():
        output = mobileNet_model(image_tensor)

    # Apply softmax to the output tensor
    probabilities = F.softmax(output, dim=1)

    # Get the predicted class and its probability confidence
    predicted_prob, predicted_class = torch.max(probabilities, 1)

    # Convert the predicted class and its probability confidence to Python scalars
    predicted_prob = predicted_prob.item()
    predicted_class = predicted_class.item()

    # Define the class labels
    class_labels = ['NORMAL', 'Otitis Media']

    # Print the predicted class and its probability confidence with color
    predicted_text = '\033[94m{}\033[0m'.format(class_labels[predicted_class])
    confidence_text = '\033[94m{:.2f}%\033[0m'.format(predicted_prob * 100)
    # Print the predicted class and its probability confidence with color
    print('The Image is predicted as',predicted_text, 'with', confidence_text,'confidence.' ' Not for diagnosis, always consult a doctor')

    # # Display the original image
    # plt.imshow(image)
    # plt.axis('off')
    # plt.show()

    return class_labels[predicted_class], format(predicted_prob * 100, '.2f')