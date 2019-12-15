import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
from PIL import Image
import seaborn as sns

# PyTorch packages
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models

import json
import time
import copy

import train
import argparse


def main():
    
    predict_args = predict_inputs()  # Getting the inputs
    
    model = load_model(predict_args.checkpoint, predict_args.gpu)  # Loading the checkpoint
    
    image = process_image(predict_args.input)   # Numpy image
        
    top_probs_list, top_classes_list = predict(image, model, predict_args.top_k, predict_args.gpu) 
    
    with open(predict_args.category_names, 'r') as f:
        cat_to_name = json.load(f)
        
    idx_to_classes = {key: val for val, key in model.class_to_idx.items()}
    
    top_labels = [idx_to_classes[label] for label in top_classes_list]
    top_names = [cat_to_name[idx_to_classes[label]] for label in top_classes_list]
    
    # Dictionary containing the names of the top K flowers and their correspondent probabilities
    results = {top_names[x]: top_probs_list[x] for x in range(predict_args.top_k - 1)}
    
    print(results)
          
    
# Function to get the parameters from the Command Line 
def predict_inputs():
    """Command line arguments"""

    predict_parser = argparse.ArgumentParser(add_help=True)
    
    predict_parser.add_argument('--input', 
                                type=str, 
                                default='flowers/test/15/image_06351.jpg', 
                                help='Image we want to predict')
    
    predict_parser.add_argument('--checkpoint', 
                                type=str, 
                                default='checkpoint.pth', 
                                help='Path to the checkpoint')
    
    predict_parser.add_argument('--top_k', 
                                type=int, 
                                default=5, 
                                help='top k classes')
    
    predict_parser.add_argument('--category_names', 
                                type=str, 
                                default='cat_to_name.json', 
                                help='JSON labels files')
    
    predict_parser.add_argument('--gpu', 
                                action='store_true',
                                default=True, 
                                help='Enable GPU')
    
    return predict_parser.parse_args()
        
        

def load_model(checkpoint_path, gpu):
    
    if gpu == False:
        # CPU
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    else:
        # GPU 
        checkpoint = torch.load(checkpoint_path)
    
    arch = checkpoint['architecture']
    model = checkpoint['model']
    # Create the classifier
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, 4096)),
                                            ('relu', nn.ReLU()),
                                            ('fc2', nn.Linear(4096, 102)),
                                            ('output', nn.LogSoftmax(dim=1))
                                          ])) 
    model.classifier = classifier  # Put the classifier on the pre-trained model
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

        
                
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a Numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model
    image = Image.open(image_path)
    
    # Resize
    if image.size[0] > image.size[1]:
        image.thumbnail((10000, 256))
    else:
        image.thumbnail((256, 10000))
        
    # Crop     
    left_margin = (image.width - 224)//2
    bottom_margin = (image.height - 224)//2
    right_margin = (image.width + 224)//2
    top_margin = (image.height + 224)//2
    
    image = image.crop((left_margin,  
                        bottom_margin, 
                        right_margin,  
                        top_margin))   
    
    # Normalization
    image = np.array(image)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    image = (image - mean)/std
    
    # Color channels to first dimension (PyTorch requisite)
    image = image.transpose((2, 0, 1))
    
    return image

        

def predict(image, model, top_k, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model. ''' 
  
    if gpu:
        model.cuda()
    else:
        model.cpu()
        
    model.eval()
    
    tensor = torch.from_numpy(image).type(torch.FloatTensor) # We have to change the image from Numpy to PyTorch
    input_image = tensor.unsqueeze_(0)  # Add batch of size 1 to image, in order to fit it in the model
    
    if gpu:
        input_image = input_image.cuda()
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print('Device: ', device)
    # input_image  = input_image.to(device)
    
    logps = model.forward(input_image)
    ps = torch.exp(logps)
    
    # Predict top 5 probabilities
    top_probs, top_classes = ps.topk(top_k)
    
    if gpu and torch.cuda.is_available():
        top_probs = top_probs.data.cpu().numpy()
        top_classes = top_classes.data.cpu().numpy()
    
        top_probs_list = top_probs[0].tolist()
        top_classes_list = top_classes[0].tolist()
    else:
        top_probs = top_probs.data.numpy()
        top_classes = top_classes.data.numpy()
    
        top_probs_list = top_probs[0].tolist()
        top_classes_list = top_classes[0].tolist()
               
    return top_probs_list, top_classes_list    


if __name__ == "__main__":
    main() 