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

import argparse


def main():
    
    train_args = train_inputs()
    
    model, num_epochs, hidden_units, optimizer, train_dataset, arch = train_model(train_args.data_dir, 
                                                                                  train_args.hidden_units, 
                                                                                  train_args.arch, 
                                                                                  train_args.learning_rate, 
                                                                                  train_args.epochs, 
                                                                                  train_args.gpu)

    # Save checkpoint
    model.class_to_idx = train_dataset.class_to_idx
    checkpoint = save_checkpoint(arch, 
                                 model, 
                                 num_epochs, 
                                 hidden_units, 
                                 optimizer, 
                                 train_args.checkpoint_dir)
    
    

# Function to get the arguments from the Command Line
def train_inputs():
    """Command line arguments"""

    train_parser = argparse.ArgumentParser(add_help=True)

    train_parser.add_argument('--data_dir',
                              type=str, 
                              default='flowers',
                              help='Path to the flowers folder')

    train_parser.add_argument('--checkpoint_dir', 
                              type=str, 
                              default='checkpoint.pth',
                              help='Path where to save the checkpoint')

    train_parser.add_argument('--arch', 
                              type=str, 
                              default='vgg16', 
                              help='Model architecture: vgg16 or densenet121')

    train_parser.add_argument('--learning_rate', 
                              type=float, 
                              default=0.001, 
                              help='Learning rate')
    
    train_parser.add_argument('--hidden_units', 
                              type=int, 
                              default=4096, 
                              help='Number of hidden units')
    
    train_parser.add_argument('--epochs', 
                              type=int, 
                              default=10, 
                              help='Number of epochs')

    train_parser.add_argument('--gpu', 
                              action='store_true', 
                              default=True, 
                              help='Enable GPU')
    
    return train_parser.parse_args()



def train_model(data_dir, hidden_units, arch, lr, num_epochs, gpu):
    """ 1. Loading the data
        2. Building the model
        3. Training and validation 
        4. Testing (accuracy) """
    
    
    # --------------- Data Loading ---------------
    # Define your transforms for the training, validation, and testing sets
    data_transforms = {
        'train': transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])
                                    ]),
        'valid': transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])
                                    ]),
        'test': transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])
                                    ]),
         }
    
    # Directories
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    dirs = {'train': train_dir, 
            'valid': valid_dir, 
            'test': test_dir}
    
    # Load the datasets with ImageFolder
    image_datasets = {x: datasets.ImageFolder(dirs[x], transform=data_transforms[x]) 
                      for x in ['train', 'valid', 'test']}

    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True) 
                   for x in ['train', 'valid', 'test']}
    
    train_dataset = image_datasets['train']
    class_names = image_datasets['train'].classes
        
        
    # --------------- Building and training the model ---------------
    
    # Model definition
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        
        for param in model.parameters():
            param.requires_grad = False
    
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(25088, hidden_units)),
                                  ('relu', nn.ReLU()),
                                  ('fc2', nn.Linear(hidden_units, 102)),
                                  ('output', nn.LogSoftmax(dim=1))
                                  ]))
        model.classifier = classifier 

    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        hidden_units = 512
        
        for param in model.parameters():
            param.requires_grad = False
    
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(1024, hidden_units)),
                                  ('relu', nn.ReLU()),
                                  ('fc2', nn.Linear(hidden_units, 102)),
                                  ('output', nn.LogSoftmax(dim=1))
                                  ]))
        model.classifier = classifier 
        
    else:
        print("Error! {} is not one of the selected architectures".format(arch))
        
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(classifier.parameters(), lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

    since = time.time()
    
    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # Training on GPU or CPU
    if gpu and torch.cuda.is_available:
        model.cuda()
    else:
        model.cpu()
    
    # Training 
    print("Training:")
    print('-' * 10)
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print('-' * 10)
    
        # Setting the training and validation phases (for each epoch)
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set the model to training mode
            else:
                model.eval()   # Set the model to evaluation mode

            running_loss = 0.0
            running_corrects = 0.0

            # Iteration over data
            for inputs, labels in dataloaders[phase]:
                if gpu:
                    inputs, labels = inputs.cuda(), labels.cuda()

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase=='train'):  # Enable gradients just if in training phase
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)  
                    loss = criterion(outputs, labels)

                    # Backward (if training)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            # Deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())

        print()
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print("Best value of accuracy: {:.4f}".format(best_acc))
    
    # Load best model weights 
    model.load_state_dict(best_model_weights)  # Loads a model’s parameter dictionary using a deserialized state_dict
    

    # --------------- Testing ---------------
    print("\nTesting:")
    print('-' * 10)
    
    model.eval()  # Set to evaluation mode
    
    correct = 0
    total = 0

    with torch.no_grad():  # No grads for testing
        for inputs, labels in dataloaders['test']:
            if gpu:
                inputs, labels = inputs.cuda(), labels.cuda()
            data = inputs, labels

            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)  
            
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    
    print("Test accuracy: {}%".format(100*correct/total))
    
    return model, num_epochs, hidden_units, optimizer, train_dataset, arch



# Save checkpoint
def save_checkpoint(arch, model, num_epochs, hidden_units, optimizer, checkpoint_dir):
    torch.save({'architecture': arch,
                'model': model,
                'epochs': num_epochs,
                'hidden_units': hidden_units,
                'optimizer': optimizer,
                'state_dict': model.state_dict(), 
                'class_to_idx': model.class_to_idx},
                checkpoint_dir)
    print('Checkpoint saved!')
    
    
    
if __name__ == "__main__":
    main()