import numpy as np
import torch
import torchvision
import json
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch import nn
import argparse
from PIL import Image

parser=argparse.ArgumentParser(description='Training the model')

parser.add_argument('data_dir',type=str, help='Data Directory. Path to flower images',default="./flowers/")
parser.add_argument('--save_dir',type=str, help='Saving Directory',default="./checkpoint.pth")
parser.add_argument('--arch',type=str, help='Architecture model, vgg16 or densenet121',default="densenet121",  choices=['vgg16', 'densenet121'])
parser.add_argument('--learning_rate',type=float, help='Learning Rate, default =0.003',default=0.003)
parser.add_argument('--hidden_units',type=int, help='Hidden Units, default = 1000',default=1000)
parser.add_argument('--gpu',type=str, help='GPU',default="gpu")
parser.add_argument('--epochs',type=int, help='No. of epochs',default=7)

args=parser.parse_args()

data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
    
traindata_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                          ])
                   
validdata_transforms = transforms.Compose([transforms.Resize(255),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                              ])
                   
testdata_transforms = transforms.Compose([transforms.Resize(255),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                              ])
                  

trainimage_datasets = datasets.ImageFolder(train_dir, transform = traindata_transforms)
validimage_datasets = datasets.ImageFolder(valid_dir, transform = validdata_transforms)
testimage_datasets = datasets.ImageFolder(test_dir, transform = testdata_transforms)


traindataloaders = torch.utils.data.DataLoader(trainimage_datasets,batch_size=64, shuffle=True)
validdataloaders = torch.utils.data.DataLoader(validimage_datasets,batch_size=64, shuffle=True)
testdataloaders = torch.utils.data.DataLoader(testimage_datasets,batch_size=64, shuffle=True)

if (args.gpu==gpu):
    choice='cuda'
else:
    choice='cpu'

arch=args.arch
if(arch=="vgg16"):
    model = models.vgg16(pretrained=True)
else:
    model = models.densenet121(pretrained=True)

for params in model.parameters():
    params.requires_grad = False
    
classifier = nn.Sequential(
    nn.Linear(1024, 1000),
    nn.ReLU(),
    nn.Dropout(0.25),
    nn.Linear(1000, args.hidden_units),
    nn.ReLU(),
    nn.Linear(args.hidden_units, 102),
    nn.LogSoftmax(dim = 1),
)

model.classifier = classifier
criterion=nn.NLLLoss()
optimizer=optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

model.to(choice)
epochs = args.epochs
steps = 0
running_loss = 0
print_every = 25

for epoch in range(epochs):
    for inputs, labels in traindataloaders:
        steps += 1

        inputs, labels = inputs.to(choice), labels.to(choice)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            valid_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validdataloaders:
                    inputs, labels = inputs.to(choice), labels.to(choice)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    valid_loss += batch_loss.item()
                    
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Valid loss: {valid_loss/len(validdataloaders):.3f}.. "
                  f"Valid accuracy: {accuracy/len(validdataloaders):.3f}")
            running_loss = 0
            model.train()
            
accuracy =0
model.eval()
with torch.no_grad ():
    for inputs,labels in testdataloaders:
        inputs, labels = inputs.to(choice), labels.to(choice)
        outputs = model (inputs)
        ps = torch.exp(outputs).data
        equality = (labels.data == ps.max(1)[1])
        accuracy += equality.type_as(torch.FloatTensor()).mean()
        
print("Test Accuracy: {:.3f}".format(accuracy/len(testdataloaders)*100)+"%")    


model.class_to_idx = trainimage_datasets.class_to_idx

checkpoint = {
              'classifier':  model.classifier,
              'mapping': model.class_to_idx,
              'optimizer' : optimizer.state_dict(),
              'state_dict': model.state_dict()}

if (args.save_dir=='/checkpoint.pth'):
    torch.save (checkpoint, 'checkpoint.pth')
else:
    torch.save(checkpoint, args.save_dir+ '/checkpoint.pth')
