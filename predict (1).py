import numpy as np
import torch
import torchvision
import json
import argparse
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch import nn
from PIL import Image

parser=argparse.ArgumentParser(description='Prediction')

parser.add_argument('--input',type=str, help='Data Directory. Path to flower images',default='./flowers/test/14/image_06083.jpg')
parser.add_argument('--checkpoint',type=str, help='Checkpoint',default="./checkpoint.pth")
parser.add_argument('--top_k',type=int, help='Top n images, default value of n=5',default=5)
parser.add_argument('--arch',type=str, help='Architecture model, vgg16 or densenet121',default="densenet121",  choices=['vgg16', 'densenet121'])
parser.add_argument('--category_names',type=str, help='Category names',default='cat_to_name.json')
parser.add_argument('--gpu',type=str, help='GPU',default="gpu")

args=parser.parse_args()

if (args.gpu=='gpu'):
    choice='cuda'
else:
    choice='cpu'

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    if(args.arch=="vgg16"):
        model = models.vgg16(pretrained=True)
    else:
        model = models.densenet121(pretrained=True)
    model.classifier = checkpoint ['classifier']
    model.class_to_idx = checkpoint ['mapping']
    model.load_state_dict (checkpoint ['state_dict'])
    
    for param in model.parameters(): 
        param.requires_grad = False    
    
    return model

def process_image(image_path):
    image = Image.open(image_path)
        
    transformations = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return transformations(image).numpy()

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    image = std * image + mean    
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, top_k=args.top_k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    if(choice=='gpu'):
        model.cuda()
    else:
        model.cpu()
    # TODO: Implement the code to predict the class from an image file
    img = process_image(image_path)
    img = torch.from_numpy(img).unsqueeze_(0)
   
    
    with torch.no_grad():
        model.eval()
        if(choice=='gpu'):
            output = model.forward(img.cuda())
        else:
            output = model.forward(img.cpu())
        
    ps = F.softmax(output.data,dim=1)
    x=ps.topk(top_k)
     
    j ={ val: key for key, val in model.class_to_idx.items()}
    probs = x[0].tolist()[0]
    classes = [j[i] for i in x[1].tolist()[0]]
    
    return probs, classes



import json

with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)

model=load_checkpoint(args.checkpoint)
im_tensor = torch.from_numpy(process_image(args.input))
#imshow(im_tensor)
#plt.show()
image_path = args.input
probs, classes = predict(image_path, model)
class_name = [cat_to_name[i] for i in classes]
print (probs)
print(class_name[0])
#fig, axs = plt.subplots(figsize=(10, 8), nrows=2)
#fig.suptitle(class_name[0])

#axs[0].set_aspect(0.1);
#axs[0].barh(class_name, probs);

#axs[1].axis('off')
#plt.show()