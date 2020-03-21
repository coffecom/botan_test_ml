import os 
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import shutil

class Flattener(nn.Module):
    def forward(self, x):
        #print(x.shape)
        batch_size, *_ = x.shape
        return x.view(batch_size, -1)

glasses_classificator = nn.Sequential(
        nn.Conv2d(3,6,5),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
        nn.Conv2d(6,16,5),
        nn.ReLU(),
        nn.MaxPool2d(2,2),

        nn.Conv2d(16,32,5),
        nn.ReLU(),
        nn.MaxPool2d(2,2),

        Flattener(),

        nn.Dropout(),
        nn.Linear(512,120),
        nn.ReLU(),
        nn.Dropout(),

        nn.Linear(120,84),
        nn.ReLU(),
        nn.Linear(84,2)
)

if torch.cuda.is_available(): device = 'cuda'
else: device = 'cpu'

glasses_classificator.load_state_dict(torch.load('dropout_conv_train_lenet_100_eph.pt', map_location=device))
print("Enter folder path")

path = input()
#path = "C:\\Users\\Evelina\\JupyterNotebooks\\glasses\\for_testing"
if path[-1]!='/' or path[-1]!='\\' : path=path+'\\'

trans1 = transforms.ToPILImage()
trans2 = transforms.ToTensor()
image_size = 64

transform=transforms.Compose([
  transforms.Resize(image_size),
  transforms.CenterCrop(image_size),
  transforms.ToTensor()])
  
for filename in os.listdir(path):
	img = Image.open(path+filename)
	tensor = transform(img)
	tensor = tensor.unsqueeze(0)
	value, indices = torch.max(glasses_classificator(tensor), 1)
	pred = indices.item()
	if pred == 1:
		shutil.copy(path+filename, './')
		print("Picture",filename,"contains person with glasses")
