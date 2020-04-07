import torch
import torchvision.models as models
import torchvision.transforms as transforms
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import shutil

#class TargetTransform(object):
#    def __init__(self):
#        self.classes_list = np.load("./classes_list_103_classes.npy")
#        self.classes_list = np.sort(self.classes_list)
#    def __call__(self, target):
#        return self.classes_list[target]

def task(filelist, pathin, pathout): 
    ### set device to CPU
    device = torch.device("cpu")
    ### Load model
    model = models.resnet34(pretrained=True)
    model.eval()
    model.to(device)
    ### Transforms to be applied on input images
    composed = transforms.Compose([
               transforms.Resize(256, Image.ANTIALIAS),
               transforms.CenterCrop(224),
#               transforms.ToTensorCollage()])
               transforms.ToTensor()])
#               normalize])
#    reverse_map = TargetTransform()
#    input_data = datasets.ImageFolder(root=pathin, transform=composed, target_transform = reverse_map)
#    input_data = datasets.ImageFolder(root=pathin, transform=composed)
#    input_loader = DataLoader(input_data, batch_size=1, shuffle=False)
#    for bi, (input_batch, target) in enumerate(val_loader):
    out_list = []
    for f in filelist:
        ### Read input files.
        img = Image.open(os.path.join(pathin, f))
        ### Apply transforms.
       	img_tensor = composed(img)
        ### 3D -> 4D (batch dimension = 1)
        img_tensor.unsqueeze_(0) 
        #img_tensor =  input_batch[0]
        ### call the ResNet model
        output = model(img_tensor) 
        pred = torch.argmax(output, dim=1).numpy().tolist()
        ### To simulate slow downs
        #distrib = np.load('/home/collage_inference/resnet/latency_distribution.npy')
        # s = np.random.choice(distrib)
        ### Copy to appropriate destination paths
        if pred[0] == 555: ### fire engine. class 1
            source = os.path.join(pathin, f)
            destination = os.path.join(pathout, '1', f)
            out_list.append(shutil.copyfile(source, destination))
        elif pred[0] == 779: ### school bus. class 2
            source = os.path.join(pathin, f)
            destination = os.path.join(pathout, '2', f)
            out_list.append(shutil.copyfile(source, destination))
        else: ### not either of the classes
            source = os.path.join(pathin, f)
            destination = os.path.join(pathout, 'none', f)
            out_list.append(shutil.copyfile(source, destination))
    return out_list

if __name__ == "__main__":
    filelist = ['n03345487_1002.JPEG', 'n04146614_10015.JPEG']
    pathin = './to_resnet/'
    pathout = './classes/'
    task(filelist, pathin, pathout)    
