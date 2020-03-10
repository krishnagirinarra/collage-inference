import torch
import torchvision.models as models
import torchvision.transforms as transforms
from skimage import io
import os
import pandas as pd
import numpy as np
from PIL import Image
import os.path
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import time
import json
from flask import Flask, jsonify, request
import socket
import struct
import fcntl
import requests

def task(filelist, pathin, pathout): 
    #device
    device = torch.device("cpu")
    #Load model
    model = models.resnet34(pretrained=True)
    model.eval()
    model.to(device)
    # Read input files
    composed = transforms.Compose([
               transforms.Resize(256, Image.ANTIALIAS),
               transforms.CenterCrop(224),
#               transforms.ToTensorCollage()])
               transforms.ToTensor()])
#               normalize])
    #reverse_map = TargetTransform()
    input_data = datasets.ImageFolder(root=pathin, transform=composed)
    input_loader = DataLoader(input_data, batch_size=1, shuffle=False)
    for bi, (input_batch, target) in enumerate(val_loader):
        img_tensor =  input_batch[0]
        output = model(img_tensor) 
        pred = torch.argmax(output, dim=1).cpu().numpy().tolist()
        ### Simulate slow downs
        #distrib = np.load('/home/collage_inference/resnet/latency_distribution.npy')
        # s = np.random.choice(distrib)
    if pred[0] == 1: # Say, class 1
        outfile = os.path.join(pathout, 'resnet1_')
    else:
        outfile = ""
    return [outfile]

def main():
    filelist = 'img1.jpeg'
    pathin = 'resnet1_in/'
    pathout = 'class_out/'
    outfile = task(filelist, pathin, pathout)
    return outfile
    

if __name__ == "__main__":
    filelist = 'img1.jpeg'
    pathin = 'resnet1_in/'
    pathout = 'class_out/'
    task(filelist, pathin, pathout)    
