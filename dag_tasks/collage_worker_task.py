import torch
import torchvision.transforms as transforms
import numpy as np
import os.path
import math

from torchvision import datasets
from torch.utils.data import Dataset, DataLoader

def calculate_iou(L1, R1, T1, B1, L2, R2, T2, B2):
    L = max(L1, L2)
    R = min(R1, R2)    
    T = max(T1, T2)
    B = min(B1, B2)
    i = max(0, R-L+1) * max(0, B-T+1) #max(0,) covers the case where no intersection exists. Why +1 I am not sure
    A1 = (R1-L1+1) * (B1-T1+1)
    A2 = (R2-L2+1) * (B2-T2+1)
    iou = i*1.0/(A1 + A2 - i) 
    return iou
	
	
def calculate_pos(left, right, top, bottom, w, spatial):
    # return box in pos with maximum iou.
    pos_dict = {}
    for i in range(0,w):
        pos_dict[i] = [0]*w
        for j in range(0,w):
            pos_dict[i][j] = i + w*j
    max_iou = 0.0
    for i in range(w):
        t = i*spatial
        b = ((i+1)*spatial) - 1
        for j in range(w):
             l = j*spatial
             r = ((j+1)*spatial) - 1
             temp = calculate_iou(left, right, top, bottom, l, r, t, b)   
             if max_iou < temp:
                 max_iou = temp
                 x = j
                 y = i 
    #print(left, right, top, bottom, x, y)
    return (pos_dict[x])[y]
	
def process_collage(pred):
    #print("pred1: ", pred.shape)
    pred = pred[pred[:, :, 4] > conf_thres]
    #print("pred2: ", pred.shape)
    if len(pred) > 0:
        detections_list = non_max_suppression(pred.unsqueeze(0), conf_thres, nms_thres)
        detections = detections_list[0]
        #print(len(detections_list)) 
        #print(detections.shape) 
    # Draw bounding boxes and labels of detections
    if detections is not None:
        # write results to .txt file
        #results_txt_path = output_dir + "/" + 'collage_preds.txt'
        #if os.path.isfile(results_txt_path):
        #    os.remove(results_txt_path)
        #print("detections info as follows:")
        #print(detections)
        predictions_prob_list = [-1]*w*w
        predictions_list = [-1]*w*w
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            #if save_txt:
            #    with open(results_txt_path, 'a') as file:
            #        file.write(('%g %g %g %g %g %g \n') % (x1, y1, x2, y2, cls_pred, cls_conf * conf))
            pos = calculate_pos(x1, x2, y1, y2, w, single_spatial)
            prob = cls_conf * conf
            cls_pred = int(cls_pred.item())
            #print("position and probability are: ", pos, prob)
            if predictions_prob_list[pos] != -1:
                if predictions_prob_list[pos] < prob:
                    predictions_list[pos] = cls_pred
            else:
                predictions_list[pos] = cls_pred
                predictions_prob_list[pos] = prob
    #print("predictions_list is: ", predictions_list)
    #predictions_arr = np.array(predictions_list)
    predictions_list = classes_list[predictions_list].tolist()
    return predictions_list
	
	
def task(filelist, pathin, pathout):
	device = torch.device("cpu")
	
	# Load collage model
	# model = .....
	# model.to(device)
	
	input_data = datasets.ImageFolder(root=pathin+filelist[0]) # Not sure what the transform to be applied to the collage image is
	input_loader = DataLoader(input_data, batch_size=1, shuffle=False)
	
	with torch.no_grad():
		for i, (image, label) in enumerate(input_loader):
			out = model(image)
		
		final_preds = process_collage(out) # List of predictions_arr
		
	
	outfiles = []
	for pred in final_preds:
		path = path + '/pred_' +  str(pred)
		outfiles.append(path)
	
	return outfiles
		
	
	
	
	
	
	