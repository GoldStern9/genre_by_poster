import torch
from tqdm import tqdm
import numpy as np
import model_engine
import math as mt
import pandas as pd
from set_data import ImageDataset
from torch.utils.data import DataLoader

def check_accuracy(loader, model, device, th, vdata) -> float:
    num_correct = 0
    num_samples = 0
    model.eval()
    
    with torch.no_grad():
        #for counter, data in enumerate(loader):
        for i, data in tqdm(enumerate(loader), total=int(len(vdata)/loader.batch_size)):
            x, y = data['image'].to(device), data['label'].to(device)
            scores = model(x)
            scores = torch.sigmoid(scores)
            scores = scores.detach().cpu()
            scores = scores[0] > th #treshhold
            num_correct += (scores == y).sum()
            num_samples += scores.size(0)
        
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}') 
    return (float(num_correct)/float(num_samples)*100)

def check_precision(loader, model, device, th, vdata) -> float:
    tp = 0 #True Positives
    fp = 0 #False Positives
    model.eval()
    
    with torch.no_grad():
        #for counter, data in enumerate(loader):
        for i, data in tqdm(enumerate(loader), total=int(len(vdata)/loader.batch_size)):
            x, y = data['image'].to(device), data['label'].to(device)
            scores = model(x)
            scores = torch.sigmoid(scores)
            scores = scores.detach().cpu()
            scores = scores[0] > th #treshhold
            """f.e, we have predictions [1,0,1,1,0] and 
            actual labels [1,1,1,0,0], then if we summarize them we get:
            [2,1,2,1,0]; 2-TP, 0-TN, 1-FP or FN
            If we subtract them: [0,-1,0,1,0]; 1-FP, 0-TP or TN, -1-FN
            """  
            tp += int(((scores + y) == 2).sum())
            fp += int(((scores.long() - y) == 1).sum())

        p = tp/(tp+fp)
        print(f'Got TP {tp} and FP {fp} values with precision: {p:.2f}')
    return p       
    
def check_recall(loader, model, device, th, vdata) -> float:
    tp = 0 #True Positives
    fn = 0 #False Negatives
    model.eval()
    
    with torch.no_grad():
        #for counter, data in enumerate(loader):
        for i, data in tqdm(enumerate(loader), total=int(len(vdata)/loader.batch_size)):
            x, y = data['image'].to(device), data['label'].to(device)
            scores = model(x)
            scores = torch.sigmoid(scores)
            scores = scores.detach().cpu()
            scores = scores[0] > th #treshhold
            """f.e, we have predictions [1,0,1,1,0] and 
            actual labels [1,1,1,0,0], then if we summarize them we get:
            [2,1,2,1,0]; 2-TP, 0-TN, 1-FP or FN
            If we subtract them: [0,-1,0,1,0]; 1-FP, 0-TP or TN, -1-FN
            """  
            tp += int(((scores + y) == 2).sum())
            fn += int(((scores.long() - y) == -1).sum())

        r = tp/(tp+fn)
        print(f'Got TP {tp} and FN {fn} values with recall: {r:.2f}')
    return r       

def check_f1(loader, model, device, th, vdata, p, r) -> float:
    f1 = 2*p*r/(p+r)
    print(f'F1: {f1:.2f}')
    return f1

# read the training csv file
train_csv = pd.read_csv('../data/preproc_res/prep_train4.csv')
# validation dataset
valid_data = ImageDataset(
    train_csv, train=False, test=False
)
# validation data loader
valid_loader = DataLoader(
    valid_data, 
    batch_size=1,
    shuffle=False
)
"""test_data = ImageDataset(
    train_csv, train=False, test=True
    )
test_loader = DataLoader(
    test_data, 
    batch_size=1,
    shuffle=False
    )
"""
# initialize the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#intialize the model
model = model_engine.model(pretrained=False, requires_grad=False).to(device)
# load the model checkpoint
#checkpoint = torch.load('../models/model.pth')
checkpoint = torch.load('../models/model.pth')
# load model weights state_dict
model.load_state_dict(checkpoint['model_state_dict'])

accur = check_accuracy(valid_loader, model, device, 0.3, valid_data)
p = check_precision(valid_loader, model, device, 0.3, valid_data)
r = check_recall(valid_loader, model, device, 0.3, valid_data)
f1 = check_f1(valid_loader, model, device, 0.3, valid_data, p, r)
