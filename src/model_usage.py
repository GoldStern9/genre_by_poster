import model_engine
import torch
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from set_data import ImageDataset
from torch.utils.data import DataLoader

genres = ['Action', 'Adventure', 'Animation', 'Biography',
       'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy',
       'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News', 'Romance',
       'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War', 'Western']


def predict_by_sample(model_path, img_path):
    # initialize the computation device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #intialize the model
    model = model_engine.model(pretrained=False, requires_grad=False).to(device)
    # load the model checkpoint
    #checkpoint = torch.load('../models/model.pth')
    checkpoint = torch.load(model_path)
    # load model weights state_dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ])
    
    image = cv2.imread(img_path)
    # convert the image from BGR to RGB color format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # apply image transforms
    image = transform(image)
    image = torch.tensor(image, dtype=torch.float32)
    image = image.to(device)
    
    outputs = model(image[None, ...])
    outputs = torch.sigmoid(outputs)
    outputs = outputs.detach().cpu()
    sorted_indices = np.argsort(outputs[0])
    top3 = sorted_indices[-3:]
    string_predicted = ''
    for i in range(len(top3)):
        string_predicted += f"{genres[top3[i]]}    "
    return string_predicted
         

def predict_by_test(data_path, model_path):
    # initialize the computation device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #intialize the model
    model = model_engine.model(pretrained=False, requires_grad=False).to(device)
    # load the model checkpoint
    #checkpoint = torch.load('../models/model.pth')
    checkpoint = torch.load(model_path)
    # load model weights state_dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    #train_csv = pd.read_csv('../data/preproc_res/prep_train0.csv')
    train_csv = pd.read_csv(data_path)
    
    #genres = train_csv.columns.values[2:]
    # prepare the test dataset and dataloader
    test_data = ImageDataset(
        train_csv, train=False, test=True
    )
    test_loader = DataLoader(
        test_data, 
        batch_size=1,
        shuffle=False
    )

    for counter, data in enumerate(test_loader):
        image, target = data['image'].to(device), data['label']
        # get all the index positions where value == 1
        target_indices = [i for i in range(len(target[0])) if target[0][i] == 1]
        # get the predictions by passing the image through the model
        outputs = model(image)
        outputs = torch.sigmoid(outputs)
        outputs = outputs.detach().cpu()
        sorted_indices = np.argsort(outputs[0])
        top3 = sorted_indices[-3:]
        string_predicted, string_actual = '', ''
        for i in range(len(top3)):
            string_predicted += f"{genres[top3[i]]}    "
        for i in range(len(target_indices)):
            string_actual += f"{genres[target_indices[i]]}    "
        image = image.squeeze(0)
        image = image.detach().cpu().numpy()
        image = np.transpose(image, (1, 2, 0))
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"ACTUAL: {string_actual}\nPREDICTED: {string_predicted}")
        plt.savefig(f"../visualisation/test_genres_vis/predictions_{counter}.jpg")
        plt.show()

print(predict_by_sample("../models/model.pth", "../data/23.jpg"))
#predict_by_test('../data/preproc_res/prep_train4.csv',"../models/model.pth")