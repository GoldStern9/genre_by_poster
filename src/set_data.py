import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class ImageDataset(Dataset):
    def __init__(self, csv, train_sz):
        self.csv = csv
        self.all_image_names = self.csv[:]['Id']
        self.labels = np.array(self.csv.drop(['Id', 'Genre'], axis=1))
        # split the data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            self.all_image_names, self.labels, test_size= (1-train_sz), random_state=17)
        
        print(f"Number of training images: {int(train_sz * len(self.csv))}")
        # set the training data images and labels
        self.train_images = list(X_train) 
        self.train_labels = list(y_train) 
        # define the training transforms
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((400, 400)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=45),
            transforms.ToTensor(),
        ])
        
        print(f"Number of validation images: {len(self.csv) - (int(train_sz * len(self.csv)))}")
        # set the validation data images and labels
        self.val_images = list(X_val)
        self.val_labels = list(y_val) 
        # define the validation transforms
        self.val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((400, 400)),
            transforms.ToTensor(),
        ])
        
        # set the test data images and labels, only last 10 images
        self.test_images = list(X_val[-10:])
        self.tets_labels = list(y_val[-10:])
        # define the test transforms
        self.test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        out = [len(self.train_images), len(self.val_images), len(self.test_images)]
        return out
    
    def __gettrain__(self, index):
        image = cv2.imread(f"../data/Images/{self.train_images[index]}.jpg")
        # convert the image from BGR to RGB color format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # apply image transforms
        image = self.train_transform(image)
        targets = self.train_labels[index]

        return {
            'image': torch.tensor(image, dtype=torch.float32),
            'label': torch.tensor(targets, dtype=torch.float32)
        }

    def __getval__(self, index):
        image = cv2.imread(f"../data/Images/{self.val_images[index]}.jpg")
        # convert the image from BGR to RGB color format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # apply image transforms
        image = self.val_transform(image)
        targets = self.val_labels[index]

        return {
            'image': torch.tensor(image, dtype=torch.float32),
            'label': torch.tensor(targets, dtype=torch.float32)
        }    

    def __gettest__(self, index):
        image = cv2.imread(f"../data/Images/{self.test_images[index]}.jpg")
        # convert the image from BGR to RGB color format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # apply image transforms
        image = self.test_transform(image)
        targets = self.test_labels[index]

        return {
            'image': torch.tensor(image, dtype=torch.float32),
            'label': torch.tensor(targets, dtype=torch.float32)
        }