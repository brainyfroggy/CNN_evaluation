from __future__ import print_function, division
import torch
from torchvision import datasets, models, transforms
import os
from torch.utils.data import Dataset, DataLoader
from torchsampler import ImbalancedDatasetSampler
from skimage import io
import numpy as np
import pandas as pd



def data_transform(train_folder, val_folder, test_folder):
    # VGG-16 Takes 224x224 images as input, so we resize all of them
    data_transforms = {
        train_folder: transforms.Compose([
            # Data augmentation is a good practice for the train set
            # Here, we randomly crop the image to 224x224 and
            # randomly flip it horizontally.
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]),
        val_folder: transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]),
        test_folder: transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
    }
    return data_transforms


def data_csv(train_folder, val_folder, test_folder, csv_train, csv_val, csv_test):
    csv = {
        train_folder: csv_train,
        val_folder: csv_val,
        test_folder: csv_test
    }
    return csv

#Classification dataloader : the name of the folder is the class name
def cls_dataloader(data_dir, train_folder, val_folder, test_folder, batch_size):

    data_transforms = data_transform(train_folder, val_folder, test_folder)

    image_datasets = {
        x: datasets.ImageFolder(
            os.path.join(data_dir, x),
            transform=data_transforms[x]
        )
        for x in [train_folder, val_folder, test_folder]
    }

    dataloaders = {
        train_folder: DataLoader(
            image_datasets[train_folder],
            sampler=ImbalancedDatasetSampler(image_datasets[train_folder]),
            batch_size=batch_size,
            num_workers=0
        ),
        val_folder: DataLoader(
            image_datasets[val_folder],
            batch_size=batch_size,
            num_workers=0
        ),
        test_folder: DataLoader(
            image_datasets[test_folder],
            batch_size=batch_size,
            num_workers=0
        )

    }

    dataset_sizes = {x: len(image_datasets[x]) for x in [train_folder, val_folder, test_folder]}

    for x in [train_folder, val_folder, test_folder]:
        print("Loaded {} images under {}".format(dataset_sizes[x], x))

    print("Classes: ")
    class_names = image_datasets[train_folder].classes
    print(image_datasets[train_folder].classes)
    return dataloaders, dataset_sizes, class_names #, train_folder, val_folder, test_folder

#regression dataloader
def reg_dataloader(data_dir, train_folder, val_folder, test_folder, csv_train, csv_val, csv_test, batch_size):

    data_transforms = data_transform(train_folder, val_folder, test_folder)
    csv_files = data_csv(train_folder, val_folder, test_folder, csv_train, csv_val, csv_test)


    image_datasets = {
        x: RegressionDataset(
            csv_file=csv_files[x],
            root_dir=os.path.join(data_dir, x),
            transform=data_transforms[x]
        )
        for x in [train_folder, val_folder, test_folder]
    }


    dataloaders = {
        train_folder: DataLoader(
            image_datasets[train_folder],
            #sampler=ImbalancedDatasetSampler(image_datasets[train_folder]),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        ),
        val_folder: DataLoader(
            image_datasets[val_folder],
            batch_size=batch_size,
            num_workers=0
        ),
        test_folder: DataLoader(
            image_datasets[test_folder],
            batch_size=batch_size,
            num_workers=0
        )

    }

    dataset_sizes = {x: len(image_datasets[x]) for x in [train_folder, val_folder, test_folder]}

    for x in [train_folder, val_folder, test_folder]:
        print("Loaded {} images under {}".format(dataset_sizes[x], x))

    return dataloaders, dataset_sizes


#The label is read from csv
class RegressionDataset(Dataset):
    """dataset only used for Regression ."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.label_csv = pd.read_csv(csv_file, header = None)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.label_csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                str(int(self.label_csv.iloc[idx, 0])
                                    if int(self.label_csv.iloc[idx, 0])== self.label_csv.iloc[idx, 0]
                                    else self.label_csv.iloc[idx, 0]) +'.'+self.label_csv.iloc[idx, 1])
        image = io.imread(img_name)
        if len(image.shape)==3: #RGB
            image = transforms.ToPILImage()(image)
        else: #grayscale
            image = image.reshape(1,image.shape[0],image.shape[1])
            image = transforms.ToPILImage()(image)

        label = self.label_csv.iloc[idx, 3]
        # sample = {'image': image, 'label': label}

        if self.transform:
            image = self.transform(image)

        return image, label
