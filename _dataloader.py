from __future__ import print_function, division
import torch
import torch.utils.data
import torchvision
from torchvision import datasets, models, transforms
import os
from torch.utils.data import Dataset, DataLoader, BatchSampler
# from torchsampler import ImbalancedDatasetSampler
from skimage import io
import numpy as np
import pandas as pd
import math
#from utils import GaussianBlur
import PIL


# def data_transform(train_folder, val_folder, test_folder):
#     # VGG-16 Takes 224x224 images as input, so we resize all of them

#     image_size = 224
#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])

#     data_transforms = {
#         train_folder: transforms.Compose([
#             transforms.Resize(size=(image_size, image_size)),
#             transforms.RandomApply([transforms.RandomRotation(20)],p=.5),
#             transforms.RandomApply([transforms.ColorJitter(hue=.05, saturation=.05)], p=0.2),
#             transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.2),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             normalize,
#         ]),
#         val_folder: transforms.Compose([
#             transforms.Resize(size=(image_size, image_size)),
#             transforms.RandomApply([transforms.RandomRotation(20)],p=.5),
#             transforms.RandomApply([transforms.ColorJitter(hue=.05, saturation=.05)], p=0.2),
#             transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.2),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             normalize,
#         ]),
#         test_folder: transforms.Compose([
#             transforms.Resize(size=(image_size, image_size)),
#             # transforms.CenterCrop(image_size),
#             transforms.ToTensor(),
#             normalize,
#         ])
#     }
#     return data_transforms

def data_transform_test( test_folder):
    # VGG-16 Takes 224x224 images as input, so we resize all of them

    image_size = 224
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    data_transforms = {
        test_folder: transforms.Compose([
            transforms.Resize(size=(image_size, image_size)),
            # transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])
    }
    return data_transforms

# def data_csv(train_folder, val_folder, test_folder, csv_train, csv_val, csv_test):
#     csv = {
#         train_folder: csv_train,
#         val_folder: csv_val,
#         test_folder: csv_test
#     }
#     return csv

def data_csv_test( test_folder,  csv_test):
    csv = {
        test_folder: csv_test
    }
    return csv


#Classification dataloader : the name of the folder is the class name
# def cls_dataloader(data_dir, train_folder, val_folder, test_folder, batch_size):

#     data_transforms = data_transform(train_folder, val_folder, test_folder)

#     image_datasets = {
#         x: datasets.ImageFolder(
#             os.path.join(data_dir, x),
#             transform=data_transforms[x]
#         )
#         for x in [train_folder, val_folder, test_folder]
#     }

#     dataloaders = {
#         train_folder: DataLoader(
#             image_datasets[train_folder],
#             sampler=ImbalancedDatasetSampler(image_datasets[train_folder]),
#             batch_size=batch_size,
#             num_workers=0
#         ),
#         val_folder: DataLoader(
#             image_datasets[val_folder],
#             batch_size=batch_size,
#             num_workers=0
#         ),
#         test_folder: DataLoader(
#             image_datasets[test_folder],
#             batch_size=batch_size,
#             num_workers=0
#         )

#     }

#     dataset_sizes = {x: len(image_datasets[x]) for x in [train_folder, val_folder, test_folder]}

#     for x in [train_folder, val_folder, test_folder]:
#         print("Loaded {} images under {}".format(dataset_sizes[x], x))

#     print("Classes: ")
#     class_names = image_datasets[train_folder].classes
#     print(image_datasets[train_folder].classes)
#     return dataloaders, dataset_sizes, class_names #, train_folder, val_folder, test_folder

#regression dataloader
# def reg_dataloader(data_dir, train_folder, val_folder, test_folder, csv_train, csv_val, csv_test, batch_size, istrain):
#     """
#            Args:
#               istrain: is training or not
#            """

#     data_transforms = data_transform(train_folder, val_folder, test_folder)
#     csv_files = data_csv(train_folder, val_folder, test_folder, csv_train, csv_val, csv_test)


#     image_datasets = {
#         x: RegressionDataset(
#             csv_file=csv_files[x],
#             root_dir=os.path.join(data_dir, x),
#             transform=data_transforms[x]
#         )
#         for x in ([train_folder, val_folder] if istrain else [test_folder])
#     }

#     if istrain:
#         dataloaders = {
#             x: DataLoader(
#                 image_datasets[x],
#                 batch_size=batch_size,
#                 num_workers=16,
#                 pin_memory=True,
#                 # shuffle=True
#                 sampler = ImbalancedDatasetSampler(image_datasets[x], callback_get_label=reg_callback_get_label),
#             )
#             for x in [train_folder, val_folder]

#         }
#     else:

#         dataloaders = {
#             test_folder: DataLoader(
#                 image_datasets[test_folder],
#                 batch_size=batch_size,
#                 num_workers=0,
#                 pin_memory=True
#             )
#         }

#     dataset_sizes = {x: len(image_datasets[x]) for x in ([train_folder, val_folder] if istrain else [test_folder])}

#     for x in ([train_folder, val_folder] if istrain else [test_folder]):
#         print("Loaded {} images under {}".format(dataset_sizes[x], x))

#     return dataloaders, dataset_sizes


# regression dataloader for testing only
def reg_dataloader_test(data_dir, test_folder,  csv_test, batch_size):
    """
    Args:
        istrain: is training or not
    """

    data_transforms = data_transform_test(test_folder)
    csv_files = data_csv_test(test_folder,  csv_test)


    image_datasets = {
        x: RegressionDataset(
            csv_file=csv_files[x],
            root_dir=os.path.join(data_dir, x),
            transform=data_transforms[x]
        )
        for x in ([test_folder])
    }


    dataloaders = {
        test_folder: DataLoader(
            image_datasets[test_folder],
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True
        )
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ([test_folder])}

    for x in ([test_folder]):
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

        name = self.label_csv.iloc[idx, 0]
        if isinstance(name, int):
            name = str(name) +'.'+self.label_csv.iloc[idx, 1]
        elif isinstance(name, float):
            name = str(int(name) if int(name)== name  else name)
            name = name +'.'+self.label_csv.iloc[idx, 1]
        else:
            name = str(name) + '.'+self.label_csv.iloc[idx, 1]

        img_name = os.path.join(self.root_dir, name)

        image = io.imread(img_name)
        if len(image.shape)==3: #RGB
            image = transforms.ToPILImage()(image)
        else: #grayscale
            image = image.reshape(1,image.shape[0],image.shape[1])
            image = transforms.ToPILImage()(image)

        label = self.label_csv.iloc[idx, 2]
        # sample = {'image': image, 'label': label}

        if self.transform:
            # print(name)
            if image.mode == 'RGBA':
                image.load()
                background = PIL.Image.new("RGB", image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3])
                image = background
            image = self.transform(image)

        return name, image, label


def reg_callback_get_label(dataset, idx):
    return round(dataset[idx][2])