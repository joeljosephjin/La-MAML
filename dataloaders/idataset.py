
import numpy as np
from PIL import Image
import torch

# what is a transform??
from torchvision import datasets, transforms
import os

# ??
from dataloaders import cifar_info

# creating a custom dataset by subclassing and overriding the Dataset class in torch.utils.data
# has funcns for length and get_item
class DummyDataset(torch.utils.data.Dataset):

    # x- image, y- label, trsf= transform, pretrsf= pre-transform, super_y= ??
    def __init__(self, x, y, trsf, pretrsf = None, imgnet_like = False, super_y = None):
        self.x, self.y = x, y
        self.super_y = super_y

        # transforms to be applied before and after conversion to imgarray
        self.trsf = trsf
        self.pretrsf = pretrsf

        # imgnet comes as array
        # if not from imgnet, needs to be converted to imgarray first
        self.imgnet_like = imgnet_like

    # since images are square, it only needs the width i guess
    def __len__(self):
        return self.x.shape[0]

    # return x[idx], y[idx], super_y[idx] after converting to array and applying transforms
    def __getitem__(self, idx):
        # does the only step thats required i guess
        x, y = self.x[idx], self.y[idx]
        # if super_y has something, return its idx-th element too
        if self.super_y is not None: super_y = self.super_y[idx]

        # apply a pre-transform if necessary
        if(self.pretrsf is not None): x = self.pretrsf(x)    
        
        # convert to array if necessary (i.e. if its not imgnet-like)
        if(not self.imgnet_like): x = Image.fromarray(x)
        
        # apply a post-transform i guess
        x = self.trsf(x)

        # return {x, y, super_y(if it exists)} 
        if self.super_y is not None: return x, y, super_y
        else: return x, y

# creating a custom dataset AGAIN, but this time the most simple
# has funcns for length and get_item - extremely simplistic and straightforward
class DummyArrayDataset(torch.utils.data.Dataset):

    # no transforms or supers or array-izers - just simple
    def __init__(self, x, y):
        self.x, self.y = x, y

    # usual stuff, getting the width
    def __len__(self):
        return self.x.shape[0]

    # simple return the idx-th item. [no transforms, no array-izing or supers]
    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]

        return x, y

# return a collection of _get_datasets from the list
def _get_datasets(dataset_names):
    return [_get_dataset(dataset_name) for dataset_name in dataset_names.split("-")]

# the real deal of previous function
# returns dataset(object)s of {cifar10,cifar100,tinyimgnet}
def _get_dataset(dataset_name):
    # lower_case dataset_name and strip it of any spaces at the end or beginning
    dataset_name = dataset_name.lower().strip()

    # straightforward returning the appropriate dataset(object)
    if dataset_name == "cifar10": return iCIFAR10
    elif dataset_name == "cifar100": return iCIFAR100
    elif dataset_name == "tinyimagenet": return iImgnet
    else: raise NotImplementedError("Unknown dataset {}.".format(dataset_name))

# a class created for nothing ??
# contains {base_dataset, train_transforms, common_transforms, class_order}
class DataHandler:
    base_dataset = None
    train_transforms = []
    common_transforms = [transforms.ToTensor()]
    class_order = None


class iImgnet(DataHandler):

    base_dataset = datasets.ImageFolder

    top_transforms = [
        lambda x: Image.open(x[0]).convert('RGB'),
    ]

    train_transforms = [
        transforms.RandomCrop(64, padding=4),           
        transforms.RandomHorizontalFlip() #,
        #transforms.ColorJitter(brightness=63 / 255)
    ]

    common_transforms = [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))    
    ]

    class_order = [                                                                     
        i for i in range(200)
    ]

# CIFAR10 with only 10 classes
# subset of DataHandler
class iCIFAR10(DataHandler):
    # get an object
    base_dataset = datasets.cifar.CIFAR10
    # get the big damn class from 'cifar_info.py'
    base_dataset_hierarchy = cifar_info.CIFAR10

    # nothing
    top_transforms = [
    ]

    # crop, flip, colorjitting
    train_transforms = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255)
    ]

    # toTensor, normalize
    common_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]

# CIFAR100 with 100 classes
class iCIFAR100(iCIFAR10):
    # 
    base_dataset = datasets.cifar.CIFAR100
    base_dataset_hierarchy = cifar_info.CIFAR100

    # toTensor, normalize
    common_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]

    # update: class order can now be chosen randomly since it just depends on seed
    class_order = [
        87, 0, 52, 58, 44, 91, 68, 97, 51, 15, 94, 92, 10, 72, 49, 78, 61, 14, 8, 86, 84, 96, 18,
        24, 32, 45, 88, 11, 4, 67, 69, 66, 77, 47, 79, 93, 29, 50, 57, 83, 17, 81, 41, 12, 37, 59,
        25, 20, 80, 73, 1, 28, 6, 46, 62, 82, 53, 9, 31, 75, 38, 63, 33, 74, 27, 22, 36, 3, 16, 21,
        60, 19, 70, 90, 89, 43, 5, 42, 65, 76, 40, 30, 23, 85, 2, 95, 56, 48, 71, 64, 98, 13, 99, 7,
        34, 55, 54, 26, 35, 39
    ]   ## some random class order

    class_order_super = [4, 95, 55, 30, 72, 73, 1, 67, 32, 91, 62, 92, 70, 54, 82, 10, 61, 28, 9, 16, 53,
        83, 51, 0, 57, 87, 86, 40, 39, 22, 25, 5, 94, 84, 20, 18, 6, 7, 14, 24, 88, 97,
        3, 43, 42, 17, 37, 12, 68, 76, 71, 60, 33, 23, 49, 38, 21, 15, 31, 19, 75, 66, 34,
        63, 64, 45, 99, 26, 77, 79, 46, 98, 11, 2, 35, 93, 78, 44, 29, 27, 80, 65, 74, 50,
        36, 52, 96, 56, 47, 59, 90, 58, 48, 13, 8, 69, 81, 41, 89, 85
    ]   ## parent-wise split
