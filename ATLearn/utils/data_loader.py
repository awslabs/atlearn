# Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
# https://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

import torch
from torchvision import datasets, transforms
import numpy as np
from pathlib import Path
import glob
import os
from PIL import Image
from .augmentation import RandAugmentMC

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')


def make_dataset(path):
    f = []
    p = Path(path)
    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
    img_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)

    return get_score(img_files)


def get_score(files):
    output = []
    for p in files:
        start = p.find('banana_')
        end = p.find('.jpg')
        score = float(p[start+7:end])
        output.append((p, score))
    return output


class ImageList(torch.utils.data.Dataset):
    def __init__(self, path, transform=None, target_transform=None, mode='RGB'):
        imgs = make_dataset(path)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def read_data_from_folder_regression(data, batch_size, mode='train'):
    transform = {"train": transforms.Compose([
        transforms.RandomResizedCrop(size=256),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "test": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    }
    if mode == 'train':
        train_dataset = ImageList(data, transform=transform["train"])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=min(len(train_dataset), batch_size),
                                                   shuffle=True, drop_last=True)
        return train_loader
    else:
        test_dataset = ImageList(data, transform=transform["test"])
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=min(len(test_dataset), batch_size),
                                                  shuffle=False, drop_last=False)
        return test_loader


class TwiceTransform(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224)
        ])
        self.strong = transforms.Compose([
            transforms.RandomResizedCrop(size=256),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(224),
            RandAugmentMC(n=5, m=10)
        ])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class DatasetSplit(torch.utils.data.Dataset):
    def __init__(self, dataset, idxs, transform=None):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.transform = transform

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        if self.transform:
            image = self.transform(image)
        return image, label


def read_data_from_folder(data, batch_size, mode='train'):
    transform = {"train": transforms.Compose([
        transforms.RandomResizedCrop(size=256),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "test": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    }
    if mode == 'train':
        train_dataset = datasets.ImageFolder(data, transform=transform["train"])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=min(len(train_dataset), batch_size),
                                                   shuffle=True, drop_last=True)
        return train_loader, len(train_dataset.classes)
    else:
        test_dataset = datasets.ImageFolder(data, transform=transform["test"])
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=min(len(test_dataset), batch_size),
                                                  shuffle=False, drop_last=False)
        return test_loader


def read_data_from_folder_pu(data, batch_size, mode='train'):
    transform = {"train": transforms.Compose([
        transforms.RandomResizedCrop(size=256),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "test": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    }
    if mode == 'train':
        train_dataset = datasets.ImageFolder(data, transform=transform["train"])
        labels = np.array(train_dataset.targets)
        idx_p = np.where(labels == 0)[0]  # assume that o indicates the positive samples
        idx_u = np.where(labels == 1)[0]  # assume that 1 indicates the unlabeled samples
        train_loader_p = torch.utils.data.DataLoader(DatasetSplit(train_dataset, idx_p),
                                                     batch_size=min(len(idx_p), batch_size),
                                                     shuffle=True,
                                                     drop_last=True)
        train_loader_u = torch.utils.data.DataLoader(DatasetSplit(train_dataset, idx_u),
                                                     batch_size=min(len(idx_u), batch_size),
                                                     shuffle=True,
                                                     drop_last=True)
        return (train_loader_p, train_loader_u), len(train_dataset.classes)
    else:
        test_dataset = datasets.ImageFolder(data, transform=transform["test"])
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=min(len(test_dataset), batch_size),
                                                  shuffle=False, drop_last=False)
        return test_loader


def read_data_from_folder_oc(data, batch_size, mode='train'):
    transform = {"train": transforms.Compose([
        transforms.RandomResizedCrop(size=256),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "test": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    }
    if mode == 'train':
        train_dataset = datasets.ImageFolder(data, transform=TwiceTransform([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        ridx = np.arange(len(train_dataset))
        np.random.shuffle(ridx)
        train_loader = torch.utils.data.DataLoader(DatasetSplit(train_dataset, ridx[:20]),
                                                   batch_size=min(len(train_dataset), 20), shuffle=True,
                                                   drop_last=True)
        return train_loader
    else:
        test_dataset = datasets.ImageFolder(data, transform=transform["test"])
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=min(len(test_dataset), batch_size),
                                                  shuffle=False, drop_last=False)
        return test_loader
