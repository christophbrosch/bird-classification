import os

import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image

class BirdDataset(Dataset):
    def __init__(self, train, transform = None):
        self.transform = transform
        with open('./data/train_test_split.txt') as f:
            test_train_split = {}
            for line in f.readlines():
                id, affiliation = line.split(' ')
                affiliation = affiliation.replace('\n', '')
                test_train_split[int(id)] = bool(int(affiliation))

        with open('./data/images.txt', 'r') as f:
            self.images = []
            for line in f.readlines():
                id, image = line.split(' ')
                image = image.replace('\n', '')
                if train and test_train_split[int(id)]:
                    self.images.append((int(id), image))
                elif not train and not test_train_split[int(id)]:
                    self.images.append((int(id), image))

        with open('./data/image_class_labels.txt', 'r') as f:
            self.labels = {}
            for line in f.readlines():
                id, label = line.split(' ')
                self.labels[int(id)] = int(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_id, image = self.images[idx]
        img = read_image('data/images/' + image)
        img = img / 255
        label = self.labels[image_id]
        if self.transform is not None:
            img = self.transform(img)
        return img, label
