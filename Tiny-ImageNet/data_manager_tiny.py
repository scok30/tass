import logging
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import datasets, transforms
import os
import sys
import pdb
import random


class tinyImageNet(object):
    def download_data(self):
        train_dir = os.path.join('../data0/CL_data', 'tiny-imagenet-200', 'train')
        test_dir = os.path.join('../data0/CL_data', 'tiny-imagenet-200', 'val')
        train_dset = datasets.ImageFolder(train_dir)

        train_images = []
        train_labels = []
        for item in train_dset.imgs:
            train_images.append(item[0])
            train_labels.append(item[1])
        self.train_data, self.train_targets = np.array(train_images), np.array(train_labels)

        test_images = []
        test_labels = []
        _, class_to_idx = find_classes(train_dir)
        imgs_path = os.path.join(test_dir, 'images')
        imgs_annotations = os.path.join(test_dir, 'val_annotations.txt')
        with open(imgs_annotations) as r:
            data_info = map(lambda s: s.split('\t'), r.readlines())
        cls_map = {line_data[0]: line_data[1] for line_data in data_info}
        for imgname in sorted(os.listdir(imgs_path)):
            if cls_map[imgname] in sorted(class_to_idx.keys()):
                path = os.path.join(imgs_path, imgname)
                test_images.append(path)
                test_labels.append(class_to_idx[cls_map[imgname]])
        self.test_data, self.test_targets = np.array(test_images), np.array(test_labels)


class DataManager(object):
    def __init__(self):
        self._setup_data()

    def get_class_order(self):
        return self._class_order

    def get_dataset(self, transform, index, train=True):
        if train:
            x, y = self._train_data, self._train_targets
        else:
            x, y = self._test_data, self._test_targets

        data, targets = [], []
        for idx in index:
            class_data, class_targets = self._select(x, y, low_range=idx, high_range=idx+1)
            data.append(class_data)
            targets.append(class_targets)
        data, targets = np.concatenate(data), np.concatenate(targets)
        return DummyDataset(data, targets, transform, train=train)

    def _setup_data(self, shuffle=True, seed=1993):
        idata = tinyImageNet()
        idata.download_data()

        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets

        order = [i for i in range(len(np.unique(self._train_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = np.arange(200).tolist()
        self._class_order = order
        logging.info("Class order:")
        logging.info(self._class_order)

        self._train_targets = _map_new_class_index(self._train_targets, self._class_order)
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)

    def _select(self, x, y, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[idxes], y[idxes]


class DummyDataset(Dataset):
    def __init__(self, images, labels, trsf, train):
        assert len(images) == len(labels), 'Data size error!'
        self.images = images
        self.labels = labels
        self.trsf = trsf        # transform
        self.train = train
        if self.train and self.trsf:
            sub_idx = [0, 1]
            self.lowlevel_trsf = transforms.Compose(      # lowlevel_trsf: wo normalization
                [self.trsf.transforms[w] for w in sub_idx]
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # image
        img_path = self.images[idx]
        image = pil_loader(self.images[idx], mode='RGB')
        # train: image & lowlevel
        if self.train:
            # sal & edge
            sal_dir = os.path.join('../data0/Lowlevel_data', 'tiny-imagenet-200', 'sals')
            edge_dir = os.path.join('../data0/Lowlevel_data', 'tiny-imagenet-200', 'edges')
            def get_classfolder_and_filename(img_path):
                # 使用 os.path.basename() 获取文件名部分
                filename = os.path.basename(img_path)
                # 切分两次，获得文件夹名称
                directory, _ = os.path.split(img_path)
                directory, _ = os.path.split(directory)
                classfolder = os.path.basename(directory)
                return classfolder, filename
            classfolder, filename = get_classfolder_and_filename(img_path)
            sal_path = os.path.join(sal_dir, classfolder, filename)
            edge_path = os.path.join(edge_dir, classfolder, filename)
            sal = pil_loader(sal_path, mode='L')
            edge = pil_loader(edge_path, mode='L')

            if self.trsf:
                state = torch.random.get_rng_state()
                image = self.trsf(image)
                state2 = torch.random.get_rng_state()
                torch.random.set_rng_state(state)
                sal = self.lowlevel_trsf(sal)
                torch.random.set_rng_state(state)
                edge = self.lowlevel_trsf(edge)
                torch.random.set_rng_state(state2)
            if isinstance(image, torch.Tensor):
                image = {
                    'img': image,
                    'sal': sal,
                    'edge': edge,
                }
        # test: only image
        else:
            if self.trsf:
                image = self.trsf(image)

        label = self.labels[idx]
        return image, label


def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


def pil_loader(path, mode='RGB'):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert(mode)


def find_classes(dir):
    if sys.version_info >= (3, 5):
        # Faster and available in Python 3.5 and above
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    else:
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx