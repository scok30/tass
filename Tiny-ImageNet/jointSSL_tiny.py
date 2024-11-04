from collections import OrderedDict

import torch
from torchvision import transforms
from torch.nn import functional as F
from PIL import Image
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.autograd import Variable
import numpy as np
import os
import sys
from tqdm import tqdm

from myNetwork import joint_network
from data_manager_tiny import *


class jointSSL:
    def __init__(self, args, feature_extractor, task_size, device):
        self.args = args
        self.model = joint_network(args.fg_nc, feature_extractor)
        self.radius = 0
        self.size = 64
        self.prototype = None
        # self.class_label = None
        # self.numsamples = None
        self.numclass = args.fg_nc
        self.task_size = task_size
        self.old_model = None
        self.device = device
        self.data_manager = DataManager()
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.train_loader = None
        self.test_loader = None

        print('Class_order: ', self.data_manager.get_class_order())


    def beforeTrain(self, current_task):
        self.model.eval()

        class_set = list(range(200))
        if current_task == 0:
            classes = class_set[:self.numclass]
        else:
            classes = class_set[self.numclass-self.task_size: self.numclass]
        print('Classes: ', classes)

        trainfolder = self.data_manager.get_dataset(self.train_transform, index=classes, train=True)
        testfolder = self.data_manager.get_dataset(self.test_transform, index=class_set[:self.numclass], train=False)

        self.train_loader = torch.utils.data.DataLoader(trainfolder, batch_size=self.args.batch_size,
            shuffle=True, drop_last=True, num_workers=8, pin_memory=True)
        self.test_loader = torch.utils.data.DataLoader(testfolder, batch_size=self.args.batch_size,
            shuffle=False, drop_last=False, num_workers=8, pin_memory=True)

        if current_task > 0:
            self.model.Incremental_learning(self.numclass)
        self.model.to(self.device)
        self.model.train()

    def train(self, current_task, old_class=0):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=2e-4)
        scheduler = CosineAnnealingLR(opt, T_max=32)
        for epoch in range(self.args.epochs):
            running_loss = 0.0
            for step, data in enumerate(tqdm(self.train_loader, desc=f'Epoch {epoch}')):
                images, labels = data
                for k in images:
                    images[k] = images[k].to(self.device)
                images, sal, edge = images['img'], images['sal'], images['edge']
                origin_img = images
                labels = labels.to(self.device)

                # # self-supervised learning based label augmentation
                images = torch.stack([torch.rot90(images, k, (2, 3)) for k in range(4)], 1)
                images = images.view(-1, 3, self.size, self.size)
                joint_labels = torch.stack([labels * 4 + k for k in range(4)], 1).view(-1)

                opt.zero_grad()
                loss = self._compute_loss(images, joint_labels, labels, old_class, sal=sal, edge=edge, oi=origin_img)
                opt.zero_grad()
                loss.backward()
                running_loss += loss.item()
                opt.step()
            scheduler.step()
            if epoch % self.args.print_freq == 0 or epoch == self.args.epochs - 1:
                accuracy = self._test(self.test_loader)
                print('epoch:%d, accuracy:%.5f' % (epoch, accuracy))
                logging.info('task: %d, epoch:%d, accuracy:%.5f' % (current_task, epoch, accuracy))
            logging.info('train loss:%.6f'%(running_loss / len(self.train_loader)))
        self.protoSave(self.model, self.train_loader, current_task)

    def _test(self, testloader):
        self.model.eval()
        correct, total = 0.0, 0.0
        for step, data in enumerate(tqdm(testloader, desc=f'Test')):
            imgs, labels = data
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            with torch.no_grad():
                outputs = self.model(imgs)
                # outputs = outputs[:, ::4]  # only compute predictions on original class nodes
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)
        accuracy = correct.item() / total
        self.model.train()
        return accuracy

    def dilation_boundary_loss(self, origin_salmaps, cam_maps):
        def dilation(img, kernel_size):
            max_pool = torch.nn.MaxPool2d(kernel_size=kernel_size,
                                          stride=1,
                                          padding=kernel_size // 2)
            return max_pool(img)
        def erosion(img, kernel_size):
            max_pool = torch.nn.MaxPool2d(kernel_size=kernel_size,
                                          stride=1,
                                          padding=kernel_size // 2)
            return -max_pool(-img)
        def opening(img, kernel_size):
            return dilation(erosion(img, kernel_size), kernel_size)
        def closing(img, kernel_size):
            return erosion(dilation(img, kernel_size), kernel_size)
        # Cal in the external region
        def process(origin_salmaps, kernel_size):
            if origin_salmaps.ndim == 3:
                origin_salmaps = origin_salmaps.unsqueeze(1)  # for process -> extend dim 1
            return dilation(closing(opening(origin_salmaps, kernel_size), kernel_size), kernel_size).squeeze(1)

        kernel_size = 5
        dilated_sal_maps = process(origin_salmaps, kernel_size)
        dilated_sal_maps = (dilated_sal_maps > 0.5).float()
        # external_region
        tar_region = dilated_sal_maps == 0.0
        dbs_loss_pixel = F.l1_loss(cam_maps,
                                   dilated_sal_maps,
                                   reduction='none')

        dbs_loss_pixel[~tar_region] = 0.0  # in_region -> 0

        return torch.mean(dbs_loss_pixel)

    def _compute_loss(self, imgs, joint_labels, labels, old_class=0, **kwargs):
        oi_sal, oi_edge, cam_maps, feature, output = self.model.forward_sal(imgs, targets=None, noise=False)

        joint_preds = self.model.fc(feature)
        single_preds = output[::4]
        joint_preds, joint_labels, single_preds, labels = joint_preds.to(self.device), joint_labels.to(self.device), single_preds.to(self.device), labels.to(self.device)
        joint_labels, labels = joint_labels.type(torch.long), labels.type(torch.long)  # label 的 type 应为 LongTensor
        joint_loss = nn.CrossEntropyLoss()(joint_preds/self.args.temp, joint_labels)
        signle_loss = nn.CrossEntropyLoss()(single_preds/self.args.temp, labels)

        agg_preds = 0
        for i in range(4):
            agg_preds = agg_preds + joint_preds[i::4, i::4] / 4

        distillation_loss = F.kl_div(F.log_softmax(single_preds, 1),
                                    F.softmax(agg_preds.detach(), 1),
                                    reduction='batchmean')

        edge, sal, oi = kwargs['edge'][:, 0], kwargs['sal'][:, 0], kwargs['oi']
        dbs_loss = self.dilation_boundary_loss(sal, cam_maps)
        lms_loss = F.mse_loss(oi_sal, sal) + F.mse_loss(oi_edge, edge)
        saliency_loss = dbs_loss + lms_loss

        if old_class == 0:
            cil_loss = joint_loss + signle_loss + distillation_loss
            return cil_loss + saliency_loss
        else:
            self.old_model.eval()
            with torch.no_grad():
                feature_old = self.old_model.feature(imgs)
            loss_kd = torch.dist(feature, feature_old.detach(), 2)

            proto_aug = []
            proto_aug_label = []
            old_class_list = list(self.prototype.keys())
            for _ in range(feature.shape[0]//4):
                i = np.random.randint(0, feature.shape[0])
                np.random.shuffle(old_class_list)
                lam = np.random.beta(0.5, 0.5)
                if lam > 0.6:
                    # lam = 1 - lam
                    lam = lam * 0.6
                # lam = 0
                # if np.random.random() >= 0.5:
                #     temp = (1 - lam) * self.prototype[old_class_list[0]] + lam * feature.detach().cpu().numpy()[i]
                # else:
                #     temp = (1 - lam) * self.prototype[old_class_list[0]] - lam * feature.detach().cpu().numpy()[i]

                if np.random.random() >= 0.5:
                    temp = (1 + lam) * self.prototype[old_class_list[0]] - lam * feature.detach().cpu().numpy()[i]
                else:
                    temp = (1 - lam) * self.prototype[old_class_list[0]] + lam * feature.detach().cpu().numpy()[i]

                proto_aug.append(temp)
                proto_aug_label.append(old_class_list[0])

            proto_aug = torch.from_numpy(np.float32(np.asarray(proto_aug))).float().to(self.device)
            proto_aug_label = torch.from_numpy(np.asarray(proto_aug_label)).to(self.device)
            proto_aug_label = proto_aug_label.type(torch.long)  # label 的 type 应为 LongTensor
            aug_preds = self.model.classifier(proto_aug)
            joint_aug_preds = self.model.fc(proto_aug)

            agg_preds = 0
            agg_preds = agg_preds + joint_aug_preds[:, ::4]

            aug_distillation_loss = F.kl_div(F.log_softmax(aug_preds, 1),
                                            F.softmax(agg_preds.detach(), 1),
                                            reduction='batchmean')
            loss_protoAug = nn.CrossEntropyLoss()(aug_preds/self.args.temp, proto_aug_label) + nn.CrossEntropyLoss()(joint_aug_preds/self.args.temp, proto_aug_label*4) + aug_distillation_loss

            cil_loss = joint_loss + signle_loss + distillation_loss + self.args.protoAug_weight * loss_protoAug + self.args.kd_weight * loss_kd
            return cil_loss + saliency_loss

    def afterTrain(self, log_root):
        path = log_root + '/'
        if not os.path.isdir(path):
            os.makedirs(path)
        self.numclass += self.task_size
        filename = path + '%d_model.pkl' % (self.numclass - self.task_size)
        torch.save(nn.Sequential(OrderedDict([
            ('feature', self.model.feature),
            ('classifier', self.model.classifier)
        ])), filename)

        self.old_model = torch.load(filename)
        self.old_model.to(self.device)
        self.old_model.eval()

    def protoSave(self, model, loader, current_task):
        features = []
        labels = []
        model.eval()
        with torch.no_grad():
            for step, data in enumerate(loader):
                images, target = data
                if isinstance(images, dict):
                    images = images['img']
                feature = model.feature(images.to(self.device))
                if feature.shape[0] == self.args.batch_size:
                    labels.append(target.numpy())
                    features.append(feature.cpu().numpy())
        labels_set = np.unique(labels)  # 从大到小
        labels = np.array(labels)
        labels = np.reshape(labels, labels.shape[0] * labels.shape[1])
        features = np.array(features)
        features = np.reshape(features, (features.shape[0] * features.shape[1], features.shape[2]))

        prototype = {}

        for item in labels_set:
            index = np.where(item == labels)[0]
            feature_classwise = features[index]
            prototype[item] = np.mean(feature_classwise, axis=0)

        if current_task == 0:
            self.prototype = prototype
        else:
            self.prototype.update(prototype)
