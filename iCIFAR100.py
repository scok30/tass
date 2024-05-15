import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR100


class iCIFAR100(CIFAR100):
    def __init__(self, root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 test_transform=None,
                 target_test_transform=None,
                 download=False):
        super(iCIFAR100, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.target_test_transform = target_test_transform
        self.test_transform = test_transform

        self.low_level_data = np.load('../data0/low_level_data/cifar100-lowlevel.npy')   # 50000x32x32x3, 0 channel: edge, 1 channel: sal
        # print('low_level_data.shape: ', self.low_level_data.shape)

        if self.transform:
            sub_idx = [0, 1, 3]
            self.low_level_transform = transforms.Compose(      # low_level_transform, 保证 low_level_data 的变换与数据集图像保持一致！
                [self.transform.transforms[w] for w in sub_idx]
            )
        self.TrainData = []
        self.TrainLabels = []
        self.TrainCoarseLabels = []
        self.TestData = []
        self.TestLabels = []
        self.TestCoarseLabels = []
        # 粗分类标签
        self.coarse_labels = np.array([4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                                      3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                                      6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                                      0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                                      5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                                      16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                                      10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                                      2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                                      16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                                      18, 1, 2, 15, 6, 0, 17, 8, 14, 13])

    # def concatenate(self, datas, labels, coarse_labels):
    #     con_data = datas[0]
    #     con_label = labels[0]
    #     coarse_label = coarse_labels[0]
    #     for i in range(1, len(datas)):
    #         con_data = np.concatenate((con_data, datas[i]), axis=0)
    #         con_label = np.concatenate((con_label, labels[i]), axis=0)
    #         coarse_label = np.concatenate((coarse_label, coarse_labels[i]), axis=0)
    #
    #     return con_data, con_label, coarse_label


    def concatenate(self, list):
        con_data = list[0]
        for i in range(1, len(list)):
            con_data = np.concatenate((con_data, list[i]), axis=0)

        return con_data


    def getTestData(self, classes):
        datas, labels = [], []
        for label in range(classes[0], classes[1]):
            data = self.data[np.array(self.targets) == label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        coarse_labels = self.coarse_labels[labels]
        # datas, labels, coarse_labels = self.concatenate(datas, labels, coarse_labels)
        datas = self.concatenate(datas)
        labels = self.concatenate(labels)
        coarse_labels = self.concatenate(coarse_labels)
        self.TestData = datas if self.TestData == [] else np.concatenate((self.TestData, datas), axis=0)
        self.TestLabels = labels if self.TestLabels == [] else np.concatenate((self.TestLabels, labels), axis=0)
        self.TestCoarseLabels = coarse_labels if self.TestCoarseLabels == [] else np.concatenate((self.TestCoarseLabels, coarse_labels), axis=0)
        print("the size of test set is %s" % (str(self.TestData.shape)))
        print("the size of test label is %s" % str(self.TestLabels.shape))
        print("the size of test coarse label is %s" % str(self.TestCoarseLabels.shape))

    def getTestData_up2now(self, classes):
        datas, labels = [], []
        for label in range(classes[0], classes[1]):
            data = self.data[np.array(self.targets) == label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        coarse_labels = self.coarse_labels[labels]
        # datas, labels, coarse_labels = self.concatenate(datas, labels, coarse_labels)
        datas = self.concatenate(datas)
        labels = self.concatenate(labels)
        coarse_labels = self.concatenate(coarse_labels)
        self.TestData = datas
        self.TestLabels = labels
        self.TestCoarseLabels = coarse_labels
        print("the size of test set is %s" % (str(datas.shape)))
        print("the size of test label is %s" % str(labels.shape))
        print("the size of test coarse label is %s" % str(coarse_labels.shape))

    def getTrainData(self, classes):
        datas, labels = [], []
        edges, sals = [], []
        for label in range(classes[0], classes[1]):
            pos = np.array(self.targets) == label

            data = self.data[pos]
            datas.append(data)
            edge = self.low_level_data[pos, :, :, 0:1]
            edges.append(edge)
            sal = self.low_level_data[pos, :, :, 1:2]
            sals.append(sal)

            labels.append(np.full((data.shape[0]), label))
        coarse_labels = self.coarse_labels[labels]
        # self.TrainData, self.TrainLabels, self.TrainCoarseLabels = self.concatenate(datas, labels, coarse_labels)
        self.TrainData = self.concatenate(datas)
        self.TrainLabels = self.concatenate(labels)
        self.TrainCoarseLabels = self.concatenate(coarse_labels)
        self.Edges = self.concatenate(edges)
        self.Sals = self.concatenate(sals)
        print("the size of train set is %s" % (str(self.TrainData.shape)))
        print("the size of train label is %s" % str(self.TrainLabels.shape))
        print("the size of train coarse label is %s" % str(self.TrainCoarseLabels.shape))
        print("the size of edges is %s" % str(self.Edges.shape))
        print("the size of sals is %s" % str(self.Sals.shape))

    def getTrainItem(self, index):
        img, target, coarse_target = Image.fromarray(self.TrainData[index]), self.TrainLabels[index], self.TrainCoarseLabels[index]
        edge, sal = (Image.fromarray(np.repeat(self.Edges[index], 3, axis=2)),
                     Image.fromarray(np.repeat(self.Sals[index], 3, axis=2)))


        if self.transform:      # 确保对同一种图像及其 LowLevelData 所做的 transform 的一致性
            state = torch.random.get_rng_state()
            img = self.transform(img)
            state2 = torch.random.get_rng_state()
            torch.random.set_rng_state(state)
            edge = self.low_level_transform(edge)
            torch.random.set_rng_state(state)
            sal = self.low_level_transform(sal)
            torch.random.set_rng_state(state2)
        if self.target_transform:
            target = self.target_transform(target)
        if isinstance(img, torch.Tensor):       # img 包含原图、边缘图、显著性图三部分
            img = {
                'img': img,
                'edge': edge,
                'sal': sal,
            }

        # torchvision.transforms.ToPILImage()(img['img']).show('img')
        # torchvision.transforms.ToPILImage()(img['edge']).show('edge')
        # torchvision.transforms.ToPILImage()(img['sal']).show('sal')
        # input("按下回车键继续...")

        return index, img, target, coarse_target

    def getTestItem(self, index):
        img, target, coarse_target = Image.fromarray(self.TestData[index]), self.TestLabels[index], self.TestCoarseLabels[index]
        if self.test_transform:
            img = self.test_transform(img)
        if self.target_test_transform:
            target = self.target_test_transform(target)
        return index, img, target, coarse_target

    def __getitem__(self, index):
        if self.TrainData != []:
            return self.getTrainItem(index)
        elif self.TestData != []:
            return self.getTestItem(index)

    def __len__(self):
        if self.TrainData != []:
            return len(self.TrainData)
        elif self.TestData != []:
            return len(self.TestData)

    def get_image_class(self, label):
        return self.data[np.array(self.targets) == label]


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]