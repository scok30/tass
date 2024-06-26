import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import os
import math
import argparse
# import resource
import pandas as pd
import numpy as np
import sklearn.metrics
from scipy import stats
from PIL import Image
from tqdm import tqdm

from ResNet import resnet18_cbam

from iCIFAR100 import iCIFAR100
import logging
from datetime import datetime
import random

# from PR_AKA import PR_AKA
from jointSSL import jointSSL

parser = argparse.ArgumentParser(description='Prototype Expansion and for Non-Exampler Class Incremental Learning')
parser.add_argument('--epochs', default=100, type=int, help='Total number of epochs to run')
parser.add_argument('--batch_size', default=128, type=int, help='Batch size for training')
parser.add_argument('--print_freq', default=10, type=int, help='print frequency (default: 10)')
parser.add_argument('--data_name', default='cifar100', type=str, help='Dataset name to use')
parser.add_argument('--total_nc', default=100, type=int, help='class number for the dataset')
parser.add_argument('--fg_nc', default=50, type=int, help='the number of classes in first task')
parser.add_argument('--task_num', default=5, type=int, help='the number of incremental steps')
parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--protoAug_weight', default=15.0, type=float, help='protoAug loss weight')
parser.add_argument('--kd_weight', default=15.0, type=float, help='knowledge distillation loss weight')
parser.add_argument('--temp', default=0.1, type=float, help='trianing time temperature')
parser.add_argument('--gpu', default='0', type=str, help='GPU id to use')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--root', default='../data0/CL_data', type=str, help='data root directory')
parser.add_argument('--save_path', default='model_saved_check/', type=str, help='save files directory')

args = parser.parse_args()
print(args)

def set_random_seed(seed: int) -> None:
    """
    Sets the seeds at a certain value.
    :param seed: the value to be set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main():
    s = datetime.now().strftime('%Y%m%d%H%M%S')
    log_root = '../log/%s/phases_%d/seed_%d-%s-onlymix'%(args.data_name, args.task_num, args.seed, s)
    if not os.path.exists(log_root): 
        os.makedirs(log_root)
    logging.basicConfig(filename='%s/train.log'%log_root,format='%(asctime)s %(message)s', level=logging.INFO)
    logging.info(args)
    set_random_seed(args.seed)
    cuda_index = 'cuda:' + args.gpu
    device = torch.device(cuda_index if torch.cuda.is_available() else "cpu")
    task_size = int((args.total_nc - args.fg_nc) / args.task_num)  # number of classes in each incremental step
    # file_name = args.data_name + '_' + str(args.fg_nc) + '_' + str(args.task_num) + '*' + str(task_size)
    feature_extractor = resnet18_cbam()

    numsuperclass = 20
    model = jointSSL(args, feature_extractor, numsuperclass, task_size, device)
    class_set = list(range(args.total_nc))

    for i in range(args.task_num+1):
        if i == 0:
            old_class = 0
            # model.epochs = 30
        else:
            old_class = len(class_set[:args.fg_nc + (i - 1) * task_size])
            # model.epochs = 100
        model.beforeTrain(i)
        model.train(i, old_class=old_class)
        model.afterTrain(log_root)


    ####### Test ######
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
    print("############# Test for each Task #############")
    test_dataset = iCIFAR100('../data0/CL_data', test_transform=test_transform, train=False, download=True)
    acc_all = []
    for current_task in range(args.task_num+1):
        class_index = args.fg_nc + current_task*task_size
        filename = log_root + '/' + '%d_model.pkl' % (class_index)
        model = torch.load(filename)
        model.eval()
        acc_up2now = []
        for i in range(current_task+1):
            if i == 0:
                classes = [0, args.fg_nc]
            else:
                classes = [args.fg_nc + (i - 1) * task_size, args.fg_nc + i * task_size]
            test_dataset.getTestData_up2now(classes)
            test_loader = DataLoader(dataset=test_dataset,
                                     shuffle=False,
                                     batch_size=args.batch_size,
                                     pin_memory=True)
            correct, total = 0.0, 0.0
            for setp, (indexs, imgs, labels, _) in enumerate(tqdm(test_loader, desc=f'Test for Task{current_task}')):
                imgs, labels = imgs.to(device), labels.to(device)
                with torch.no_grad():
                    fine_outputs, _, _  = model(imgs, noise=False, sal=False)
                # outputs = outputs[:, ::4]
                predicts = torch.max(fine_outputs, dim=1)[1]
                correct += (predicts.cpu() == labels.cpu()).sum()
                total += len(labels)
            accuracy = correct.item() / total
            acc_up2now.append(accuracy)
        if current_task < args.task_num:
            acc_up2now.extend((args.task_num-current_task)*[0])
        acc_all.append(acc_up2now)
        logging.info("phase%d:" % (current_task))
        logging.info(acc_up2now)
        print(acc_up2now)
    print(acc_all)

    a = np.array(acc_all)
    result = []
    for i in range(args.task_num + 1):
        if i == 0:
            result.append(0)
        else:
            res = 0
            for j in range(i + 1):
                res += (np.max(a[:, j]) - a[i][j])
            res = res / i
            result.append(100 * res)
    logging.info(30 * '#')
    logging.info("Forgetting result:")
    logging.info(result)

    print("############# Test for up2now Task #############")
    test_dataset = iCIFAR100('../data0/CL_data', test_transform=test_transform, train=False, download=True)
    incremental_acc = []
    for current_task in range(args.task_num+1):
        class_index = args.fg_nc + current_task*task_size
        filename = log_root + '/' + '%d_model.pkl' % (class_index)
        model = torch.load(filename)
        model.to(device)
        model.eval()

        classes = [0, args.fg_nc + current_task * task_size]
        test_dataset.getTestData_up2now(classes)
        test_loader = DataLoader(dataset=test_dataset,
                                 shuffle=False,
                                 batch_size=args.batch_size,
                                 pin_memory=True)
        correct, total = 0.0, 0.0
        for setp, (indexs, imgs, labels, _) in enumerate(tqdm(test_loader, desc=f'Test for up2 Task{current_task}')):
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                fine_outputs, _, _ = model(imgs, noise=False, sal=False)
            # outputs = outputs[:, ::4]
            predicts = torch.max(fine_outputs, dim=1)[1]
            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)
        accuracy = correct.item() / total
        incremental_acc.append(accuracy)
        print(accuracy)
        logging.info("phase%d: %.8f" % (current_task, accuracy))
    logging.info("average incremental_acc:%.8f"%(np.mean(incremental_acc)))

if __name__ == "__main__":
    main()