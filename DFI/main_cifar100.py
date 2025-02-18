import os 
import argparse
from dataset.dataset import get_loader
from solver import Solver
import cv2
import numpy as np
from tqdm import tqdm

def main(config):

    test_loader = get_loader(test_mode=config.dataset_test_mode, sal_mode=config.sal_mode)
    if not os.path.exists(config.test_fold): os.makedirs(config.test_fold)
    test = Solver(test_loader, config)
    test.test(test_mode=config.solver_test_mode)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyper-parameters
    parser.add_argument('--cuda', type=bool, default=True)

    # Testing settings
    parser.add_argument('--model', type=str, default='pretrained/dfi.pth')
    parser.add_argument('--test_fold', type=str, default='demo/predictions')
    parser.add_argument('--dataset_test_mode', type=int, default=3) # choose dataset
    parser.add_argument('--solver_test_mode', type=int, default=1)  # choose task -> saliency
    parser.add_argument('--sal_mode', type=str, default='e') # choose dataset, details in 'dataset/dataset.py'
    parser.add_argument('--img_form', type=str, default='.png')  # choose dataset, details in 'dataset/dataset.py'

    config = parser.parse_args()
    main(config)

    # 将 sal 与 edge 打包为 npy 文件，供后续使用
    sal_img_root = './demo/predictions'
    edge_img_root = './demo/edges'
    sal_img_list = os.listdir(sal_img_root)
    image_num = len(sal_img_list)  # 图片数目
    # 存储 low_level_data 的列表
    low_level_data = []
    for i in tqdm(range(image_num), desc=f'Images '):
        sal = cv2.imread(os.path.join(sal_img_root, f'image_{i}.png'), cv2.IMREAD_GRAYSCALE)
        edge = cv2.imread(os.path.join(edge_img_root, f'image_{i}.png'), cv2.IMREAD_GRAYSCALE)

        low_level_data.append(np.dstack((edge, sal)))  # 在通道维度上叠加 (0 通道: edge, 1 通道: sal)

    output_path = 'lowlevel.npy'
    np.save(output_path, low_level_data)