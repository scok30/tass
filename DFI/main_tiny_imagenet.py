import os 
import argparse
from dataset.dataset import get_loader
from solver import Solver
import cv2
from tqdm import tqdm

def tiny_imagenet_sal_detection(config, dataset_root, save_root):
    print('*'*30, ' Sal Detection ', '*'*30)
    train_root = os.path.join(dataset_root, 'train')
    folder_list = os.listdir(train_root)
    for i, folder_name in enumerate(folder_list):
        print('#'*15, f' Class{i} ', '#'*15)
        train_dir = os.path.join(train_root, folder_name, 'images')
        test_loader = get_loader(test_mode=config.dataset_test_mode, sal_mode=config.sal_mode, root=train_dir, pin=True)

        config.test_fold = os.path.join(save_root, 'sals', folder_name)     # sal 的保存路径
        if not os.path.exists(config.test_fold): os.makedirs(config.test_fold)
        test = Solver(test_loader, config)
        test.test(test_mode=1)      # sal detection

        # break


# 由 sal 计算 edge 图并保存
def cal_edge_map(lowlevel_root, ksize, img_form='.JPEG'):
    def LaplacianEdgeDection(img):
        return cv2.Laplacian(img, cv2.CV_8U, ksize=ksize)

    print('*' * 30, ' Cal Edge Map ', '*' * 30)
    sal_root = os.path.join(lowlevel_root, 'sals')
    folder_list = os.listdir(sal_root)
    for i, folder_name in enumerate(folder_list):
        print('#' * 15, f' Class{i} ', '#' * 15)
        sal_dir = os.path.join(sal_root, folder_name)
        if not os.path.exists(sal_dir): continue
        sal_img_list = os.listdir(sal_dir)
        edge_img_dir = os.path.join(lowlevel_root, 'edges', folder_name)
        if not os.path.exists(edge_img_dir): os.makedirs(edge_img_dir)
        for i, filename in enumerate(tqdm(sal_img_list, desc=f'Cal edge map')):
            sal_map = cv2.imread(os.path.join(sal_dir, filename), cv2.IMREAD_GRAYSCALE)
            edge_map = LaplacianEdgeDection(sal_map)
            cv2.imwrite(os.path.join(edge_img_dir, filename[:-len(img_form)] + img_form), edge_map)

        print('Cal Edge Map Finished.')
        # break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Hyper-parameters
    parser.add_argument('--cuda', type=bool, default=True)
    # Testing settings
    parser.add_argument('--model', type=str, default='pretrained/dfi.pth')
    parser.add_argument('--test_fold', type=str, default='demo/predictions')
    parser.add_argument('--dataset_test_mode', type=int, default=3) # choose dataset
    parser.add_argument('--solver_test_mode', type=int, default=1)  # choose task -> sal
    parser.add_argument('--sal_mode', type=str, default='e') # choose dataset, details in 'dataset/dataset.py'
    parser.add_argument('--img_form', type=str, default='.JPEG')  # choose dataset, details in 'dataset/dataset.py'

    config = parser.parse_args()
    # config.img_form = '.JPEG'       # 图片格式
    dataset_root = './data/tiny-imagenet-200'
    save_root = './lowlevel_data/tiny-imagenet-200'
    # Sal detection
    tiny_imagenet_sal_detection(config,
                                dataset_root=dataset_root,
                                save_root=save_root)
    # Cal edge map from sal map
    cal_edge_map(lowlevel_root=save_root, ksize=3, img_form='.JPEG')