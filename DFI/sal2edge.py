import os
import re
import cv2
import numpy as np
from tqdm import tqdm

# 由 sal 计算 edge 图并保存

def LaplacianEdgeDection(img):
    laplacian = cv2.Laplacian(img, cv2.CV_8U, ksize=3)

    return laplacian

def extract_number(filename):
    # 通过正则表达式提取文件名中的数字部分
    number = re.findall(r'\d+', filename)
    return int(number[0]) if number else -1


if __name__ == '__main__':
    sal_img_root = './demo/predictions'
    sal_img_list = os.listdir(sal_img_root)
    # sal_img_list = sorted(sal_img_list, key=extract_number)
    edge_img_root = './demo/edges'
    if not os.path.exists(edge_img_root): os.makedirs(edge_img_root)

    for i, filename in enumerate(tqdm(sal_img_list, desc=f'Images ')):
        sal = cv2.imread(os.path.join(sal_img_root, filename), cv2.IMREAD_GRAYSCALE)
        edge = LaplacianEdgeDection(sal)
        cv2.imwrite(os.path.join(edge_img_root, filename), edge)