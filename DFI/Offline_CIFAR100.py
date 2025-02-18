import os
import pickle
from PIL import Image


def load_cifar100_data(data_path):
    train_data = []
    test_data = []

    # 加载训练集数据
    with open(os.path.join(data_path, 'train'), 'rb') as f:
        train_dict = pickle.load(f, encoding='bytes')
        train_data = train_dict[b'data']

    # 加载测试集数据
    with open(os.path.join(data_path, 'test'), 'rb') as f:
        test_dict = pickle.load(f, encoding='bytes')
        test_data = test_dict[b'data']

    return train_data, test_data


def save_images(data, save_folder):
    num_images = len(data)

    for i in range(num_images):
        image = data[i].reshape(3, 32, 32).transpose(1, 2, 0)  # 将图像数据从(C, H, W)转换为(H, W, C)
        image = Image.fromarray(image)
        image_path = os.path.join(save_folder, f'image_{i}.png')
        image.save(image_path)
        print(f'Saved image {i + 1}/{num_images}')



if __name__ == '__main__':
    data_path = './data/cifar-100-python'  # CIFAR-100数据集所在的文件夹路径

    train_data, test_data = load_cifar100_data(data_path)

    save_folder = './demo/images'  # 图像保存的文件夹路径

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    save_images(train_data, save_folder)  # 保存训练集图像