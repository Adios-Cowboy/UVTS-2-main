import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
from torchvision import transforms


class RandomHorizontalFlipPair(transforms.RandomHorizontalFlip):
    """
    自定义的水平翻转变换，确保图像对同时翻转。
    """

    def __init__(self, p=0.5):
        super().__init__(p)

    def __call__(self, sample):
        img_1, img_2 = sample
        # 随机决定是否翻转
        if random.random() < self.p:
            img_1 = transforms.functional.hflip(img_1)
            img_2 = transforms.functional.hflip(img_2)
        return img_1, img_2


class PairedImageDataset(Dataset):
    def __init__(self, folder_1, folder_2, transform=None, p=0.5):
        """
        初始化数据集。

        :param folder_1: 存储第一个图像集的文件夹路径
        :param folder_2: 存储第二个图像集的文件夹路径
        :param transform: 用于图像预处理的转换
        """
        self.folder_1 = folder_1
        self.folder_2 = folder_2
        self.transform = transform
        self.p = p

        # 获取两个文件夹中的文件名列表
        self.images_1 = sorted(os.listdir(folder_1))
        self.images_2 = sorted(os.listdir(folder_2))

        # 确保两个文件夹中的图像数目相同
        assert len(self.images_1) == len(self.images_2), "两个文件夹中的图像数量必须一致！"

    def __len__(self):
        """
        返回数据集的大小，即图像对的数量。
        """
        return len(self.images_1)

    def __getitem__(self, idx):
        """
        根据索引返回一对图像，来自 folder_1 和 folder_2，按名称一一对应。
        """
        # 获取当前图像的文件路径
        img_name_1 = os.path.join(self.folder_1, self.images_1[idx])
        img_name_2 = os.path.join(self.folder_2, self.images_2[idx])

        if random.random() < self.p:
            flip = True
        else:
            flip = False

        # 使用with语句确保图像正确关闭
        with Image.open(img_name_1) as img_1, Image.open(img_name_2) as img_2:
            img_1 = img_1.convert("RGB")  # 转为RGB格式
            img_2 = img_2.convert("RGB")

            # 应用预处理
            if flip:
                img_1 = self.transform(transforms.functional.hflip(img_1))
                img_2 = self.transform(transforms.functional.hflip(img_2))
            else:
                img_1 = self.transform(img_1)
                img_2 = self.transform(img_2)

        return img_1, img_2

class PairedImageDataset_No_Flip(Dataset):
    def __init__(self, folder_1, folder_2, transform=None):
        """
        初始化数据集。

        :param folder_1: 存储第一个图像集的文件夹路径
        :param folder_2: 存储第二个图像集的文件夹路径
        :param transform: 用于图像预处理的转换
        """
        self.folder_1 = folder_1
        self.folder_2 = folder_2
        self.transform = transform

        # 获取两个文件夹中的文件名列表
        self.images_1 = sorted(os.listdir(folder_1))
        self.images_2 = sorted(os.listdir(folder_2))

        # 确保两个文件夹中的图像数目相同
        assert len(self.images_1) == len(self.images_2), "两个文件夹中的图像数量必须一致！"

    def __len__(self):
        """
        返回数据集的大小，即图像对的数量。
        """
        return len(self.images_1)

    def __getitem__(self, idx):
        """
        根据索引返回一对图像，来自 folder_1 和 folder_2，按名称一一对应。
        """
        # 获取当前图像的文件路径
        img_name_1 = os.path.join(self.folder_1, self.images_1[idx])
        img_name_2 = os.path.join(self.folder_2, self.images_2[idx])
        # 使用with语句确保图像正确关闭
        with Image.open(img_name_1) as img_1, Image.open(img_name_2) as img_2:
            img_1 = img_1.convert("RGB")  # 转为RGB格式
            img_2 = img_2.convert("RGB")
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)

        return img_1, img_2

class test_Dataset(Dataset):
    def __init__(self, folder, transform=None, p=0.5):

        self.folder = folder
        self.transform = transform
        self.p = p
        self.images = sorted(os.listdir(folder))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_name = os.path.join(self.folder, self.images[idx])

        if random.random() < self.p:
            flip = False
        else:
            flip = False

        # 使用with语句确保图像正确关闭
        with Image.open(img_name) as img:
            img = img.convert("RGB")  # 转为RGB格式
            # 应用预处理
            if flip:
                img = self.transform(transforms.functional.hflip(img))
            else:
                img = self.transform(img)
        return img