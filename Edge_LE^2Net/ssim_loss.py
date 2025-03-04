import torch
import torch.nn as nn
import numpy as np
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F
from math import exp
from torch.autograd import Variable

from torch import Tensor
from torchvision import models, transforms


def psnr(img1, img2):
    """Calculate the PSNR (Peak Signal-to-Noise Ratio) between two images."""

    # Ensure the input images are in the expected format
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same shape.")

    # Calculate MSE (Mean Squared Error)
    mse = torch.mean((img1 - img2) ** 2)

    # Handle the case where MSE is zero
    if mse == 0:
        return float('inf')  # PSNR is infinite if there's no noise (identical images)

    # Calculate PSNR
    max_pixel = 1.0  # Assuming images are normalized to [0, 1]
    psnr_value = 20 * torch.log10(max_pixel / torch.sqrt(mse))

    return psnr_value.item()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, loss_weight=1.0):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)
        self.loss_weight = loss_weight

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel
        loss = self.loss_weight * (1 - _ssim(img1,
                                             img2,
                                             window,
                                             self.window_size,
                                             channel,
                                             self.size_average))
        return loss


class SSIM_PSNR_Loss(nn.Module):
    def __init__(self, window_size=11, size_average=True, loss_weight=1.0, psnr_weight=0.005):
        super(SSIM_PSNR_Loss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)
        self.loss_weight = loss_weight
        self.psnr_weight = psnr_weight  # New weight for PSNR loss

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        # Create the window for SSIM calculation if needed
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        # Calculate SSIM loss
        ssim_loss = self.loss_weight * (1 - _ssim(img1,
                                                 img2,
                                                 window,
                                                 self.window_size,
                                                 channel,
                                                 self.size_average))

        # Calculate PSNR loss
        psnr_value = psnr(img1, img2)
        psnr_loss = -psnr_value  # We use the negative PSNR as loss to minimize

        # Combine SSIM loss and PSNR loss with their respective weights
        total_loss = ssim_loss + self.psnr_weight * psnr_loss

        return total_loss


class PSNRLoss(nn.Module):
    """PSNR Loss for image comparison."""

    def __init__(self, max_pixel=1.0):
        super(PSNRLoss, self).__init__()
        self.max_pixel = max_pixel

    def forward(self, img1, img2):
        """Calculate the PSNR loss between two images."""

        # Ensure the input images are in the expected format
        if img1.shape != img2.shape:
            raise ValueError("Input images must have the same shape.")

        # Calculate Mean Squared Error (MSE)
        mse = torch.mean((img1 - img2) ** 2)

        # Handle the case where MSE is zero (identical images)
        if mse == 0:
            return torch.tensor(0.0, dtype=torch.float32)  # No loss for identical images

        # Calculate PSNR
        psnr_value = 20 * torch.log10(self.max_pixel / torch.sqrt(mse))

        # PSNR loss is negative of PSNR, since we usually want to minimize the loss
        psnr_loss = -psnr_value
        return psnr_loss



class ssim_l1_Perceptual_loss(nn.Module):
    def __init__(self, ssim_weight=1.0, l1_weight=1.0, perceptual_weight=1.0):
        super(ssim_l1_Perceptual_loss, self).__init__()

        # 预训练的VGG16，用于计算感知损失
        self.vgg16 = models.vgg16(pretrained=True).features[:16].eval()
        for param in self.vgg16.parameters():
            param.requires_grad = False

        # SSIM参数
        self.ssim_weight = ssim_weight
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight

        # 设备设置（使用GPU或CPU）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vgg16 = self.vgg16.to(self.device)

    def forward(self, img1: Tensor, img2: Tensor) -> Tensor:
        """
        计算SSIM损失、L1损失和感知损失，并返回加权总损失
        :param img1: 第一张图像，尺寸为 [B, C, H, W]
        :param img2: 第二张图像，尺寸为 [B, C, H, W]
        :return: 总损失（加权和）
        """

        # 计算SSIM损失
        ssim_loss = self.ssim(img1, img2)

        # 计算L1损失
        l1_loss = F.l1_loss(img1, img2)

        # 计算感知损失
        perceptual_loss = self.perceptual(img1, img2)

        # 计算加权和的总损失
        total_loss = self.ssim_weight * ssim_loss + self.l1_weight * l1_loss + self.perceptual_weight * perceptual_loss

        return total_loss

    def ssim(self, img1: Tensor, img2: Tensor) -> Tensor:
        """
        计算SSIM损失
        :param img1: 第一张图像，尺寸为 [B, C, H, W]
        :param img2: 第二张图像，尺寸为 [B, C, H, W]
        :return: SSIM损失
        """
        # 使用PyTorch实现SSIM损失函数
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu1 = F.avg_pool2d(img1, 3, 1, 1)
        mu2 = F.avg_pool2d(img2, 3, 1, 1)

        sigma1_sq = F.avg_pool2d(img1 * img1, 3, 1, 1) - mu1 * mu1
        sigma2_sq = F.avg_pool2d(img2 * img2, 3, 1, 1) - mu2 * mu2
        sigma12 = F.avg_pool2d(img1 * img2, 3, 1, 1) - mu1 * mu2

        ssim_map = ((2 * sigma12 + C2) * (2 * mu1 * mu2 + C1)) / ((sigma1_sq + sigma2_sq + C2) * (mu1 * mu1 + mu2 * mu2 + C1))

        return 1 - torch.mean(ssim_map)

    def perceptual(self, img1: Tensor, img2: Tensor) -> Tensor:
        """
        计算感知损失
        :param img1: 第一张图像，尺寸为 [B, C, H, W]
        :param img2: 第二张图像，尺寸为 [B, C, H, W]
        :return: 感知损失
        """
        # 使用VGG16模型的特征图作为感知损失
        img1 = (img1 + 1) / 2  # 将图像归一化到[0, 1]区间
        img2 = (img2 + 1) / 2

        # 计算VGG16特征
        feat1 = self.vgg16(img1)
        feat2 = self.vgg16(img2)

        perceptual_loss = F.mse_loss(feat1, feat2)
        return perceptual_loss

