import torch
import torch.nn as nn
import numpy as np


__all__ = ['C_module',]

def calculate_covariance_tensor(tensor1, tensor2):
    B, C, H, W = tensor1.shape
    covariance_matrix = torch.zeros((B, C, H, W), device=tensor1.device)

    for b in range(B):
        for c in range(C):
            matrix1 = tensor1[b, c]
            matrix2 = tensor2[b, c]

            mean1 = matrix1.mean()
            mean2 = matrix2.mean()

            matrix1_centered = matrix1 - mean1
            matrix2_centered = matrix2 - mean2

            covariance_matrix[b, c] = (matrix1_centered * matrix2_centered).sum() / (H * W - 1)

    return covariance_matrix

def calculate_small_covariance_tensor(tensor1, tensor2):
    B, C, H, W = tensor1.shape
    covariance_matrix = torch.zeros((B, C, H, W), device=tensor1.device)

    for b in range(B):
        for c in range(C):
            matrix1 = tensor1[b, c]
            matrix2 = tensor2[b, c]
            mean1 = matrix1.mean(dim=1)
            mean2 = matrix2.mean(dim=0)
            covariance_matrix[b, c] = mean1.unsqueeze(1) @ mean2.unsqueeze(0)

    return covariance_matrix

@torch.no_grad()
class C_module(nn.Module):
    def __init__(self,ratio):
        super().__init__()
        if ratio > 1:
            self.re_size = torch.nn.AvgPool2d(kernel_size=ratio, stride=ratio) ##下采样
        elif ratio < 1:
            self.re_size = nn.Upsample(None, 1//ratio, 'nearest')
        elif ratio == 1:
            self.re_size = nn.Identity()
        else:
            raise ValueError("ratio is uncorrect")

    def forward(self, x):
        x1, x2 = x[0], x[1]
        x2 = self.re_size(x2)
        x3 = calculate_small_covariance_tensor(x1, x2)
        return x3


# def center_matrix_with_pytorch(matrix):
#     mean_vector = matrix.mean(dim=0)
#     centered_matrix = matrix - mean_vector
#     return centered_matrix
#
#
# def get_covariance_matrix(tensor1,tensor2,size=20):
#     if tensor1.dim() != 4 or tensor2.dim() != 4 :
#         raise ValueError("Input tensor must have shape B*C*H*W.")
#     if tensor1.shape != tensor2.shape:
#         raise ValueError("Input tensor must have same shape")
#     B, C, H, W = tensor1.shape
#     processed_tensor = tensor1.clone()
#     for b in range(B):
#         for c in range(C):
#             part_1 = center_matrix_with_pytorch(tensor1[b, c])
#             part_2 = center_matrix_with_pytorch(tensor2[b, c])
#             part_1_transpose = part_1.t()
#             part_3 = torch.mm(part_1_transpose,part_2)
#             processed_tensor[b, c] =part_3 / (B-1)
#     return processed_tensor

# def calculate_covariance_tensor(tensor1, tensor2):
#     """
#     计算两个张量中对应的每个 H*W 张量的协方差矩阵
#
#     参数：
#     tensor1 (ndarray): 第一个输入张量，形状为 B*C*H*W
#     tensor2 (ndarray): 第二个输入张量，形状为 B*C*H*W
#
#     返回：
#     ndarray: 协方差矩阵，形状为 B*C*H*W
#     """
#     # 确保输入张量尺寸相同
#     assert tensor1.shape == tensor2.shape, "两个张量的形状必须相同"
#
#     B, C, H, W = tensor1.shape
#     covariance_matrix = np.zeros((B, C, H, W))
#
#     for b in range(B):
#         for c in range(C):
#             # 提取每个 H*W 的矩阵
#             matrix1 = tensor1[b, c]
#             matrix2 = tensor2[b, c]
#
#             # 计算均值
#             mean1 = np.mean(matrix1)
#             mean2 = np.mean(matrix2)
#
#             # 中心化方阵
#             matrix1_centered = matrix1 - mean1
#             matrix2_centered = matrix2 - mean2
#
#             # 计算协方差，H*W的协方差
#             covariance_matrix[b, c] = (matrix1_centered * matrix2_centered) / (H * W - 1)
#
#     return covariance_matrix