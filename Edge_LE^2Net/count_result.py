import cv2
import numpy as np
import torch
from skimage import io, color, filters
import math
import torch.nn.functional as F
from torchvision.transforms import functional as TF
import torchmetrics
from skimage.metrics import structural_similarity as ssim

__all__=['uciqe',
         'uiqm',
         'psnr',
         'nmetrics',
         'ssim']





def uciqe(image):
    """Calculate the UCIQE (Underwater Color Image Quality Evaluation) of an image."""

    # Check if the image is a PyTorch tensor and convert to NumPy if needed
    if isinstance(image, torch.Tensor):
        # Check if the tensor has 4 dimensions (batch_size, channels, height, width)
        if image.dim() == 4:
            # Convert each image in the batch
            uciqe_scores = []
            for img in image:
                img = img.permute(1, 2, 0).detach().cpu().numpy()  # Convert CHW to HWC and move to CPU
                uciqe_scores.append(calculate_uciqe(img))
            return uciqe_scores  # Return a list of UCIQE scores for each image
        elif image.dim() == 3:
            image = image.permute(1, 2, 0).detach().cpu().numpy()  # Convert CHW to HWC and move to CPU
        else:
            raise ValueError("Input tensor must have 3 or 4 dimensions.")
    elif not isinstance(image, np.ndarray):
        raise ValueError("Input image must be a NumPy array or a PyTorch tensor.")

    # Calculate UCIQE for a single image
    return calculate_uciqe(image)[0]

def calculate_uciqe(img):
    """Helper function to calculate UCIQE for a single image."""
    # Convert image to float32 and normalize
    img = img.astype(np.float32)

    # Split the image into its RGB components
    R, G, B = cv2.split(img)

    # Calculate the mean and standard deviation for each channel
    mean_R, std_R = np.mean(R), np.std(R)
    mean_G, std_G = np.mean(G), np.std(G)
    mean_B, std_B = np.mean(B), np.std(B)

    # Calculate the saturation and brightness
    saturation = np.mean(np.sqrt(R ** 2 + G ** 2 + B ** 2))
    brightness = np.mean(0.299 * R + 0.587 * G + 0.114 * B)

    # Calculate the UCIQE components
    UCIQE = (std_R + std_G + std_B) / 3.0 + saturation / 2.0 + brightness / 2.0

    return UCIQE


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

    return psnr_value.item()  # Return the PSNR value as a Python float


def uiqm(image):
    """Calculate the UIQM (Underwater Image Quality Metric) for a single underwater image."""

    # Ensure the input is a PyTorch tensor or a NumPy array
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    if not isinstance(image, np.ndarray):
        raise ValueError("Input image must be a NumPy array or a PyTorch tensor.")

    # Ensure input is in the format (1, channels, height, width)
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
        raise ValueError("Input must have shape (1, 3, height, width).")

    # Normalize the image to [0, 1]
    img = image[0].astype(np.float32)  # Shape (3, height, width)

    # Split the image into its RGB components
    R, G, B = img[0], img[1], img[2]

    # Calculate mean and standard deviation for each channel
    mean_R, std_R = np.mean(R), np.std(R)
    mean_G, std_G = np.mean(G), np.std(G)
    mean_B, std_B = np.mean(B), np.std(B)

    # Calculate brightness
    brightness = np.mean(0.299 * R + 0.587 * G + 0.114 * B)

    # Calculate colorfulness
    colorfulness = np.sqrt((mean_R - mean_G) ** 2 + (mean_G - mean_B) ** 2 + (mean_B - mean_R) ** 2)

    # UIQM calculation
    UIQM = (std_R + std_G + std_B) / 3.0 + colorfulness + brightness / 2.0

    return UIQM  # Return the UIQM value as a float

# Example usage:
# image = cv2.imread('path/to/underwater_image.jpg')
# score = uiem(image)
# print("UIEM Score:", score)

def nmetrics(a):
    # 确保输入是一个 PyTorch 张量，且在范围 [0, 1]
    if not isinstance(a, torch.Tensor):
        raise ValueError("Input must be a PyTorch tensor.")

    # 获取批量大小
    batch_size = a.size(0)

    # 初始化结果列表
    uicm_list = []
    uism_list = []
    uiconm_list = []
    uiqm_list = []

    # 处理每张图片
    for i in range(batch_size):
        # 获取当前批次中的单张图像，并移动到 CPU
        rgb = a[i].permute(1, 2, 0).detach().cpu().numpy()

        # 将 RGB 图像范围调整为 [0, 255] 以适应 skimage 的颜色转换
        rgb = (rgb * 255).astype(np.uint8)

        # 计算 lab 和 gray
        lab = color.rgb2lab(rgb)
        gray = color.rgb2gray(rgb)

        # UIQM 参数
        p1 = 0.0282
        p2 = 0.2953
        p3 = 3.5753

        # 1st term UICM
        rg = rgb[:, :, 0] - rgb[:, :, 1]
        yb = (rgb[:, :, 0] + rgb[:, :, 1]) / 2 - rgb[:, :, 2]
        rgl = np.sort(rg, axis=None)
        ybl = np.sort(yb, axis=None)
        al1 = 0.1
        al2 = 0.1
        T1 = np.int_(al1 * len(rgl))
        T2 = np.int_(al2 * len(rgl))
        rgl_tr = rgl[T1:-T2]
        ybl_tr = ybl[T1:-T2]

        urg = np.mean(rgl_tr)
        s2rg = np.mean((rgl_tr - urg) ** 2)
        uyb = np.mean(ybl_tr)
        s2yb = np.mean((ybl_tr - uyb) ** 2)

        uicm = -0.0268 * np.sqrt(urg ** 2 + uyb ** 2) + 0.1586 * np.sqrt(s2rg + s2yb)

        # 2nd term UISM (k1k2=8x8)
        Rsobel = rgb[:, :, 0] * filters.sobel(rgb[:, :, 0])
        Gsobel = rgb[:, :, 1] * filters.sobel(rgb[:, :, 1])
        Bsobel = rgb[:, :, 2] * filters.sobel(rgb[:, :, 2])

        Rsobel = np.round(Rsobel).astype(np.uint8)
        Gsobel = np.round(Gsobel).astype(np.uint8)
        Bsobel = np.round(Bsobel).astype(np.uint8)

        # 计算边缘能量
        Reme = eme(Rsobel)
        Geme = eme(Gsobel)
        Beme = eme(Bsobel)

        uism = 0.299 * Reme + 0.587 * Geme + 0.114 * Beme

        # 3rd term UIConM
        uiconm = _uiconm(rgb)
        # uiconm = logamee(gray)
        uiqm = p1 * uicm + p2 * uism + p3 * uiconm

        # 保存每个样本的结果
        uicm_list.append(uicm)
        uism_list.append(uism)
        uiconm_list.append(uiconm)
        uiqm_list.append(uiqm)

    # 返回整个批次的结果
    return (
        torch.tensor(uicm_list),
        torch.tensor(uism_list),
        torch.tensor(uiconm_list),
        torch.tensor(uiqm_list)
    )



# def nmetrics(a):
#     # 确保输入是一个 PyTorch 张量，且在范围 [0, 1]
#     if not isinstance(a, torch.Tensor):
#         raise ValueError("Input must be a PyTorch tensor.")
#
#     # 转换为 NumPy 数组，并移动到 CPU
#     rgb = a.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
#
#     # 将 RGB 图像范围调整为 [0, 255] 以适应 skimage 的颜色转换
#     rgb = (rgb * 255).astype(np.uint8)
#
#     # 计算 lab 和 gray
#     lab = color.rgb2lab(rgb)
#     gray = color.rgb2gray(rgb)
#
#     # UIQM
#     p1 = 0.0282
#     p2 = 0.2953
#     p3 = 3.5753
#
#     # 1st term UICM
#     rg = rgb[:, :, 0] - rgb[:, :, 1]
#     yb = (rgb[:, :, 0] + rgb[:, :, 1]) / 2 - rgb[:, :, 2]
#     rgl = np.sort(rg, axis=None)
#     ybl = np.sort(yb, axis=None)
#     al1 = 0.1
#     al2 = 0.1
#     T1 = np.int_(al1 * len(rgl))
#     T2 = np.int_(al2 * len(rgl))
#     rgl_tr = rgl[T1:-T2]
#     ybl_tr = ybl[T1:-T2]
#
#     urg = np.mean(rgl_tr)
#     s2rg = np.mean((rgl_tr - urg) ** 2)
#     uyb = np.mean(ybl_tr)
#     s2yb = np.mean((ybl_tr - uyb) ** 2)
#
#     uicm = -0.0268 * np.sqrt(urg ** 2 + uyb ** 2) + 0.1586 * np.sqrt(s2rg + s2yb)
#
#     # 2nd term UISM (k1k2=8x8)
#     Rsobel = rgb[:, :, 0] * filters.sobel(rgb[:, :, 0])
#     Gsobel = rgb[:, :, 1] * filters.sobel(rgb[:, :, 1])
#     Bsobel = rgb[:, :, 2] * filters.sobel(rgb[:, :, 2])
#
#     Rsobel = np.round(Rsobel).astype(np.uint8)
#     Gsobel = np.round(Gsobel).astype(np.uint8)
#     Bsobel = np.round(Bsobel).astype(np.uint8)
#
#     Reme = eme(Rsobel)
#     Geme = eme(Gsobel)
#     Beme = eme(Bsobel)
#
#     uism = 0.299 * Reme + 0.587 * Geme + 0.114 * Beme
#
#     # 3rd term UIConM
#     uiconm = logamee(gray)
#
#     uiqm = p1 * uicm + p2 * uism + p3 * uiconm
#
#     return (uicm, uism, uiconm, uiqm)


def gaussian_kernel(size, sigma):
    """Generate a Gaussian kernel."""
    kernel = torch.arange(size).float() - size // 2
    kernel = torch.exp(-(kernel ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, -1) * kernel.view(1, -1, 1)


def ssim(x, y, window_size=11, size_average=True, full=False):
    """
    Compute SSIM between two images (B x C x H x W).

    Args:
    - x: tensor of shape (B, C, H, W)
    - y: tensor of shape (B, C, H, W)
    - window_size: size of the Gaussian window
    - size_average: whether to average the SSIM over the batch
    - full: whether to return full SSIM score with contrast and structure components

    Returns:
    - ssim_map (B, C, H, W) or scalar SSIM score
    """
    assert x.size() == y.size(), "Input images must have the same shape"

    window = gaussian_kernel(window_size, 1.5).to(x.device)  # Gaussian kernel
    window = window.expand(x.size(1), 1, window_size, window_size)

    mu_x = F.conv2d(x, window, padding=window_size // 2, groups=x.size(1))
    mu_y = F.conv2d(y, window, padding=window_size // 2, groups=y.size(1))

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(x * x, window, padding=window_size // 2, groups=x.size(1)) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, window, padding=window_size // 2, groups=y.size(1)) - mu_y_sq
    sigma_xy = F.conv2d(x * y, window, padding=window_size // 2, groups=x.size(1)) - mu_xy

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))

    if size_average:
        return ssim_map.mean().item()
    if full:
        return ssim_map, (mu_x_sq, mu_y_sq, mu_xy, sigma_x_sq, sigma_y_sq, sigma_xy)
    return ssim_map.mean().item()

# def nmetrics(a):
#     rgb = a.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
#
#     lab = color.rgb2lab(rgb)
#     gray = color.rgb2gray(rgb)
#
#     # UIQM
#     p1 = 0.0282
#     p2 = 0.2953
#     p3 = 3.5753
#
#     #1st term UICM
#     rg = rgb[:,:,0] - rgb[:,:,1]
#     yb = (rgb[:,:,0] + rgb[:,:,1]) / 2 - rgb[:,:,2]
#     rgl = np.sort(rg,axis=None)
#     ybl = np.sort(yb,axis=None)
#     al1 = 0.1
#     al2 = 0.1
#     T1 = np.int_(al1 * len(rgl))
#     T2 = np.int_(al2 * len(rgl))
#     rgl_tr = rgl[T1:-T2]
#     ybl_tr = ybl[T1:-T2]
#
#     urg = np.mean(rgl_tr)
#     s2rg = np.mean((rgl_tr - urg) ** 2)
#     uyb = np.mean(ybl_tr)
#     s2yb = np.mean((ybl_tr- uyb) ** 2)
#
#     uicm =-0.0268 * np.sqrt(urg**2 + uyb**2) + 0.1586 * np.sqrt(s2rg + s2yb)
#
#     #2nd term UISM (k1k2=8x8)
#     Rsobel = rgb[:,:,0] * filters.sobel(rgb[:,:,0])
#     Gsobel = rgb[:,:,1] * filters.sobel(rgb[:,:,1])
#     Bsobel = rgb[:,:,2] * filters.sobel(rgb[:,:,2])
#
#     Rsobel=np.round(Rsobel).astype(np.uint8)
#     Gsobel=np.round(Gsobel).astype(np.uint8)
#     Bsobel=np.round(Bsobel).astype(np.uint8)
#
#     Reme = eme(Rsobel)
#     Geme = eme(Gsobel)
#     Beme = eme(Bsobel)
#
#     uism = 0.299 * Reme + 0.587 * Geme + 0.114 * Beme
#
#     #3rd term UIConM
#     uiconm = logamee(gray)
#
#     uiqm = p1 * uicm + p2 * uism + p3 * uiconm
#
#     return uiqm


def eme(ch, blocksize=8):
    num_x = math.ceil(ch.shape[0] / blocksize)
    num_y = math.ceil(ch.shape[1] / blocksize)

    eme = 0
    w = 2. / (num_x * num_y)
    for i in range(num_x):

        xlb = i * blocksize
        if i < num_x - 1:
            xrb = (i + 1) * blocksize
        else:
            xrb = ch.shape[0]

        for j in range(num_y):

            ylb = j * blocksize
            if j < num_y - 1:
                yrb = (j + 1) * blocksize
            else:
                yrb = ch.shape[1]

            block = ch[xlb:xrb, ylb:yrb]

            blockmin = np.float_(np.min(block))
            blockmax = np.float_(np.max(block))

            # # old version
            # if blockmin == 0.0: eme += 0
            # elif blockmax == 0.0: eme += 0
            # else: eme += w * math.log(blockmax / blockmin)

            # new version
            if blockmin == 0: blockmin += 1
            if blockmax == 0: blockmax += 1
            eme += w * math.log(blockmax / blockmin)
    return eme

def plipsum(i,j,gamma=1026):
    return i + j - i * j / gamma

def plipsub(i,j,k=1026):
    return k * (i - j) / (k - j)

def plipmult(c,j,gamma=1026):
    return gamma - gamma * (1 - j / gamma)**c

def logamee(ch, blocksize=8):
    print(ch)
    num_x = math.ceil(ch.shape[0] / blocksize)
    num_y = math.ceil(ch.shape[1] / blocksize)

    s = 0
    w = 1. / (num_x * num_y)
    for i in range(num_x):

        xlb = i * blocksize
        if i < num_x - 1:
            xrb = (i + 1) * blocksize
        else:
            xrb = ch.shape[0]

        for j in range(num_y):

            ylb = j * blocksize
            if j < num_y - 1:
                yrb = (j + 1) * blocksize
            else:
                yrb = ch.shape[1]

            block = ch[xlb:xrb, ylb:yrb]
            blockmin = np.float_(np.min(block))
            blockmax = np.float_(np.max(block))

            top = plipsub(blockmax, blockmin)
            bottom = plipsum(blockmax, blockmin)

            m = top / bottom
            if m == 0.:
                s += 0
            else:
                s += (m) * np.log(m)

    return plipmult(w, s)


# def UICONM(rbg, L=8):  # wrong
#     m, n, o = np.shape(rbg)  # 横向为n列 纵向为m行
#     number_m = math.floor(m / L)
#     number_n = math.floor(n / L)
#     A1 = np.zeros((L, L))  # 全0矩阵
#     m1 = 0
#     logAMEE = 0
#     for i in range(number_m):
#         n1 = 0
#         for t in range(number_n):
#             A1 = rbg[m1:m1 + L, n1:n1 + L]
#             rbg_min = int(np.amin(np.amin(A1)))
#             rbg_max = int(np.amax(np.amax(A1)))
#             plip_add = rbg_max + rbg_min - rbg_max * rbg_min / 1026
#             if 1026 - rbg_min > 0:
#                 plip_del = 1026 * (rbg_max - rbg_min) / (1026 - rbg_min)
#                 if plip_del > 0 and plip_add > 0:
#                     local_a = plip_del / plip_add
#                     local_b = math.log(plip_del / plip_add)
#                     phi = local_a * local_b
#                     logAMEE = logAMEE + phi
#             n1 = n1 + L
#         m1 = m1 + L
#     logAMEE = 1026 - 1026 * ((1 - logAMEE / 1026) ** (1 / (number_n * number_m)))
#     return logAMEE

def UICONM(rbg, L=8):  # wrong
    rbg = rbg[:, :, [0, 2, 1]]
    m, n, o = np.shape(rbg)  # 横向为n列 纵向为m行
    number_m = math.floor(m / L)
    number_n = math.floor(n / L)
    A1 = np.zeros((L, L))  # 全0矩阵
    m1 = 0
    logAMEE = 0
    for i in range(number_m):
        n1 = 0
        for t in range(number_n):
            A1 = rbg[m1:m1 + L, n1:n1 + L]
            rbg_min = int(np.amin(np.amin(A1)))
            rbg_max = int(np.amax(np.amax(A1)))
            plip_add = rbg_max + rbg_min - rbg_max * rbg_min / 1026
            if 1026 - rbg_min > 0:
                plip_del = 1026 * (rbg_max - rbg_min) / (1026 - rbg_min)
                if plip_del > 0 and plip_add > 0:
                    local_a = plip_del / plip_add
                    local_b = math.log(plip_del / plip_add)
                    phi = local_a * local_b
                    logAMEE = logAMEE + phi
            n1 = n1 + L
        m1 = m1 + L
    logAMEE = 1026 - 1026 * ((1 - logAMEE / 1026) ** (1 / (number_n * number_m)))
    return logAMEE

def _uiconm(x, window_size=10):
    x.astype(np.float32)
    x = NormalizeData(x)

    plip_lambda = 1026.0
    plip_gamma = 1026.0
    plip_beta = 1.0
    plip_mu = 1026.0
    plip_k = 1026.0
    # if 4 blocks, then 2x2...etc.
    k1 = x.shape[1]/window_size
    k2 = x.shape[0]/window_size
    # weight
    w = -1./(k1*k2)
    blocksize_x = window_size
    blocksize_y = window_size
    # make sure image is divisible by window_size - doesn't matter if we cut out some pixels
    x = x[:int(blocksize_y*k2), :int(blocksize_x*k1)]
    # entropy scale - higher helps with randomness
    alpha = 1
    val = 0
    for l in range(int(k1)):
        for k in range(int(k2)):
            block = x[k*window_size:window_size*(k+1), l*window_size:window_size*(l+1), :]
            max_ = np.max(block)
            min_ = np.min(block)
            top = max_-min_
            bot = max_+min_
            if math.isnan(top) or math.isnan(bot) or bot == 0.0 or top == 0.0: val += 0.0
            else: val += alpha*math.pow((top/bot),alpha) * math.log(top/bot)
            #try: val += plip_multiplication((top/bot),math.log(top/bot))
    return w*val

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))