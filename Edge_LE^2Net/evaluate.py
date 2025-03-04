import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from ssim_loss import SSIMLoss
from Dataloader import PairedImageDataset
from count_result import *

def evaluate_model(model, val_dataloader, device, loging):

    """ 在每轮训练后进行验证 """
    model.eval()
    ssim_loss_fn = SSIMLoss().to(device)
    running_loss = 0.0
    PSNR_history_per_epoch = 0.0
    UCIQE_history_per_epoch = 0.0
    UIQM_history_per_epoch = 0.0
    UICM_history_per_epoch = 0.0
    UISM_history_per_epoch = 0.0
    UICONM_history_per_epoch = 0.0
    for idx, (clean, raw) in enumerate(val_dataloader):
        clean, raw = clean.to(device), raw.to(device)
        with torch.no_grad():
            output = model(raw)
        loss = ssim_loss_fn(output, clean)
        running_loss += loss.item()
        PSNR = psnr(output, clean)
        UCIQE = uciqe(output)
        UICM, UISM, UICONM, UIQM = nmetrics(output)
        PSNR_history_per_epoch += PSNR
        UCIQE_history_per_epoch += UCIQE[0]
        UIQM_history_per_epoch += UIQM[0]
        UICM_history_per_epoch += UICM[0]
        UISM_history_per_epoch += UISM[0]
        UICONM_history_per_epoch += UICONM[0]
        # print(f'valid[{idx}|{len(val_dataloader)}]: SSIM: {loss:.3f}, PSNR: {PSNR:.3f}'
        #         f"UCIQE: {UCIQE[0]:.3f}, "
        #         f'UICM: {UICM[0]:.3f}, UISM: {UISM[0]:.3f}, UICONM: {UICONM[0]:.3f}, '
        #         f'UIQM: {UIQM[0]:.3f}', end = '', flush = True)

    avg_loss = running_loss / len(val_dataloader)
    avg_PSNR = PSNR_history_per_epoch / len(val_dataloader)
    avg_UCIQE = UCIQE_history_per_epoch / len(val_dataloader)
    avg_UICM = UICM_history_per_epoch / len(val_dataloader)
    avg_UISM = UISM_history_per_epoch / len(val_dataloader)
    avg_UICONM = UICONM_history_per_epoch / len(val_dataloader)
    avg_UIQM = UIQM_history_per_epoch / len(val_dataloader)
    return [avg_loss, avg_PSNR, avg_UCIQE, avg_UICM, avg_UISM, avg_UICONM, avg_UIQM]



def get_val_dataloader(dataset_path, batch_size, input_size):
    """ 获取验证数据集的 DataLoader """
    clean_image_path = dataset_path + 'clean'
    raw_image_path = dataset_path + 'raw'
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
    ])
    val_dataset = PairedImageDataset(clean_image_path, raw_image_path, transform)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    return val_dataloader
