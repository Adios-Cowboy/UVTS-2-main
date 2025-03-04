import torch
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms
from model import version_4_6_1_2
from Dataloader import PairedImageDataset
from ssim_loss import SSIMLoss
from count_result import *
from utils import create_new_folder, log_init

def parse_args():
    parser = argparse.ArgumentParser(description='Validate a model')
    parser.add_argument('--dataset_path', type=str, default='/home/robot/桌面/UVTS^2-main/Edge_LE^2Net/dataset', help='dataset file path')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--input_size', type=int, default=256, help='input image size')
    parser.add_argument('--model_path', type=str, default='/home/robot/桌面/UVTS^2-main/Edge_LE^2Net/best.pth', help='path to the trained model checkpoint')
    parser.add_argument('--save_dir', type=str, default='/home/robot/桌面/UVTS^2-main/Edge_LE^2Net/result/val_result',help='path to save results')
    args = parser.parse_args()
    return args

def validate(args):
    raw_image_path = args.dataset_path + '/val/raw'
    clean_image_path = args.dataset_path + '/val/clean'
    transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
    ])
    val_dataset = PairedImageDataset(clean_image_path, raw_image_path, transform)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    model = version_4_6_1_2()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    new_folder = create_new_folder(args.save_dir)
    loging = log_init(new_folder)
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
        SSIM = ssim(output, clean)
        PSNR_history_per_epoch += PSNR
        UCIQE_history_per_epoch += UCIQE[0]
        UIQM_history_per_epoch += UIQM[0]
        UICM_history_per_epoch += UICM[0]
        UISM_history_per_epoch += UISM[0]
        UICONM_history_per_epoch += UICONM[0]
        loging.info(f'Batch[{idx + 1}|{len(val_dataloader)}]: SSIM: {SSIM:.3f}, PSNR: {PSNR:.3f}, UCIQE: {UCIQE[0]:.3f}, '
                    f'UICM: {UICM[0]:.3f}, UISM: {UISM[0]:.3f}, UICONM: {UICONM[0]:.3f}, '
                    f'UIQM: {UIQM[0]:.3f}')
    avg_loss = running_loss / len(val_dataloader)
    avg_PSNR = PSNR_history_per_epoch / len(val_dataloader)
    avg_UCIQE = UCIQE_history_per_epoch / len(val_dataloader)
    avg_UICM = UICM_history_per_epoch / len(val_dataloader)
    avg_UISM = UISM_history_per_epoch / len(val_dataloader)
    avg_UICONM = UICONM_history_per_epoch / len(val_dataloader)
    avg_UIQM = UIQM_history_per_epoch / len(val_dataloader)
    loging.info('---------------------')
    loging.info(f"Validation Loss: {avg_loss:.4f}, SSIM: {(1-avg_loss):.4f}, PSNR: {avg_PSNR:.4f}")
    loging.info(f"SSIM: {(1-avg_loss):.4f}")
    loging.info(f"PSNR: {avg_PSNR:.4f}")
    loging.info(f"UCIQE: {avg_UCIQE:.4f}")
    loging.info(f"UICM: {avg_UICM:.4f}")
    loging.info(f"UISM: {avg_UISM:.4f}")
    loging.info(f"UICONM: {avg_UICONM:.4f}")
    loging.info(f"UIQM: {avg_UIQM:.4f}")
    loging.info('---------------------')
    loging.info(f'Validation complete. Average loss: {(1-avg_loss):.4f}, PSNR: {avg_PSNR:.4f}')

if __name__ == '__main__':
    args = parse_args()
    validate(args)
