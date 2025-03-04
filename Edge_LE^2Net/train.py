import torch.nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import argparse
import time
from thop import profile
from model import version_4_6_1_2
from Dataloader import PairedImageDataset_No_Flip
from ssim_loss import ssim_l1_Perceptual_loss
from count_result import *
from scheduler import *
from utils import create_new_folder, log_init
from evaluate import evaluate_model

lr_config = dict(warmup='linear',
                 step=[100, 300],
                 liner_end=0.00001,
                 step_gamma=0.1,
                 exp_gamma=0.9)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--dataset_path', type=str, default='/home/robot/桌面/UVTS^2-main/Edge_LE^2Net/dataset', help='dataset file path')
    parser.add_argument('--resume_from', type=str, default=None, help='resume from path')
    parser.add_argument('--epochs', type=int, default=500, help='number of training epochs')
    parser.add_argument('--num_workers', type=int, default=16, help='number of data loading')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--input_size', type=int, default=256, help='input image size')
    parser.add_argument('--save_freq', type=int, default=50, help='frequency of saving')
    parser.add_argument('--save_dir', type=str, default='/home/robot/桌面/UVTS^2-main/Edge_LE^2Net/result/train_result', help='path to save checkpoints')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    clean_image_train = args.dataset_path + '/train/clean'
    raw_image_train = args.dataset_path + '/train/raw'
    clean_image_val = args.dataset_path + '/val/clean'
    raw_image_val = args.dataset_path + '/val/raw'
    num_epochs = args.epochs
    transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
    ])
    new_folder = create_new_folder(args.save_dir)
    loging = log_init(new_folder)
    train_dataset = PairedImageDataset_No_Flip(clean_image_train, raw_image_train, transform)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataset = PairedImageDataset_No_Flip(clean_image_val, raw_image_val, transform)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    model = version_4_6_1_2()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    if args.resume_from != None:
        checkpoint = torch.load(args.resume_from)
        model.load_state_dict(checkpoint, strict=False)
        loging.info(f"load from {args.resume_from}")
    else:
        pass
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    input_tensor = torch.randn(1, 3, args.input_size, args.input_size).to("cuda")
    flops, params = profile(model, inputs=(input_tensor,))
    gflops = flops / 1e9
    params_in_millions = params / 1e6
    loging.info('---------------------')
    # loging.info(model)
    loging.info(f"GFLOPs: {gflops:.2f} GFLOPs")
    loging.info(f"Parameters: {params_in_millions:.2f} M")
    loging.info('----------start training-----------')
    ssim_loss_fn = ssim_l1_Perceptual_loss().to(device)
    loss_history = []
    best_loss = 1
    epoch_scheduler = Epoch(**lr_config)
    scheduler = epoch_scheduler(optimizer, cfg={'total_epoch': 1000})
    for epoch in range(1, num_epochs+1):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        PSNR_history_per_epoch = 0.0
        SSIM_history_per_epoch = 0.0
        for Iteration_idx, (clean, raw) in enumerate(train_dataloader):
            clean, raw = clean.to(device), raw.to(device)
            output = model(raw)
            loss = ssim_loss_fn(output, clean)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            PSNR = psnr(output, clean)
            PSNR_history_per_epoch += PSNR
            SSIM = ssim(output, clean)
            SSIM_history_per_epoch += SSIM
            current_lr = scheduler.get_last_lr()[0]
            print(f'\rEpoch [{epoch}][{Iteration_idx}|{len(train_dataloader)}]: lr: {current_lr}, SSIM: {SSIM:.3f}, PSNR: {PSNR:.3f}', end='', flush=True)
        scheduler.step()
        print("\nvalidating......")
        val_result = evaluate_model(model, val_dataloader, device, loging)
        avg_loss = running_loss / len(train_dataloader)
        val_result.append(avg_loss)
        loss_history.append(avg_loss)
        end_time = time.time()
        loging.info(f"Epoch [{epoch}/{num_epochs}], time: {(end_time-start_time):.4f}s, "
                    f"Loss: {val_result[7]}, SSIM: {val_result[0]:.4f}, PSNR: {val_result[1]:.4f}, "
                    f"UCIQE: {val_result[2]:.3f}, "
                    f'UICM: {val_result[3]:.3f}, UISM: {val_result[4]:.3f}, UICONM: {val_result[5]:.3f}, '
                    f'UIQM: {val_result[6]:.3f}')
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_result = val_result
            best_model_path = new_folder + '/best.pth'
            torch.save(model.state_dict(), best_model_path)
            loging.info(f'Best model saved after epoch {epoch} to {best_model_path}')
        print(f"Best Epoch: [{epoch}/{num_epochs}], "
                    f"Loss: {best_result[7]}, SSIM: {best_result[0]:.4f}, PSNR: {best_result[1]:.4f}, "
                    f"UCIQE: {best_result[2]:.3f}, "
                    f'UICM: {best_result[3]:.3f}, UISM: {best_result[4]:.3f}, UICONM: {best_result[5]:.3f}, '
                    f'UIQM: {best_result[6]:.3f}')
        if epoch % args.save_freq == 0:
            save_path = new_folder + f'/epoch_{epoch}.pth'
            torch.save(model.state_dict(), save_path)
            loging.info(f'Model saved after epoch {epoch} to {save_path}')
        loging.info('---------------------')
    loging.info(f'----training ending----')
    loging.info(f'Average loss: {loss_history}')
    loging.info(f"GFLOPs: {gflops:.2f} GFLOPs")
    loging.info(f"Parameters: {params_in_millions:.2f} M")
    loging.info(f'Best: Epoch[{loss_history.index(min(loss_history)) + 1}], Loss: {min(loss_history)}, SSIM: {ssim_history[loss_history.index(min(loss_history))]}, PSNR: {psnr_history[loss_history.index(min(loss_history))]}')