import cv2
import time
import subprocess
import numpy as np
import torch
from torchvision import transforms
from model import version_4_6_1_2
import argparse

# 加载训练好的模型
def load_model(model_path, device):
    model = version_4_6_1_2()
    model = model.to(device)
    model.load_state_dict(torch.load(model_path),strict=False)
    model.eval()
    return model

def enhance_image(model, frame, device, transform):
    # 对图像进行预处理
    input_tensor = transform(frame).unsqueeze(0).to(device)
    # 使用模型进行增强
    with torch.no_grad():
        enhanced = model(input_tensor)
    enhanced_image = enhanced.squeeze().cpu().numpy().transpose(1, 2, 0)
    enhanced_image = np.clip(enhanced_image * 255, 0, 255).astype(np.uint8)
    return enhanced_image

def parse_args():
    parser = argparse.ArgumentParser(description="Real-time Image Enhancement using Pretrained Model.")
    parser.add_argument('--model_path', type=str, default='/home/nvidia/Enhancement-main/result/train_result/32/best.pth', help='Path to the pretrained model file')
    parser.add_argument('--img_size', type=int, default=640, help='Resize image to this size (default: 256)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    parser.add_argument('--camera_id', type=int, default=0, help='ID of the camera to use (default: 0)')
    return parser.parse_args()


# 主程序入口
if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    model = load_model(args.model_path, device)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    server_command = "./rtsp-simple-server"
    subprocess.Popen(["gnome-terminal", "--", "bash", "-c", server_command])
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        exit()

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    command = [
        "ffmpeg",
        "-re",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{frame_width}x{frame_height}",
        "-i", "-",
        "-vcodec", "libx264",
        "-acodec", "aac",
        "-f", "rtsp",
        "rtsp://10.162.31.131:8554/stream"
    ]
    ffmpeg_process = subprocess.Popen(command, stdin=subprocess.PIPE)
    frame_count = 0
    prev_time = time.time()
    fps = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        enhanced_frame = enhance_image(model, frame_rgb, device, transform)
        enhanced_frame_bgr = cv2.cvtColor(enhanced_frame, cv2.COLOR_RGB2BGR)
        frame_count += 1
        curr_time = time.time()
        elapsed_time = curr_time - prev_time
        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            prev_time = curr_time
            frame_count = 0
        cv2.putText(enhanced_frame_bgr, f'EnhancementFPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        ffmpeg_process.stdin.write(enhanced_frame_bgr.tobytes())
        enhanced_frame_bgr = cv2.resize(enhanced_frame_bgr, (0, 0), fx=2,fy=2)  # fx 和 fy 分别表示水平和垂直的缩放因子 # 显示放大的图像 cv2.imshow('Larger Image', larger_img)
        cv2.imshow("Enhanced Image", enhanced_frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # 清理资源
    cap.release()
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()
    cv2.destroyAllWindows()
