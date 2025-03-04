import torch
from torch import nn
from torchvision.transforms import transforms
from PIL import Image, ImageDraw
import sys
sys.path.append("")
from src.core import YAMLConfig
import argparse
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import ImageDraw


class Model(nn.Module):
    def __init__(self, config=None, ckpt="") -> None:
        super().__init__()
        self.cfg = YAMLConfig(config, resume=ckpt)
        if ckpt:
            checkpoint = torch.load(ckpt, map_location='cpu')
            if 'ema' in checkpoint:
                state = checkpoint['ema']['module']
            else:
                state = checkpoint['model']
        else:
            raise AttributeError('only support resume to load model.state_dict by now.')

        # NOTE load train mode state -> convert to deploy mode
        self.cfg.model.load_state_dict(state)

        self.model = self.cfg.model.deploy()
        self.postprocessor = self.cfg.postprocessor.deploy()
        # print(self.postprocessor.deploy_mode)

    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        return self.postprocessor(outputs, orig_target_sizes)



def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/home/robot/RT-DETR-main/RT-DETR-main/rtdetrv2_pytorch/configs/rtdetrv2/v11backbone_2.yml", help="配置文件路径")
    parser.add_argument("--ckpt", default="/home/robot/RT-DETR-main/RT-DETR-main/output/rtdetrv2_r18vd_120e_coco 33+57+56+16+16/best.pth", help="权重文件路径")
    parser.add_argument("--image", default="/home/robot/RUOD/RUOD_COCO/val2017/000000006723.jpg", help="待推理图片路径")
    parser.add_argument("--output_dir", default="/home/robot/RT-DETR-main/RT-DETR-main", help="输出文件保存路径")
    parser.add_argument("--device", default="cuda")

    return parser

class ImageReader:
    def __init__(self, resize=640):
        self.resize = resize

    def __call__(self, pil_img):
        # 直接处理传入的 PIL 图像
        img = pil_img.convert('RGB').resize((self.resize, self.resize))
        img = np.array(img).transpose(2, 0, 1)  # 转换为 (C, H, W)
        img = torch.tensor(img, dtype=torch.float32) / 255.0  # 归一化
        img = img.unsqueeze(0)  # 增加 batch 维度
        return img

def detection(model, device):
    rtsp_url = 'rtsp://10.162.31.131:8554/stream'
    cap = cv2.VideoCapture(rtsp_url)

    reader = ImageReader(resize=640)  # 图像读取器，尺寸缩放至640x640
    thrh = 0.6  # 设置阈值

    frame_count = 0
    prev_time = time.time()
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 如果没有读取到帧，退出循环



        # 记录原始图像的尺寸
        original_height, original_width = frame.shape[:2]

        # 将 OpenCV 图像格式 (BGR) 转换为 PIL 图像格式 (RGB)
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # 将图像传递给模型
        img_tensor = reader(pil_img).to(device)
        size = torch.tensor([[img_tensor.shape[2], img_tensor.shape[3]]]).to(device)

        # 执行推理
        start = time.time()
        output = model(img_tensor, size)
        print(f"推理耗时：{time.time() - start:.4f}s")

        labels, boxes, scores = output

        # 将检测框坐标从归一化坐标转换为原始图像的坐标
        draw = ImageDraw.Draw(pil_img)



        for i in range(img_tensor.shape[0]):
            scr = scores[i]
            lab = labels[i][scr > thrh]
            box = boxes[i][scr > thrh]

            for b, l in zip(box, lab):
                # 归一化坐标转换为原始图像尺寸的坐标
                b = b.int().tolist()
                b[0] = int(b[0] * original_width / 640)  # x坐标映射回原始图像
                b[1] = int(b[1] * original_height / 640)  # y坐标映射回原始图像
                b[2] = int(b[2] * original_width / 640)  # w坐标映射回原始图像
                b[3] = int(b[3] * original_height / 640)  # h坐标映射回原始图像

                draw.rectangle(list(b), outline='red', width=2)
                draw.text((b[0], b[1] - 10), text=str(l.item()), fill='blue')

        # 将带检测结果的 PIL 图像转换回 OpenCV 格式
        frame_with_detections = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        frame_count += 1
        curr_time = time.time()
        elapsed_time = curr_time - prev_time

        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            prev_time = curr_time
            frame_count = 0

        cv2.putText(frame_with_detections, f"DetectionFPS: {fps:.2f},", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        # 显示带检测结果的视频帧

        frame_with_detections = cv2.resize(frame_with_detections, (0, 0), fx=2, fy=2)

        cv2.imshow('Detection Image', frame_with_detections)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头并关闭窗口
    cap.release()
    cv2.destroyAllWindows()


def main(args):
    # 加载模型并设置设备
    device = torch.device(args.device)
    model = Model(config=args.config, ckpt=args.ckpt)
    model.to(device=device)

    # 运行实时检测
    detection(model, device)

if __name__ == "__main__":
    main(get_argparser().parse_args())