import os
import torch
import numpy as np
import logging

def create_new_folder(base_path):
    try:
        subfolders = [folder for folder in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, folder))]
    except FileNotFoundError:
        raise FileNotFoundError(f"The directory {base_path} does not exist.")
    numeric_folders = []
    for folder in subfolders:
        if folder.isdigit():
            numeric_folders.append(int(folder))
    if numeric_folders:
        max_folder_num = max(numeric_folders)
    else:
        max_folder_num = 0
    new_folder_name = str(max_folder_num + 1)
    new_folder_path = os.path.join(base_path, new_folder_name)
    os.makedirs(new_folder_path)
    return new_folder_path

def log_init(log_dir):
    os.makedirs(log_dir, exist_ok=True)

    # 设置日志配置
    log_file = os.path.join(log_dir, 'training_log.txt')
    logging.basicConfig(filename=log_file,
                        level=logging.INFO,  # 记录INFO及以上级别的信息
                        format='%(asctime)s - %(message)s')

    # 输出日志到控制台，同时也写入文件
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(console_handler)
    return logging