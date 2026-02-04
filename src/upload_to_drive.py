#!/usr/bin/env python3
"""
upload_to_drive.py

将训练数据和模型文件上传到 Google Drive，供 Colab 使用。

使用方法：
1. 首次运行会打开浏览器进行 Google 授权
2. 授权后文件会自动上传到 Drive 的 privacy-audit 文件夹

依赖：
    pip install pydrive2
"""

import os
import sys
from pathlib import Path

# 检查依赖
try:
    from pydrive2.auth import GoogleAuth
    from pydrive2.drive import GoogleDrive
except ImportError:
    print("❌ 请先安装 pydrive2:")
    print("   pip install pydrive2")
    sys.exit(1)

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 需要上传的文件
FILES_TO_UPLOAD = [
    "data/preference_data.jsonl",
    "models/stage1_sft/adapter_config.json",
    "models/stage1_sft/adapter_model.safetensors",
    "models/stage1_sft/tokenizer.json",
    "models/stage1_sft/tokenizer_config.json",
    "models/stage1_sft/special_tokens_map.json",
    "models/stage1_sft/vocab.json",
    "models/stage1_sft/merges.txt",
]

# Drive 目标文件夹
DRIVE_FOLDER_NAME = "privacy-audit"


def authenticate():
    """Google Drive 认证"""
    print("[INFO] 正在进行 Google Drive 认证...")
    
    gauth = GoogleAuth()
    
    # 尝试加载已保存的凭证
    gauth.LoadCredentialsFile("credentials.json")
    
    if gauth.credentials is None:
        # 首次认证，使用命令行方式
        print("[INFO] 首次使用，请在浏览器中完成授权...")
        try:
            gauth.LocalWebserverAuth(port_numbers=[8080, 8090, 9000, 9090])
        except:
            # 如果本地服务器失败，使用命令行方式
            print("[INFO] 使用命令行认证方式...")
            gauth.CommandLineAuth()
    elif gauth.access_token_expired:
        print("[INFO] 刷新认证 token...")
        gauth.Refresh()
    else:
        gauth.Authorize()
    
    gauth.SaveCredentialsFile("credentials.json")
    print("[OK] 认证成功!")
    
    return GoogleDrive(gauth)


def get_or_create_folder(drive, folder_name):
    """获取或创建 Drive 文件夹"""
    # 搜索是否已存在
    query = f"title='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    file_list = drive.ListFile({'q': query}).GetList()
    
    if file_list:
        folder = file_list[0]
        print(f"[INFO] 使用已存在的文件夹: {folder_name}")
    else:
        # 创建新文件夹
        folder = drive.CreateFile({
            'title': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        })
        folder.Upload()
        print(f"[OK] 创建文件夹: {folder_name}")
    
    return folder['id']


def get_or_create_subfolder(drive, parent_id, folder_name):
    """在父文件夹下获取或创建子文件夹"""
    query = f"title='{folder_name}' and '{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    file_list = drive.ListFile({'q': query}).GetList()
    
    if file_list:
        return file_list[0]['id']
    else:
        folder = drive.CreateFile({
            'title': folder_name,
            'parents': [{'id': parent_id}],
            'mimeType': 'application/vnd.google-apps.folder'
        })
        folder.Upload()
        return folder['id']


def upload_file(drive, local_path, parent_id, filename=None):
    """上传单个文件到指定文件夹"""
    if filename is None:
        filename = os.path.basename(local_path)
    
    # 检查文件是否已存在
    query = f"title='{filename}' and '{parent_id}' in parents and trashed=false"
    existing = drive.ListFile({'q': query}).GetList()
    
    if existing:
        # 更新已存在的文件
        file = existing[0]
        file.SetContentFile(str(local_path))
        file.Upload()
        print(f"  [更新] {filename}")
    else:
        # 创建新文件
        file = drive.CreateFile({
            'title': filename,
            'parents': [{'id': parent_id}]
        })
        file.SetContentFile(str(local_path))
        file.Upload()
        print(f"  [上传] {filename}")
    
    return file['id']


def main():
    print("=" * 60)
    print("Privacy Audit - 上传文件到 Google Drive")
    print("=" * 60)
    
    # 认证
    drive = authenticate()
    
    # 创建主文件夹
    root_folder_id = get_or_create_folder(drive, DRIVE_FOLDER_NAME)
    
    # 创建子文件夹结构
    data_folder_id = get_or_create_subfolder(drive, root_folder_id, "data")
    models_folder_id = get_or_create_subfolder(drive, root_folder_id, "models")
    sft_folder_id = get_or_create_subfolder(drive, models_folder_id, "stage1_sft")
    
    print(f"\n[INFO] 开始上传文件...")
    
    # 上传文件
    for file_path in FILES_TO_UPLOAD:
        full_path = PROJECT_ROOT / file_path
        
        if not full_path.exists():
            print(f"  [跳过] {file_path} (文件不存在)")
            continue
        
        # 确定目标文件夹
        if file_path.startswith("data/"):
            parent_id = data_folder_id
        elif file_path.startswith("models/stage1_sft/"):
            parent_id = sft_folder_id
        else:
            parent_id = root_folder_id
        
        upload_file(drive, full_path, parent_id)
    
    print("\n" + "=" * 60)
    print("[DONE] 所有文件上传完成!")
    print(f"[INFO] Drive 文件夹: {DRIVE_FOLDER_NAME}/")
    print("=" * 60)
    print("\n下一步: 在 Colab 中运行 notebooks/02_dpo_training.ipynb")


if __name__ == "__main__":
    main()
