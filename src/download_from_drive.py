#!/usr/bin/env python3
"""
download_from_drive.py

从 Google Drive 下载训练好的 DPO 模型到本地。

使用方法：
    python src/download_from_drive.py

依赖：
    pip install pydrive2
"""

import os
import sys
from pathlib import Path

try:
    from pydrive2.auth import GoogleAuth
    from pydrive2.drive import GoogleDrive
except ImportError:
    print("❌ 请先安装 pydrive2:")
    print("   pip install pydrive2")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).parent.parent
DRIVE_FOLDER_NAME = "privacy-audit"
LOCAL_OUTPUT_DIR = PROJECT_ROOT / "models" / "stage2_dpo"


def authenticate():
    """Google Drive 认证"""
    print("[INFO] 正在进行 Google Drive 认证...")
    gauth = GoogleAuth()
    gauth.LoadCredentialsFile("credentials.json")
    
    if gauth.credentials is None:
        gauth.LocalWebserverAuth()
    elif gauth.access_token_expired:
        gauth.Refresh()
    else:
        gauth.Authorize()
    
    gauth.SaveCredentialsFile("credentials.json")
    return GoogleDrive(gauth)


def find_folder(drive, folder_name, parent_id=None):
    """查找文件夹"""
    if parent_id:
        query = f"title='{folder_name}' and '{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    else:
        query = f"title='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    
    file_list = drive.ListFile({'q': query}).GetList()
    return file_list[0]['id'] if file_list else None


def download_folder(drive, folder_id, local_path):
    """下载文件夹中的所有文件"""
    os.makedirs(local_path, exist_ok=True)
    
    query = f"'{folder_id}' in parents and trashed=false"
    file_list = drive.ListFile({'q': query}).GetList()
    
    for file in file_list:
        if file['mimeType'] == 'application/vnd.google-apps.folder':
            # 递归下载子文件夹
            sub_path = local_path / file['title']
            download_folder(drive, file['id'], sub_path)
        else:
            # 下载文件
            local_file = local_path / file['title']
            print(f"  [下载] {file['title']}")
            file.GetContentFile(str(local_file))


def main():
    print("=" * 60)
    print("Privacy Audit - 从 Google Drive 下载模型")
    print("=" * 60)
    
    drive = authenticate()
    
    # 查找文件夹路径
    root_id = find_folder(drive, DRIVE_FOLDER_NAME)
    if not root_id:
        print(f"❌ 未找到 Drive 文件夹: {DRIVE_FOLDER_NAME}")
        return
    
    models_id = find_folder(drive, "models", root_id)
    if not models_id:
        print("❌ 未找到 models 文件夹")
        return
    
    dpo_id = find_folder(drive, "stage2_dpo", models_id)
    if not dpo_id:
        print("❌ 未找到 stage2_dpo 文件夹，训练可能尚未完成")
        return
    
    print(f"\n[INFO] 下载到: {LOCAL_OUTPUT_DIR}")
    download_folder(drive, dpo_id, LOCAL_OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("[DONE] 下载完成!")
    print(f"[INFO] 模型位置: {LOCAL_OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
