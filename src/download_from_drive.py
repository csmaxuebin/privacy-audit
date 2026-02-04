#!/usr/bin/env python3
"""
download_from_drive.py

Download trained DPO model from Google Drive to local.

Usage:
    python src/download_from_drive.py

Dependencies:
    pip install pydrive2
"""

import os
import sys
from pathlib import Path

try:
    from pydrive2.auth import GoogleAuth
    from pydrive2.drive import GoogleDrive
except ImportError:
    print("Please install pydrive2 first:")
    print("   pip install pydrive2")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).parent.parent
DRIVE_FOLDER_NAME = "privacy-audit"
LOCAL_OUTPUT_DIR = PROJECT_ROOT / "models" / "stage2_dpo"


def authenticate():
    """Google Drive authentication"""
    print("[INFO] Authenticating with Google Drive...")
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
    """Find folder"""
    if parent_id:
        query = f"title='{folder_name}' and '{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    else:
        query = f"title='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    
    file_list = drive.ListFile({'q': query}).GetList()
    return file_list[0]['id'] if file_list else None


def download_folder(drive, folder_id, local_path):
    """Download all files in folder"""
    os.makedirs(local_path, exist_ok=True)
    
    query = f"'{folder_id}' in parents and trashed=false"
    file_list = drive.ListFile({'q': query}).GetList()
    
    for file in file_list:
        if file['mimeType'] == 'application/vnd.google-apps.folder':
            # Recursively download subfolders
            sub_path = local_path / file['title']
            download_folder(drive, file['id'], sub_path)
        else:
            # Download file
            local_file = local_path / file['title']
            print(f"  [Downloading] {file['title']}")
            file.GetContentFile(str(local_file))


def main():
    print("=" * 60)
    print("Privacy Audit - Download Model from Google Drive")
    print("=" * 60)
    
    drive = authenticate()
    
    # Find folder path
    root_id = find_folder(drive, DRIVE_FOLDER_NAME)
    if not root_id:
        print(f"Drive folder not found: {DRIVE_FOLDER_NAME}")
        return
    
    models_id = find_folder(drive, "models", root_id)
    if not models_id:
        print("models folder not found")
        return
    
    dpo_id = find_folder(drive, "stage2_dpo", models_id)
    if not dpo_id:
        print("stage2_dpo folder not found, training may not be complete")
        return
    
    print(f"\n[INFO] Downloading to: {LOCAL_OUTPUT_DIR}")
    download_folder(drive, dpo_id, LOCAL_OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("[DONE] Download complete!")
    print(f"[INFO] Model location: {LOCAL_OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
