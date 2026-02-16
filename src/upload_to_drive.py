#!/usr/bin/env python3
"""
upload_to_drive.py

Upload training data and model files to Google Drive for Colab use.

Usage:
1. First run will open browser for Google authorization
2. After authorization, files will be automatically uploaded to privacy-audit folder in Drive

Dependencies:
    pip install pydrive2
"""

import os
import sys
from pathlib import Path

# Check dependencies
try:
    from pydrive2.auth import GoogleAuth
    from pydrive2.drive import GoogleDrive
except ImportError:
    print("Please install pydrive2 first:")
    print("   pip install pydrive2")
    sys.exit(1)

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Files to upload
FILES_TO_UPLOAD = [
    "data/preference_data_no_canary.jsonl",
    "data/preference_data_with_canary.jsonl",
    "models/stage1_sft/adapter_config.json",
    "models/stage1_sft/adapter_model.safetensors",
    "models/stage1_sft/tokenizer.json",
    "models/stage1_sft/tokenizer_config.json",
    "models/stage1_sft/special_tokens_map.json",
    "models/stage1_sft/vocab.json",
    "models/stage1_sft/merges.txt",
]

# Drive target folder
DRIVE_FOLDER_NAME = "privacy-audit"


def authenticate():
    """Google Drive authentication"""
    print("[INFO] Authenticating with Google Drive...")
    
    gauth = GoogleAuth()
    
    # Try to load saved credentials
    gauth.LoadCredentialsFile("credentials.json")
    
    if gauth.credentials is None:
        # First time authentication, use command line method
        print("[INFO] First time use, please complete authorization in browser...")
        try:
            gauth.LocalWebserverAuth(port_numbers=[8080, 8090, 9000, 9090])
        except:
            # If local server fails, use command line method
            print("[INFO] Using command line authentication...")
            gauth.CommandLineAuth()
    elif gauth.access_token_expired:
        print("[INFO] Refreshing authentication token...")
        gauth.Refresh()
    else:
        gauth.Authorize()
    
    gauth.SaveCredentialsFile("credentials.json")
    print("[OK] Authentication successful!")
    
    return GoogleDrive(gauth)


def get_or_create_folder(drive, folder_name):
    """Get or create Drive folder"""
    # Search if already exists
    query = f"title='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    file_list = drive.ListFile({'q': query}).GetList()
    
    if file_list:
        folder = file_list[0]
        print(f"[INFO] Using existing folder: {folder_name}")
    else:
        # Create new folder
        folder = drive.CreateFile({
            'title': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        })
        folder.Upload()
        print(f"[OK] Created folder: {folder_name}")
    
    return folder['id']


def get_or_create_subfolder(drive, parent_id, folder_name):
    """Get or create subfolder under parent folder"""
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
    """Upload single file to specified folder"""
    if filename is None:
        filename = os.path.basename(local_path)
    
    # Check if file already exists
    query = f"title='{filename}' and '{parent_id}' in parents and trashed=false"
    existing = drive.ListFile({'q': query}).GetList()
    
    if existing:
        # Update existing file
        file = existing[0]
        file.SetContentFile(str(local_path))
        file.Upload()
        print(f"  [Updated] {filename}")
    else:
        # Create new file
        file = drive.CreateFile({
            'title': filename,
            'parents': [{'id': parent_id}]
        })
        file.SetContentFile(str(local_path))
        file.Upload()
        print(f"  [Uploaded] {filename}")
    
    return file['id']


def main():
    print("=" * 60)
    print("Privacy Audit - Upload Files to Google Drive")
    print("=" * 60)
    
    # Authenticate
    drive = authenticate()
    
    # Create main folder
    root_folder_id = get_or_create_folder(drive, DRIVE_FOLDER_NAME)
    
    # Create subfolder structure
    data_folder_id = get_or_create_subfolder(drive, root_folder_id, "data")
    models_folder_id = get_or_create_subfolder(drive, root_folder_id, "models")
    sft_folder_id = get_or_create_subfolder(drive, models_folder_id, "stage1_sft")
    
    print(f"\n[INFO] Starting file upload...")
    
    # Upload files
    for file_path in FILES_TO_UPLOAD:
        full_path = PROJECT_ROOT / file_path
        
        if not full_path.exists():
            print(f"  [Skipped] {file_path} (file does not exist)")
            continue
        
        # Determine target folder
        if file_path.startswith("data/"):
            parent_id = data_folder_id
        elif file_path.startswith("models/stage1_sft/"):
            parent_id = sft_folder_id
        else:
            parent_id = root_folder_id
        
        upload_file(drive, full_path, parent_id)
    
    print("\n" + "=" * 60)
    print("[DONE] All files uploaded successfully!")
    print(f"[INFO] Drive folder: {DRIVE_FOLDER_NAME}/")
    print("=" * 60)
    print("\nNext step: Run notebooks/02_dpo_training.ipynb in Colab")


if __name__ == "__main__":
    main()
