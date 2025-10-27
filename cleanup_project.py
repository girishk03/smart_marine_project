#!/usr/bin/env python3
"""
Safe Project Cleanup Script
===========================

Moves old/unnecessary files to _archive/ folder.
Nothing is deleted - everything is safely backed up!
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent
ARCHIVE_DIR = PROJECT_ROOT / "_archive" / datetime.now().strftime("%Y%m%d_%H%M%S")

# Files to KEEP (everything else gets archived)
KEEP_FILES = {
    # Main application
    'reliable_web_app.py',
    'requirements.txt',
    
    # Dataset & training
    'data_plastic_only.yaml',
    'create_plastic_only_dataset.py',
    'train_plastic_only_model.py',
    
    # Documentation (essential only)
    'README.md',
    'QUICK_START.md',
    'DATASET_CLEANUP_COMPLETE.md',
    'PROJECT_CLEANUP_PLAN.md',
    
    # Git
    '.gitignore',
    '.git',
    
    # This script
    'cleanup_project.py',
    'PROJECT_CLEANUP_PLAN.md',
}

# Directories to KEEP
KEEP_DIRS = {
    'smart_marine_project',
    'smart_marine_project_backup',
    'vessel_modules',
    'train_plastic_only',
    'valid_plastic_only',
    'yolov5',
    'runs',
    '_archive',
    '.git',
    '.venv',
    '.devcontainer',
}

def should_archive(path):
    """Check if file/dir should be archived"""
    name = path.name
    
    # Keep essential files
    if name in KEEP_FILES:
        return False
    
    # Keep essential directories
    if path.is_dir() and name in KEEP_DIRS:
        return False
    
    # Archive everything else
    return True

def main():
    print("=" * 70)
    print("🧹 SAFE PROJECT CLEANUP")
    print("=" * 70)
    print()
    print("This script will:")
    print("  1. Move old/test files to _archive/ folder")
    print("  2. Keep only essential production files")
    print("  3. Nothing is deleted (safe backup)")
    print()
    
    # Create archive directory
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📂 Archive location: {ARCHIVE_DIR}")
    print()
    
    # Scan root directory
    items = list(PROJECT_ROOT.iterdir())
    to_archive = [item for item in items if should_archive(item)]
    
    print(f"📊 Found {len(items)} items in root directory")
    print(f"   Keep: {len(items) - len(to_archive)} essential files")
    print(f"   Archive: {len(to_archive)} old/test files")
    print()
    
    if not to_archive:
        print("✅ Project is already clean!")
        return
    
    # Show what will be archived
    print("📋 Files to archive:")
    print("-" * 70)
    
    # Group by type
    py_files = [f for f in to_archive if f.suffix == '.py']
    md_files = [f for f in to_archive if f.suffix == '.md']
    img_files = [f for f in to_archive if f.suffix in ['.jpg', '.png', '.jpeg']]
    yaml_files = [f for f in to_archive if f.suffix in ['.yaml', '.yml']]
    other_files = [f for f in to_archive if f not in py_files + md_files + img_files + yaml_files]
    
    if py_files:
        print(f"\n🐍 Python scripts ({len(py_files)}):")
        for f in sorted(py_files)[:10]:
            print(f"   - {f.name}")
        if len(py_files) > 10:
            print(f"   ... and {len(py_files) - 10} more")
    
    if md_files:
        print(f"\n📄 Markdown docs ({len(md_files)}):")
        for f in sorted(md_files)[:10]:
            print(f"   - {f.name}")
        if len(md_files) > 10:
            print(f"   ... and {len(md_files) - 10} more")
    
    if img_files:
        print(f"\n🖼️  Images ({len(img_files)}):")
        for f in sorted(img_files)[:5]:
            print(f"   - {f.name}")
        if len(img_files) > 5:
            print(f"   ... and {len(img_files) - 5} more")
    
    if yaml_files:
        print(f"\n⚙️  Config files ({len(yaml_files)}):")
        for f in yaml_files:
            print(f"   - {f.name}")
    
    if other_files:
        print(f"\n📦 Other files ({len(other_files)}):")
        for f in sorted(other_files)[:5]:
            print(f"   - {f.name}")
        if len(other_files) > 5:
            print(f"   ... and {len(other_files) - 5} more")
    
    print()
    print("-" * 70)
    print()
    
    # Ask for confirmation
    response = input("🤔 Proceed with archiving? (yes/no): ").strip().lower()
    
    if response not in ['yes', 'y']:
        print("❌ Cleanup cancelled")
        return
    
    print()
    print("🚀 Starting cleanup...")
    print()
    
    # Archive files
    archived_count = 0
    failed_count = 0
    
    for item in to_archive:
        try:
            dest = ARCHIVE_DIR / item.name
            if item.is_file():
                shutil.move(str(item), str(dest))
            else:
                shutil.move(str(item), str(dest))
            archived_count += 1
            print(f"✅ Archived: {item.name}")
        except Exception as e:
            print(f"❌ Failed: {item.name} - {e}")
            failed_count += 1
    
    print()
    print("=" * 70)
    print("✅ CLEANUP COMPLETE!")
    print("=" * 70)
    print()
    print(f"📊 Results:")
    print(f"   Archived: {archived_count} items")
    print(f"   Failed: {failed_count} items")
    print(f"   Kept: {len(items) - archived_count} essential files")
    print()
    print(f"📂 Archived files location:")
    print(f"   {ARCHIVE_DIR}")
    print()
    print("💡 To restore a file:")
    print(f"   cp {ARCHIVE_DIR}/filename.py .")
    print()
    print("🎯 Your project is now clean and organized!")
    print()

if __name__ == "__main__":
    main()
