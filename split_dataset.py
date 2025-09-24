import os
import shutil
import random

# --- PATHS PRE-FILLED FOR YOU ---
# Source folder containing the 5 'im_(class)' folders
source_dir = '/Users/aishwaryapundir/Downloads'

# Destination where the new 'dataset' folder will be created
dest_dir_parent = '/Users/aishwaryapundir/documents/hit_cervical_project'
# ---------------------------------

# --- SETTINGS ---
train_ratio = 0.70
val_ratio = 0.15
test_ratio = 0.15
# -----------------

dest_dir = os.path.join(dest_dir_parent, 'dataset')

def create_dir_structure(base_dir, classes):
    """Creates the train/val/test folders and clean class subfolders."""
    if os.path.exists(base_dir):
        print(f"Destination directory '{base_dir}' already exists. Please remove it or choose a new name.")
        return False
        
    print(f"Creating directory structure at: {base_dir}")
    for split in ['train', 'val', 'test']:
        for class_name in classes:
            # We use the clean class name for the destination
            path = os.path.join(base_dir, split, class_name)
            os.makedirs(path, exist_ok=True)
    return True

def split_files(source_base, dest_base):
    """Finds images, shuffles them, and copies them to the new structure."""
    
    # Get source class folders (e.g., 'im_superficial-intermediate')
    source_classes_with_prefix = [d for d in os.listdir(source_base) if d.startswith('im_')]
    
    # Get clean class names for destination (e.g., 'superficial-intermediate')
    dest_classes_clean = [d.replace('im_', '') for d in source_classes_with_prefix]

    if not create_dir_structure(dest_base, dest_classes_clean):
        return

    for source_class_name in source_classes_with_prefix:
        clean_class_name = source_class_name.replace('im_', '')
        print(f"\nProcessing class: {clean_class_name}")
        
        # Path to the cropped images for the current class
        cropped_dir = os.path.join(source_base, source_class_name, 'CROPPED')
        
        if not os.path.isdir(cropped_dir):
            print(f"  Warning: 'CROPPED' subdirectory not found in '{source_class_name}'. Skipping.")
            continue
            
        # Get all .bmp images and ignore .dat files
        images = [f for f in os.listdir(cropped_dir) if f.lower().endswith('.bmp')]
        random.shuffle(images)
        
        total_images = len(images)
        train_count = int(total_images * train_ratio)
        val_count = int(total_images * val_ratio)
        
        train_files = images[:train_count]
        val_files = images[train_count : train_count + val_count]
        test_files = images[train_count + val_count:]
        
        def copy_files(files, split_name):
            for filename in files:
                source_path = os.path.join(cropped_dir, filename)
                dest_path = os.path.join(dest_base, split_name, clean_class_name, filename)
                shutil.copyfile(source_path, dest_path)
        
        copy_files(train_files, 'train')
        copy_files(val_files, 'val')
        copy_files(test_files, 'test')
        
        print(f"  Total: {total_images} | Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

    print(f"\n✅ Dataset splitting complete! Your data is ready at:\n{dest_base}")

if __name__ == '__main__':
    if not os.path.isdir(source_dir):
        print(f"Error: Source directory not found at '{source_dir}'")
    elif not os.path.isdir(dest_dir_parent):
        # Create the project directory if it doesn't exist
        print(f"Project directory not found. Creating it at '{dest_dir_parent}'")
        os.makedirs(dest_dir_parent)
        split_files(source_dir, dest_dir)
    else:
        split_files(source_dir, dest_dir)