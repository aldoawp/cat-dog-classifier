import os
import shutil
import random

SOURCE_DIR = "src/data/raw"
OUTPUT_DIR = "src/data"

train_ratio = 0.7
validation_ratio = 0.2
test_ratio = 0.1

split_folder = ["train", "validation", "test"]
class_folder = ["cats", "dogs"]

# Check whether the folder exist or not
for split in split_folder:
  for cls in class_folder:
    os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

def copy_files(file_list, dir, class_name):
    for file in file_list:
      src = os.path.join(SOURCE_DIR, class_name, file)
      dst = os.path.join(OUTPUT_DIR, dir, class_name, file)
      shutil.copy2(src, dst)

def split_data(class_name):
    src_dir = os.path.join(SOURCE_DIR, class_name)
    files = os.listdir(src_dir)
    random.shuffle(files)

    n_files = len(files)
    n_train_files = int(train_ratio * n_files)
    n_validation_files = int(validation_ratio * n_files) 

    train_files = files[:n_train_files]
    validation_files = files[n_train_files:n_train_files + n_validation_files]
    test_files = files[n_train_files + n_validation_files:]

    print(f"Start splitting and copying files for {class_name}...")
    print(f"Total files: {n_files}")
    print(f"Train files: {len(train_files)}")
    print(f"Validation files: {len(validation_files)}")
    print(f"Test files: {len(test_files)}")

    copy_files(train_files, "train", class_name)
    copy_files(validation_files, "validation", class_name)
    copy_files(test_files, "test", class_name)

    print(f"Files split and copy successful for {class_name}!")

split_data("cats")
split_data("dogs")