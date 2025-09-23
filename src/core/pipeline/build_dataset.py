import numpy as np
from pipeline.pre_processing import pre_process_traditional
from pipeline.extract_hog_features import extract_hog_features

# Conventional ML dataset building
def build_dataset_traditional(all_files, all_labels):
  all_hog_features = []
  all_file_labels = []
  
  for file_path, label in zip(all_files, all_labels):
    try:
      print(f"Processing file: {file_path}")
      img = pre_process_traditional(file_path)
      img = extract_hog_features(img)
      all_hog_features.append(img)
      all_file_labels.append(label)
    except Exception as e:
      print(f"Error while processing {file_path}: {e}")
      continue

  x_files = np.array(all_hog_features)
  y_labels = np.array(all_file_labels)

  return x_files, y_labels