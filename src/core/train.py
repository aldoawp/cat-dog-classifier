# To do tomorrow:
# - Add labels into each data [V]
# - Combine cats and dogs dataset & shuffle [V]
# - Train model in batch of 16
# - Use 5 different conventional ML algorithm: 
#   - linear regression
#   - decision tree
#   - random forest
#   - support vector machines
#   - k-nearest neighbor 
# - Use CNN deep learning algorithm

import os
from pipeline.build_dataset import build_dataset_traditional
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

train_path = "src/data/train"
val_path = "src/data/validation"
test_path = "src/data/test"

# Get full file paths instead of just filenames
train_cat_files = [os.path.join(train_path, "cats", f) for f in os.listdir(f"{train_path}/cats")]
train_dog_files = [os.path.join(train_path, "dogs", f) for f in os.listdir(f"{train_path}/dogs")]
val_cat_files = [os.path.join(val_path, "cats", f) for f in os.listdir(f"{val_path}/cats")]
val_dog_files = [os.path.join(val_path, "dogs", f) for f in os.listdir(f"{val_path}/dogs")]
test_cat_files = [os.path.join(test_path, "cats", f) for f in os.listdir(f"{test_path}/cats")]
test_dog_files = [os.path.join(test_path, "dogs", f) for f in os.listdir(f"{test_path}/dogs")]

# Labels
train_cat_labels = [0] * len(train_cat_files)
train_dog_labels = [1] * len(train_dog_files)
val_cat_labels = [0] * len(val_cat_files)
val_dog_labels = [1] * len(val_dog_files)
test_cat_labels = [0] * len(test_cat_files)
test_dog_labels = [1] * len(test_dog_files)

# Combined
all_train_files = train_cat_files + train_dog_files
all_train_labels = train_cat_labels + train_dog_labels
all_val_files = val_cat_files + val_dog_files
all_val_labels = val_cat_labels + val_dog_labels
all_test_files = test_cat_files + test_dog_files
all_test_labels = test_cat_labels + test_dog_labels

# Build datasets
print("Building train dataset...")
x_train, y_train = build_dataset_traditional(all_train_files, all_train_labels)

print("Building validation dataset...")
x_val, y_val = build_dataset_traditional(all_val_files, all_val_labels)

print("Building test dataset...")
x_test, y_test = build_dataset_traditional(all_test_files, all_test_labels)

print(f"Train HOG features shape: {x_train.shape}")
print(f"Train labels shape: {y_train.shape}")
print(f"Validation HOG features shape: {x_val.shape}")
print(f"Validation labels shape: {y_val.shape}")
print(f"Test HOG features shape: {x_test.shape}")
print(f"Test labels shape: {y_test.shape}")

# Normalize HOG features
print("Normalizing HOG features...")
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

# Train and validate models
print("Training and validating models...")

algorithms = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'SVM': SVC(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
}

results = {}
best_model = None
best_accuracy = 0
best_model_name = None

for name, model in algorithms.items():
  print(f"\n{'='*20}")
  print(f"Training {name}...")

  model.fit(x_train, y_train)

  # Check validation accuracy
  val_prediction = model.predict(x_val)
  val_accuracy = accuracy_score(y_val, val_prediction)

  # Check training accuracy
  train_prediction = model.predict(x_train)
  train_accuracy = accuracy_score(y_train, train_prediction)

  results[name] = {
    'model': model,
    'train_accuracy': train_accuracy,
    'val_accuracy': val_accuracy
  }

  print(f"{name} Results:")
  print(f"Training Accuracy: {train_accuracy:.4f}")
  print(f"Validation Accuracy: {val_accuracy:.4f}")
  print(f"{'='*20}")

  if val_accuracy > best_accuracy:
    best_accuracy = val_accuracy
    best_model = model
    best_model_name = name

print(f"Best Model: {best_model_name} with validation accuracy {best_accuracy:.4f}")














# all_img = []
# all_labels = []


# for i in range(0, len(all_train_files), BATCH_SIZE):
#   batch_files = all_train_files[i:i+BATCH_SIZE]
#   batch_labels = all_train_labels[i:i+BATCH_SIZE]

#   batch_img = []
#   for file_path, label in zip(batch_files, batch_labels):
#     try:
#       print(f"Processing file: {file_path}")
#       img = pre_process_image(file_path, augment=True)
#       img_arr = img.numpy().flatten()
#       batch_img.append(img_arr)
#     except Exception as e:
#       print(f"Error while processing {file_path}: {e}")
#       continue

#   if batch_img:
#     batch_array = np.array(batch_img)
#     all_img.append(batch_array)
#     all_labels.extend(batch_labels[:len(batch_img)])
    
#   print(f"Processed batch {i//BATCH_SIZE + 1}, total images: {len(all_labels)}")

# if all_img:
#   x_train = np.vstack(all_img)
#   y_train = np.array(all_labels)


# # Train
# algorithms = {
#     'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
#     'Decision Tree': DecisionTreeClassifier(random_state=42),
#     'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
#     'SVM': SVC(random_state=42),
#     'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
# }

## Train and evaluate each algorithm
# results = {}
# for name, model in algorithms.items():
#     print(f"\nTraining {name}...")
#     model.fit(x_train, y_train)
    
#     # Make predictions on training data (you should use validation data)
#     y_pred = model.predict(x_train)
#     accuracy = accuracy_score(y_train, y_pred)
#     results[name] = accuracy
#     print(f"{name} Accuracy: {accuracy:.4f}")


## Dataset building for deep learning
# train_dataset = tf.data.Dataset.from_tensor_slices((all_train_files, all_train_labels))
# train_dataset = train_dataset.map(lambda x: pre_process_image(x, augment=True))
# train_dataset = train_dataset.shuffle(1000)
# train_dataset = train_dataset.batch(8)

# augmentation function [V]
# pre-process function [V]
# convert to HOG
# build conventional ML dataset function (train/val/test dataset)
# build deep learning ML dataset function (train/val/test dataset)
# train


