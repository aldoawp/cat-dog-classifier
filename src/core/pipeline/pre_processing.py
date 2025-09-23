import tensorflow as tf
from pipeline.augmentation import augmentation
from skimage.feature import hog
from skimage import color

def pre_process_traditional(path):
  img = tf.io.read_file(path)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.uint8)
  img = tf.image.resize(img, [64, 64])

  img_np = img.numpy()
  img_gray = color.rgb2gray(img_np)

  return img_gray

def pre_process_dl(path, augment=False):
  img = tf.io.read_file(path)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)
  img = tf.image.resize(img, [224, 224])

  if augment:
    img = augmentation(img)

  return img
