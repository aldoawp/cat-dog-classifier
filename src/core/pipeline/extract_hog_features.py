from skimage.feature import hog

def extract_hog_features(image_array, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
  hog_features = hog(
    image_array,
    orientations=orientations,
    pixels_per_cell=pixels_per_cell,
    cells_per_block=cells_per_block,
    block_norm='L2-Hys',
    feature_vector=True
  )

  return hog_features
