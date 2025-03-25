
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from skimage.transform import resize

from utils import get_image_index
from constants import size

print("dataloader.py loaded successfully")  # added for Debugging line

def load_images_and_gdfms(images_folder_path, gdfm_folder_path, selected_filenames=None):
    """
    Loads images and corresponding GDFMs from the specified folder paths.

    Args:
        images_folder_path (str): Path to the folder containing test images.
        gdfm_folder_path (str): Path to the folder containing GDFMs.
        selected_filenames (list, optional): List of filenames to load. If None, all images are loaded.

    Yields:
        tuple: (image_array, gdfm_ground_truth, img_size, INDEX, image)
    """
    print("Function load_images_and_gdfms is defined")  # checking and Debugging 
    
    # If no specific filenames are provided, load all files from the directory
    if selected_filenames is None:
        selected_filenames = os.listdir(images_folder_path)

    for filename in selected_filenames:
        if filename == '.DS_Store':
            continue
        
        # Obtaining image, fixation, and GDFM names
        image_name = os.path.splitext(filename)[0]
        gdfm_filename = filename.replace("_N_", "_GFDM_N_")

        # Paths to image and GFDM
        file_image_path = os.path.join(images_folder_path, filename)
        file_gdfm_path = os.path.join(gdfm_folder_path, gdfm_filename)

        # Check if GDFM exists (handle missing files)
        if not os.path.exists(file_gdfm_path):
            print(f"Warning: GDFM not found for {filename}")
            continue

        # Load image and GDFM
        image = tf.keras.preprocessing.image.load_img(file_image_path)
        gdfm_ground_truth = Image.open(file_gdfm_path).convert("L")

        INDEX = get_image_index(filename)

        # Preprocess image
        img_size = image.size
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        img_del = np.transpose(image_array, (1, 0, 2))
        gdfm_ground_truth = np.transpose(np.array(gdfm_ground_truth))
        
        # Resize and prepare image
        image_array = resize(image_array, size)
        image_array = tf.expand_dims(image_array, axis=0)

        yield image_array, gdfm_ground_truth, img_size, INDEX, image

