# utils.py
import tensorflow as tf
from skimage.transform import resize
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import cv2

# Counter for sequential output filenames
output_counter = 1

def get_sequential_filename(base_path, prefix, extension=".jpg"):
    """
    Generate a sequential filename for saving output images.

    Args:
        base_path (str): The base path to save the file.
        prefix (str): The prefix for the filename.
        extension (str): The file extension.

    Returns:
        str: The full path for the output file.
    """
    global output_counter
    filename = os.path.join(base_path, f"{prefix}_{output_counter}{extension}")
    output_counter += 1
    return filename

def normalise(matrix):
    """
    Normalizes a matrix by dividing it by its maximum value.

    Args:
        matrix (numpy.ndarray): The input matrix to be normalized.

    Returns:
        numpy.ndarray: The normalized matrix.
    """
    max_value = np.max(matrix)
    return matrix / max_value

def apply_blur(image, sigma):
    """
    Apply Gaussian blur to an image.

    Args:
        image (numpy.ndarray): The input image.
        sigma (float): The standard deviation for Gaussian blur.

    Returns:
        numpy.ndarray: The blurred image.
    """
    return cv2.GaussianBlur(image, (0, 0), sigma)

def min_max_normalize(saliency_map):
    """
    Apply min-max normalization to a saliency map.

    Args:
        saliency_map (numpy.ndarray): The input saliency map.

    Returns:
        numpy.ndarray: The normalized saliency map.
    """
    min_value = np.min(saliency_map)
    max_value = np.max(saliency_map)
    return (saliency_map - min_value) / (max_value - min_value)

def scale_by_sum_normalize(saliency_map):
    """
    Normalize a saliency map by dividing by the sum of its values.

    Args:
        saliency_map (numpy.ndarray): The input saliency map.

    Returns:
        numpy.ndarray: The normalized saliency map.
    """
    sum_of_values = np.sum(saliency_map)
    return saliency_map / sum_of_values

def get_last_layer_name(model_name):
    """
    Get the name of the last convolutional layer for a given model.

    Args:
        model_name (str): The name of the model, either 'Xception' or 'ResNet'.

    Returns:
        str: The name of the last convolutional layer.
    """
    if model_name == 'Xception':
        return 'block14_sepconv2_act'
    elif model_name == 'ResNet':
        return 'conv5_block3_out'
    return None

def get_image_index(filename):
    """
    Determine the index based on the image filename prefix.

    Args:
        filename (str): The filename of the image.

    Returns:
        int: The index based on the image class.
    """
    prefix = filename.split('_')[0]
    if prefix == 'Colonial':
        return 0
    elif prefix == 'Modern':
        return 1
    elif prefix == 'Prehispanic':
        return 2
    return -1

def grid_layout(images, titles, suptitle):
    """
    Create and save a grid layout of images with titles.

    Args:
        images (list): List of images to be plotted.
        titles (list): List of titles for these images.
        suptitle (str): The overall title for the grid.
    """
    plt.figure(figsize=(12, 8))  # Create a new figure for the grid layout
    plt.suptitle('Grid layout of Results ' + suptitle, fontsize=16)

    # Plot each image in a 2x2 grid with titles and colorbars
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        cmap = 'gray' if i == 1 else None
        im = plt.imshow(images[i], cmap=cmap)
        plt.colorbar(im, ax=plt.gca())
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout(pad=2.0)  # Adjust layout for spacing

    # Save the grid layout to a file
    os.makedirs('Output_Image_Direcotory', exist_ok=True)
    save_path = get_sequential_filename('Output_Image_Direcotory', 'grid_layout')
    
    plt.savefig(save_path)
    plt.show()  # make this comment if you dont want to cheeck each output . but still it will be saved at output folder.
    plt.close()  # Close the figure after saving

def plot_auc(deletion, n_values_d, scores_d, insertion, n_values_i, scores_i):
    """
    Plot and save AUC plots for Deletion and Insertion metrics.

    Args:
        deletion (float): AUC value for deletion.
        n_values_d (list): List of percentages of deleted pixels.
        scores_d (list): List of scores for deletion.
        insertion (float): AUC value for insertion.
        n_values_i (list): List of percentages of inserted pixels.
        scores_i (list): List of scores for insertion.
    """
    plt.figure(figsize=(10, 5))  # Create a new figure for AUC plots

    # Subplot for Deletion AUC
    plt.subplot(1, 2, 1)
    plt.title('Deletion AUC')
    plt.plot(n_values_d, scores_d)
    plt.xlabel('Deleted Pixels (%)')
    plt.ylabel('Score')
    plt.text(0.5, 0.9, f'AUC: {deletion:.3f}', transform=plt.gca().transAxes, fontsize=12, ha='center')
    plt.fill_between(n_values_d, 0, scores_d, alpha=0.2)

    # Subplot for Insertion AUC
    plt.subplot(1, 2, 2)
    plt.title('Insertion AUC')
    plt.plot(n_values_i, scores_i)
    plt.xlabel('Inserted Pixels (%)')
    plt.ylabel('Score')
    plt.text(0.5, 0.9, f'AUC: {insertion:.3f}', transform=plt.gca().transAxes, fontsize=12, ha='center')
    plt.fill_between(n_values_i, 0, scores_i, alpha=0.2)

    plt.suptitle('Deletion & Insertion AUC Evaluation', fontsize=16)
    plt.tight_layout()

    # Save the AUC plots to a file
    os.makedirs('Output_Image_Direcotory', exist_ok=True)
    save_path = get_sequential_filename('Output_Image_Direcotory', 'auc_plot')
    plt.savefig(save_path)
    plt.show()  # make this comment if you dont want to cheeck each output . but still i will be saved at output folder.
    plt.close()  # Close the figure after saving
