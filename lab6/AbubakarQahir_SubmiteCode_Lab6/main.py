import os
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image
from skimage.transform import resize
import cv2

# Importing custom modules for data handling, model operations, and visual representation
from dataloader import load_images_and_gdfms
from model import load_model, compute_saliency_map
from representations import represent_heatmap, represent_heatmap_overlaid
from constants import *
from evaluation import *

# Importing specific visual explanation methods
from GRADCAM import *
from FEM import *
from RISE import *
from LIME import *

# Re-importing certain functions to ensure availability within the script
from representations import (represent_heatmap, represent_heatmap_overlaid)
from constants import *
from utils import (normalise, get_image_index, grid_layout, plot_auc)
from evaluation import *

def parse_args():
    parser = argparse.ArgumentParser(description="Run main.py with the specified parameters for test images, GDFMs, explanation method, model, and display type.")
    
#
#E:\Bordo\DL\Lab6\MexCulture142\test\test1  (contains 9 images where 3 images for each class)
    #E:\Bordo\DL\Lab6\MexCulture142\images_val  ( contain full validation set)
    parser.add_argument("--test_images_folder_path", default=r'E:\Bordo\DL\Lab6\MexCulture142\images_val', 
                        help="Path to the folder contwaining the test images.")
    parser.add_argument("--test_gdfm_folder_path", default=r'E:\Bordo\DL\Lab6\MexCulture142\gazefixationsdensitymaps', 
                        help="Path to the folder containing the Ground-truth Gaze Fixation Density Maps (GDFMs).")
    parser.add_argument("--explanation_method", choices=['GRADCAM', 'FEM', 'RISE', 'LIME'], default='LIME',
                        help="Explanation type to be used; options are GRADCAM, FEM, RISE, or LIME.")
    parser.add_argument("--model_name", choices=['ResNet'], default='ResNet', help="Classifier model to be used; currently supported: ResNet.")
    parser.add_argument("--display_type", choices=['grid', 'singles'], default='grid', type=str, help="Display layout for results; choose 'grid' or 'singles'.")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    #Load the pre-trained model
    model = load_model()

    # Dictionary to store evaluation metrics for the batch of images
    batch_errors = {'PCC': [], 'SSIM': [], 'Insertion': [], 'Deletion': []}

    #   Iterate over images and their corresponding GDFMs from the specified directories
    for image_array, gdfm_ground_truth, img_size, INDEX, image in load_images_and_gdfms(args.test_images_folder_path, args.test_gdfm_folder_path):
        
        # Compute the saliency map using the chosen explanation method
        saliency_map = compute_saliency_map(model, args.explanation_method, image_array, INDEX)
        
        # Resize the saliency map to match the original image dimensions
        saliency_map = resize(saliency_map, img_size, order=3, mode='wrap', anti_aliasing=False)
        gray_saliency = represent_heatmap(saliency_map, 'gray')
        heatmapped_saliency = represent_heatmap(saliency_map, 'jet').resize(img_size)
        #Overlay the saliency map on the original image
        saliency_blended = represent_heatmap_overlaid(saliency_map, image, 'jet')
        
        # Calculate metrics
        pcc = calculate_pcc(normalise(gdfm_ground_truth), saliency_map)
        ssim = calculate_sim(normalise(gdfm_ground_truth), saliency_map)

        # Insertion and Deletion metrics without ground truth
        ins_auc, ins_scores, n_values_i = insertion(model, np.array(tf.squeeze(image_array)), resize(saliency_map, size), 500, INDEX)
        del_auc, del_scores, n_values_d = deletion(model, np.array(tf.squeeze(image_array)), resize(saliency_map, size), 500, INDEX)
        
         # Store computed metrics in the batch_errors dictionary

        batch_errors['PCC'].append(pcc)
        batch_errors['SSIM'].append(ssim)
        batch_errors['Insertion'].append(ins_auc)
        batch_errors['Deletion'].append(del_auc)
        
        # Display images based on the chosen display type
        images = [gray_saliency, gdfm_ground_truth, heatmapped_saliency, saliency_blended]
        titles = ['Computed Saliency Map', 'Groundtruth Saliency Map', 'Heatmap Visualization of Saliency', 'Blended Saliency Representation']
        # supertitle = f"{args.explanation_method} Explanation Method, using {args.model_name} Transfer Learned Model"
        supertitle = f"Visualization of {args.explanation_method} with {args.model_name} Transfer Learning"
        
        plot_auc(del_auc, n_values_d, del_scores, ins_auc, n_values_i, ins_scores)
        
        if args.display_type == 'singles':
            for image in images:
                image.show()
        elif args.display_type == 'grid':
            grid_layout(images, titles, supertitle)
    
    #  Print the mean and variance of metrics for all processed images
    for error_type, values in batch_errors.items():
        print(f"{error_type} ----> mean: {np.mean(values)}, variance: {np.std(values)}")


