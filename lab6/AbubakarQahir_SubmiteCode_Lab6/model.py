# model.py

import tensorflow as tf
import cv2
from GRADCAM import GradCAM
from FEM import FEM
from RISE import RISE
from LIME import explain_with_lime
from utils import normalise
from skimage.transform import resize
import numpy as np
from constants import *

#from constants import mask_number, low_res_mask_size, threshold,top_labels

def load_model(model_path='model1.h5'):
    return tf.keras.models.load_model(model_path, compile=False)

def compute_saliency_map(model, method, image_array, INDEX):
    saliency_map = None

    if method == 'GRADCAM':
        gradCAM = GradCAM(model, 'ResNet', image_array, INDEX)
        saliency_map = gradCAM.compute_saliency_map()

    elif method == 'FEM':
        fem = FEM(model, 'ResNet', image_array)
        saliency_map = fem.compute_saliency_map()

    elif method == 'RISE':
        Rise = RISE(model, image_array, INDEX, mask_number, low_res_mask_size, threshold)
        saliency_map = Rise.compute_saliency_map()

    # elif method == 'LIME':
    #     saliency_map = explain_with_lime(model, image_array, top_labels, hide_color=0, num_lime_features=5,
    #                                      num_samples=1000, positive_only=True, negative_only=False, 
    #                                      num_superpixels=50, hide_rest=False, rand_index=0)
    #     saliency_map = cv2.GaussianBlur(saliency_map, (0, 0), 0.5)

    elif method == 'LIME':
            saliency_map = explain_with_lime(model, image_array,
                                             top_labels, hide_color, num_lime_features, num_samples,
                                             positive_only, negative_only, num_superpixels, hide_rest, rand_index)

            # Define the standard deviation (sigma) for the Gaussian blur
            sigma = 0.5

            # Apply Gaussian blur to the heatmap
            saliency_map = cv2.GaussianBlur(saliency_map, (0, 0), sigma)    

    return normalise(saliency_map)
