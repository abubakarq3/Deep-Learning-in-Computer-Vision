import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import matplotlib.pyplot as plt

# Constants
VERTICAL_RES = 1200   # given in the lab instructions
SCREEN_HEIGHT = 325   # given in the lab instructions
SCREEN_WIDTH = 1920  # given in the lab instructions
A = 1  # means that the Gaussian function will not be scaled, and the maximum value of the Gaussian will be 1 at the fixation point.

def load_image(file_path):
    return Image.open(file_path)

def load_fixations(file_path):
    fixations = []
    with open(file_path, 'r') as file:
        for line in file:
            columns = line.strip().split()
            fixation_point = (int(columns[0]), int(columns[1]))
            fixations.append(fixation_point)
    return fixations

def load_ground_truth(file_path):
    return Image.open(file_path)

def calculate_sigma(image, D, alpha):
    R = VERTICAL_RES / SCREEN_HEIGHT
    return R * D * np.tan(alpha)

def calculate_error_metrics(ground_truth, saliency_map):
    ground_truth = np.array(ground_truth)
    saliency_map = np.array(saliency_map)

    # Calculate Mean Absolute Error (MAE)
    mae = np.mean(np.abs(ground_truth - saliency_map))
    
    return {'MAE': mae}


def normalise(matrix):
    max_value = np.max(matrix)
    return matrix / max_value

def calculate_partial_saliency_map(image, fixation_point, sigma):
    w, h = image.size
    x = np.arange(w)
    y = np.arange(h)
    X, Y = np.meshgrid(x, y)

    mesh_grid = np.dstack((X, Y))
    mesh_grid = (mesh_grid - fixation_point)**2 / (2 * sigma**2)
    return A * np.exp(-np.sum(mesh_grid, axis=2))

def generate_saliency_map(image, fixation_points):
    sigma = calculate_sigma(image, D=325, alpha=np.deg2rad(2))
    saliency_map = sum(calculate_partial_saliency_map(image, fixation_point, sigma) for fixation_point in fixation_points)
    return normalise(saliency_map)
