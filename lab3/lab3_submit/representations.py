from PIL import Image, ImageChops
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
import random
import os

class Saliency:
    def __init__(self, saliency):
        self.image = saliency  # Saliency map

class RGBImage:
    def __init__(self, image):
        self.image = image  # Pillow Image

def normalise(matrix):
    max_value = np.max(matrix)
    return matrix / max_value

def represent_heatmap(saliency: Saliency, cmap: str = 'gray') -> RGBImage:
    if np.max(saliency) > 1:
        saliency = normalise(saliency)

    colormap = plt.get_cmap(cmap)
    heatmapped_saliency = (colormap(saliency) * 255).astype(np.uint8)
    heatmapped_saliency_image = Image.fromarray(heatmapped_saliency)
    return RGBImage(heatmapped_saliency_image).image

def represent_heatmap_overlaid(saliency: Saliency, image: RGBImage, cmap: str) -> RGBImage:
    heatmapped_saliency = represent_heatmap(saliency, cmap)
    blended_image = Image.blend(image.convert('RGBA'), heatmapped_saliency.convert('RGBA'), alpha=0.5)
    return RGBImage(blended_image).image

def represent_isolines(saliency: Saliency, cmap: str) -> RGBImage:
    contour_levels = np.linspace(0, 1, 11)
    level_list = []

    saliency = normalise(saliency)
    w, h = saliency.shape
    isoline = np.zeros((w, h))

    for level in contour_levels:
        contours_list = measure.find_contours(saliency, level)
        level_list.append(contours_list)

    for level in level_list:
        for contour in level:
            for x, y in contour:
                isoline[int(x), int(y)] = saliency[int(x), int(y)]

    return represent_heatmap(isoline, cmap)

def represent_isolines_superimposed(saliency: Saliency, image: RGBImage, cmap: str) -> RGBImage:
    isolines_heatmapped = represent_isolines(saliency, cmap)
    blended_image = Image.blend(image.convert('RGBA'), isolines_heatmapped.convert('RGBA'), alpha=0.5)
    return RGBImage(blended_image).image

def represent_hard_selection(saliency: Saliency, image: RGBImage, threshold: int) -> RGBImage:
    mask = (saliency >= threshold).astype(np.uint8)
    mask = Image.fromarray(mask * 255)

    w, h = saliency.shape
    blank = Image.fromarray(np.zeros((w, h)).astype(np.uint8)).convert('RGBA')

    hard_image = Image.composite(image, blank, mask)
    return RGBImage(hard_image).image

def represent_soft_selection(saliency: Saliency, image: RGBImage) -> RGBImage:
    saliency = Image.fromarray(saliency).convert('RGB')
    soft_image = ImageChops.multiply(image.convert('RGB'), saliency)
    return RGBImage(soft_image).image


def grid_layout(images, titles):
    """
    Displays images in a 2x3 grid with improved text styling and arrangement.
    """
    # Creates a 2x3 grid for the images
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))  # Larger figure size for clarity
    fig.suptitle('Results', fontsize=22, weight='bold', color='black')  # Bigger and bold grid title

    # Loop through images and titles, and plot them in the grid
    for i, (image, title) in enumerate(zip(images, titles)):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        # Display the image
        ax.imshow(image)
        
        # Set title with improved styling
        ax.set_title(title, fontsize=14, weight='bold', color='darkblue')
        
        # Remove axis for cleaner presentation
        ax.axis('off')

    # Adjust the layout to improve spacing
    plt.subplots_adjust(wspace=0.2, hspace=0.3)  # Add more space between images

    # Save and display the grid
    os.makedirs('lab3_output_images', exist_ok=True)
    save_index = random.randint(1, 100)
    plt.savefig(f'lab3_output_images/grid_{save_index}.jpg', bbox_inches='tight')  # Save with tight bounding box for neatness
    plt.show()
