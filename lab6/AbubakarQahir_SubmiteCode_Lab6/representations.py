from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from utils import normalise

def represent_heatmap(saliency, cmap='gray'):
    """
    Create a heatmapped version of a saliency map using a specified colormap.
    
    Args:
        saliency (np.ndarray): Saliency map (unsigned or signed).
        cmap (str): Colormap to be applied.
        
    Returns:
        Image: Heatmapped saliency as a PIL image.
    """
    # Normalize saliency if needed
    if np.max(saliency) > 1:
        saliency = normalise(saliency)

    # Apply colormap and convert to an image
    colormap = plt.get_cmap(cmap)
    heatmapped_saliency = (colormap(saliency) * 255).astype(np.uint8)
    return Image.fromarray(heatmapped_saliency)

def represent_heatmap_overlaid(saliency, image, cmap):
    """
    Overlay a heatmapped saliency map on an RGB image.
    
    Args:
        saliency (np.ndarray): Saliency map (unsigned or signed).
        image (Image): Original RGB image.
        cmap (str): Colormap to be applied.
        
    Returns:
        Image: RGB image with overlaid saliency map.
    """
    # Generate heatmapped saliency and resize to match the input image size
    heatmapped_saliency = represent_heatmap(saliency, cmap)
    heatmapped_saliency = heatmapped_saliency.resize(image.size)

    # Blend the input image and the heatmapped saliency
    blended_image = Image.blend(image.convert('RGBA'), heatmapped_saliency.convert('RGBA'), alpha=0.5)
    return blended_image
