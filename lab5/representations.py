from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from utils import normalize_matrix

# Converts saliency map to a heatmap using a colormap
def create_heatmap(saliency_map, colormap='gray'):
    # Normalize the saliency map if needed
    if np.max(saliency_map) > 1:
        saliency_map = normalize_matrix(saliency_map)

    # Apply colormap and convert to an image
    cmap = plt.get_cmap(colormap)
    heatmap_image = (cmap(saliency_map) * 255).astype(np.uint8)
    return Image.fromarray(heatmap_image)

# Overlays heatmap onto the original image
def overlay_heatmap_on_image(saliency_map, original_image, colormap):
    # Create heatmap from saliency map
    heatmap_image = create_heatmap(saliency_map, colormap)

    # Resize heatmap to match original image size
    heatmap_image = heatmap_image.resize(original_image.size)

    # Blend heatmap with the original image
    blended_image = Image.blend(original_image.convert('RGBA'), heatmap_image.convert('RGBA'), alpha=0.5)
    return blended_image
