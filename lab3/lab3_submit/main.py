import argparse
import numpy as np
from PIL import Image
from representations import (represent_heatmap, represent_heatmap_overlaid, 
                   represent_isolines, represent_isolines_superimposed, 
                   represent_hard_selection, represent_soft_selection, grid_layout)
from representations import Saliency, RGBImage


def load_images(test_image_path, test_saliency_map_path):
    """Load test image and saliency map."""
    image = Image.open(test_image_path)
    saliency = np.array(Image.open(test_saliency_map_path))
    return RGBImage(image).image, Saliency(saliency).image

def get_user_inputs():
    """Parse and return user input arguments."""
    input_parser = argparse.ArgumentParser(description="Generate saliency map representations.")
    input_parser.add_argument("--test_image_path", default='Prehispanic_EstructuraVIIIDeEdzna_Campeche_N_1.png', help="Path to test image")
    input_parser.add_argument("--test_saliency_map_path", default='Prehispanic_EstructuraVIIIDeEdzna_Campeche_GFDM_N_1.png', help="Path to test saliency map")
    input_parser.add_argument("--display_type", default='grid', choices=['grid', 'singles'], help="How to display results")
    return input_parser.parse_args()


def generate_representations(saliency, image, colormap='gist_heat'):
    """Generate various saliency map representations."""
    return {
        'Heat Mapped Saliency Image': represent_heatmap(saliency, colormap),
        'Blended Saliency Image': represent_heatmap_overlaid(saliency, image, colormap),
        'Heat Mapped Isoline Image': represent_isolines(saliency, colormap),
        'Blended Isoline Image': represent_isolines_superimposed(saliency, image, colormap),
        'Hard Masked Image': represent_hard_selection(saliency, image, 200),
        'Soft Masked Image': represent_soft_selection(saliency, image)
    }

def display_images(images, display_type):
    """Display images based on the selected display type."""
    if display_type == 'singles':
        for title, img in images.items():
            print(f"Displaying: {title}")
            img.show()
    elif display_type == 'grid':
        grid_layout(list(images.values()), list(images.keys()))

if __name__ == "__main__":
    args = get_user_inputs()

    # Load test image and saliency map
    image, saliency = load_images(args.test_image_path, args.test_saliency_map_path)

    # Generate representations of the saliency map
    representations = generate_representations(saliency, image)

    # Display images based on the display type
    display_images(representations, args.display_type)
