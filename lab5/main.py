

import argparse
from PIL import Image
from skimage.transform import resize
import tensorflow as tf
import cv2
from GRADCAM import *
from FEM import *
from representations import create_heatmap, overlay_heatmap_on_image
from utils import (normalize_matrix, load_model_and_last_layer, get_decode_predictions_function, 
                   get_preprocess_input_function, get_input_image_size, display_grid, 
                   CLASS_NAMES, MASK_SIZE, NUM_MASKS)
#data\African_elephant\ILSVRC2012_val_00001177.JPEG
#ILSVRC2012_val_00013193
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_image_path", default='data\\black_bear\ILSVRC2012_val_00045189.JPEG', help='Path to test image')
    parser.add_argument("--test_image_index", default=1, help='Index: 0 for Elephant, 1 for Bear')
    parser.add_argument("--explanation_method", default='GRADCAM', help='Method: GRADCAM or FEM')
    parser.add_argument("--model_name", default='ResNet', help='Model: ResNet or Xception')
    parser.add_argument("--display_type", default='grid', help='Display layout for results')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Loading arguments
    image_path = args.test_image_path
    image_class_index = int(args.test_image_index)
    display_type = args.display_type
    model_name = args.model_name
    method = args.explanation_method
    colormap = 'turbo'
    selected_class_name = CLASS_NAMES[image_class_index]

    # Load image
    image = Image.open(image_path)
    img_size = image.size

    # Get array of image
    image_array = tf.keras.preprocessing.image.img_to_array(image)

    if method == 'GRADCAM':
        grad_cam_instance = GradCAM(model_name, image_array)
        grad_cam_model = grad_cam_instance.get_model()
        gradients = grad_cam_instance.compute_gradients(grad_cam_model, selected_class_name)
        pooled_gradients = grad_cam_instance.pool_gradients(gradients)
        grad_cam_instance.weight_activation_map(pooled_gradients)
        grad_cam_instance.apply_relu()
        saliency_map = grad_cam_instance.apply_dimension_average_pooling()
        saliency_map = normalize_matrix(saliency_map)
        saliency_map = cv2.GaussianBlur(saliency_map, (0, 0), 0.5)

    elif method == 'FEM':
        model, last_conv_layer_name = load_model_and_last_layer(model_name)
        preprocess_input = get_preprocess_input_function(model_name)
        img_array = tf.expand_dims(preprocess_input(resize(image_array, get_input_image_size(model_name))), axis=0)
        saliency_map = compute_fem(img_array, model, last_conv_layer_name)

    saliency_map = normalize_matrix(saliency_map)

    # Resize the saliency map to match the original image size
    saliency_map = resize(saliency_map, img_size, order=3, mode='wrap', anti_aliasing=False)

    # Generate heatmap and resize it to match the original image size
    heatmap_image = create_heatmap(saliency_map, colormap)
    heatmap_image = heatmap_image.resize(image.size)

    # Overlay heatmap on the original image
    saliency_blended = overlay_heatmap_on_image(saliency_map, image, colormap)

    # Prepare images and titles for display
    images = [heatmap_image, saliency_blended]
    titles = ['Saliency Heatmap', 'Heatmap Overlay on Original Image']

    overall_title = f'{method} Method with {model_name} Model'


    # Display the images in the selected format
    if display_type == 'singles':
        for img in images:
            img.show()
    elif display_type == 'grid':
        display_grid(images, titles, overall_title)
