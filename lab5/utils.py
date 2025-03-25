import tensorflow as tf
from skimage.transform import resize
import numpy as np
import os
import random
import matplotlib.pyplot as plt

# Constants
CLASS_NAMES = ['African_elephant', 'American_black_bear']
MASK_SIZE = 7
NUM_MASKS = 20

# Normalize a matrix by dividing by its max value
def normalize_matrix(matrix):
    max_value = np.max(matrix)
    return matrix / max_value

# Get the name of the last convolutional layer based on the model
def get_last_conv_layer(model_name):
    if model_name == 'Xception':
        return 'block14_sepconv2_act'
    elif model_name == 'ResNet':
        return 'conv5_block3_out'

# Load model and return the last convolutional layer name
def load_model_and_last_layer(model_name):
    model = create_classifier(model_name)
    model.layers[-1].activation = None  # Remove softmax
    return model, get_last_conv_layer(model_name)

# Get the decode_predictions function based on the model architecture
def get_decode_predictions_function(model_name):
    if model_name == 'Xception':
        return tf.keras.applications.xception.decode_predictions
    elif model_name == 'ResNet':
        return tf.keras.applications.resnet_v2.decode_predictions

# Get the preprocess_input function based on the model architecture
def get_preprocess_input_function(model_name):
    if model_name == 'Xception':
        return tf.keras.applications.xception.preprocess_input
    elif model_name == 'ResNet':
        return tf.keras.applications.resnet_v2.preprocess_input

# Get the input image size for the specified model architecture
def get_input_image_size(model_name):
    if model_name == 'Xception':
        return (299, 299)
    elif model_name == 'ResNet':
        return (224, 224)

# Create a pre-trained classifier model for the backbone architecture
def create_classifier(backbone):
    if backbone == 'Xception':
        return tf.keras.applications.xception.Xception(weights="imagenet", classifier_activation="softmax")
    elif backbone == 'ResNet':
        return tf.keras.applications.resnet_v2.ResNet50V2(weights="imagenet", classifier_activation="softmax")

# Display images in a grid with titles
def display_grid(images, titles, overall_title):
    plt.subplot(121)
    plt.imshow(images[0])
    plt.title(titles[0])

    plt.subplot(122)
    plt.imshow(images[1])
    plt.title(titles[1])

    plt.tight_layout(pad=2.0)

    os.makedirs('output_images', exist_ok=True)
    save_index = random.randint(1, 100)
    plt.savefig(f'output_images/{save_index}.jpg')
    plt.show()
