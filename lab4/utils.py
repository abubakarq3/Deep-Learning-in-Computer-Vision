import tensorflow as tf
from skimage.transform import resize
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from PIL import Image


classes = ['African_elephant', 'black_bear']
low_res_mask_size =7 # 8 # 
mask_number = 20 #16 

# Utility functions
def normalise(matrix):
    """
    Normalizes a matrix by dividing it by its maximum value.
    """
    max_value = np.max(matrix)
    return matrix / max_value

def generate_masks(image, n_masks, mask_size, threshold):
    """
    Generates a set of random masks for an image.
    """
    H, W, _ = image.shape
    image = resize(image, (min(H, W), min(H, W)))
    H, W, _ = image.shape

    upsampled_H = int((mask_size + 1) * (H / mask_size))
    upsampled_W = int((mask_size + 1) * (W / mask_size))

    diff_H = upsampled_H - H
    diff_W = upsampled_W - W

    masks = np.empty((n_masks, H, W))
    perturbed_images = np.empty((n_masks, H, W, 3))

    for i in range(n_masks):
        mask = (np.random.rand(mask_size, mask_size) >= threshold).astype('int')

        peturbed_x_origin = random.randint(0, diff_W)
        peturbed_y_origin = random.randint(0, diff_H)

        masks[i, :, :] = resize(mask, (upsampled_H, upsampled_W), order=1,
                                mode='reflect', anti_aliasing=False)[peturbed_x_origin:peturbed_x_origin + W, peturbed_y_origin:peturbed_y_origin + H]

        masks[i, :, :] = normalise(masks[i, :, :])

        mask_3d = masks[i, :, :][..., None].repeat(3, axis=2)
        pertubed_image = mask_3d * image
        perturbed_images[i, :, :, :] = pertubed_image

    return [perturbed_images, masks]

def make_prediction(model, model_name, perturbed_images, class_name):
    """
    Make predictions using a pre-trained deep learning model.
    """
    scores = []

    if model_name == 'Xception':
        preprocess_input = tf.keras.applications.xception.preprocess_input
        decode_predictions = tf.keras.applications.xception.decode_predictions
        size = (299, 299)

    elif model_name == 'ResNet':
        preprocess_input = tf.keras.applications.resnet_v2.preprocess_input
        decode_predictions = tf.keras.applications.resnet_v2.decode_predictions
        size = (224, 224)

    for image_index in range(perturbed_images.shape[0]):
        image = perturbed_images[image_index, :, :, :]
        image = resize(image, size)
        img_array = tf.expand_dims(preprocess_input(image), axis=0)

        predictions = model.predict(img_array).flatten()
        labels = decode_predictions(np.asarray([predictions]), top=1000)[0]
        score = next((label[2] for label in labels if label[1] == class_name))

        scores.append(score)

    return scores, labels

def make_classifier(backbone):
    """
    Create a pre-trained deep learning classifier model.
    """
    if backbone == 'Xception':
        model_builder = tf.keras.applications.xception.Xception
        model = model_builder(weights="imagenet", classifier_activation="softmax")

    elif backbone == 'ResNet':
        model_builder = tf.keras.applications.resnet_v2.ResNet50V2
        model = model_builder(weights="imagenet", classifier_activation="softmax")

    return model

def calculate_saliency_map(scores, masks):
    """
    Calculate a saliency map based on scores and masks.
    """
    sum_of_scores = np.sum(scores)
    saliency_map = np.zeros(masks[0].shape, dtype=np.float64)
    for i, mask_i in enumerate(masks):
        score_i = scores[i]
        saliency_map += score_i * mask_i

    saliency_map /= sum_of_scores
    return saliency_map

def represent_heatmap(saliency, cmap='gray'):
    """
    Represent saliency map as a heatmap.
    """
    if np.max(saliency) > 1:
        saliency = normalise(saliency)

    colormap = plt.get_cmap(cmap)
    heatmapped_saliency = (colormap(saliency) * 255).astype(np.uint8)

    return Image.fromarray(heatmapped_saliency)

def represent_heatmap_overlaid(saliency, image, cmap):
    """
    Overlay a saliency map onto the original image.
    """
    heatmapped_saliency = represent_heatmap(saliency, cmap)
    heatmapped_saliency = heatmapped_saliency.resize(image.size)

    return Image.blend(image.convert('RGBA'), heatmapped_saliency.convert('RGBA'), alpha=0.5)

def grid_layout(images, titles):
    """
    Display images in a grid layout.
    """
    # plt.suptitle('Grid layout of Results', fontsize=16)
    # plt.figure(figsize=(12, 8))

    plt.subplot(121)
    plt.imshow(images[0])
    plt.title(titles[0])

    plt.subplot(122)
    plt.imshow(images[1])
    plt.title(titles[1])

    plt.tight_layout(pad=2.0)

    os.makedirs('outpu_Result_images', exist_ok=True)
    save_index = random.randint(1, 100)
    plt.savefig(f'outpu_Result_images/{save_index}.jpg')

    plt.show()

