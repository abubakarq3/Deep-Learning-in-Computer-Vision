import numpy as np
import tensorflow as tf
import keract
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Load an image and convert it to a numpy array
def get_img_array(img_path, size):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    array = tf.keras.preprocessing.image.img_to_array(img)
    array = np.expand_dims(array, axis=0)  # Add batch dimension
    return array

# Expand values to match the activation map shape
def expand_flat_values_to_activation_shape(values, W_layer, H_layer):
    expanded = values.reshape((1, 1, -1)) * np.ones((W_layer, H_layer, len(values)))
    return expanded

# Compute binary maps from the feature map
def compute_binary_maps(feature_map, sigma=2):
    batch_size, W_layer, H_layer, N_channels = feature_map.shape
    thresholded_tensor = np.zeros((batch_size, W_layer, H_layer, N_channels))

    for B in range(batch_size):
        activation = feature_map[B, :, :, :]
        mean_activation = activation.mean(axis=(0, 1))  # Mean per channel
        std_activation = activation.std(axis=(0, 1))    # Std per channel

        mean_expanded = expand_flat_values_to_activation_shape(mean_activation, W_layer, H_layer)
        std_expanded = expand_flat_values_to_activation_shape(std_activation, W_layer, H_layer)

        # Apply threshold to generate binary map
        thresholded_tensor[B, :, :, :] = 1.0 * (activation > (mean_expanded + sigma * std_expanded))

    return thresholded_tensor

# Aggregate binary maps using the original feature map
def aggregate_binary_maps(binary_feature_map, original_feature_map):
    batch_size, W_layer, H_layer, N_channels = original_feature_map.shape

    original_feature_map = original_feature_map[0]
    binary_feature_map = binary_feature_map[0]

    # Calculate channel weights as mean values
    channel_weights = np.mean(original_feature_map, axis=(0, 1))
    expanded_weights = expand_flat_values_to_activation_shape(channel_weights, W_layer, H_layer)

    # Apply weights to the binary feature maps and aggregate
    weighted_map = np.multiply(expanded_weights, binary_feature_map)
    feature_map_sum = np.sum(weighted_map, axis=2)  # Sum over channels

    # Normalize the feature map
    if np.max(feature_map_sum) != 0:
        feature_map_sum = feature_map_sum / np.max(feature_map_sum)
    return feature_map_sum

# Compute Feature Extraction Maps (FEM)
def compute_fem(img_array, model, last_conv_layer_name):
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    activations = keract.get_activations(model, img_array, auto_compile=True)
    feature_map = activations.get(last_conv_layer_name)
    
    # Compute binary maps and aggregate them to get saliency
    binary_feature_map = compute_binary_maps(feature_map)
    saliency = aggregate_binary_maps(binary_feature_map, feature_map)
    return saliency
