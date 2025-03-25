import numpy as np
import tensorflow as tf
import keract  # Import for potential activation visualization, not currently used

from utils import get_last_layer_name  # Utility function for obtaining the last convolutional layer name

# Display
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class FEM:
    """
    A class for implementing FEM (Feature Explanation Method) for classification explainability.
    This class provides methods to compute and visualize activation maps
    highlighting the regions most important for a given class prediction.
    """

    def __init__(self, model, model_name, img_array):
        """
        Initialize a FEM instance.

        Args:
            model (tf.keras.Model): The neural network model used for prediction.
            model_name (str): The name of the deep learning model.
            img_array (numpy.ndarray): The input image as a NumPy array.
        """
        self.model = model
        self.model_name = model_name
        self.img_array = img_array

    def expand_flat_values_to_activation_shape(self, values, W_layer, H_layer):
        """
        Expand a 1D array of values to the shape of a neural network activation map.

        Args:
            values (np.ndarray): 1D array of values to be expanded.
            W_layer (int): Width of the activation map.
            H_layer (int): Height of the activation map.

        Returns:
            np.ndarray: An expanded array with the shape (W_layer, H_layer, len(values)).
        """
        # Simplified implementation for expanding values
        expanded = values.reshape((1, 1, -1)) * np.ones((W_layer, H_layer, len(values)))
        return expanded

    def compute_binary_maps(self, feature_map, sigma=None):
        """
        Compute binary maps based on the feature map and thresholding.

        Args:
            feature_map (np.ndarray): The feature map from a convolutional layer.
            sigma (float, optional): Multiplier for the standard deviation threshold. Defaults to 2.

        Returns:
            np.ndarray: The binary maps for each feature channel.
        """
        batch_size, W_layer, H_layer, N_channels = feature_map.shape
        thresholded_tensor = np.zeros((batch_size, W_layer, H_layer, N_channels))
        
        # Set default sigma if not provided
        feature_sigma = sigma if sigma is not None else 2

        # Iterate through each sample in the batch
        for B in range(batch_size):
            # Extract activation map for the current sample
            activation = feature_map[B, :, :, :]

            # Compute mean and standard deviation for each channel
            mean_activation_per_channel = tf.reduce_mean(activation, axis=[0, 1])
            std_activation_per_channel = tf.math.reduce_std(activation, axis=(0, 1))

            # Expand mean and standard deviation to match activation shape
            mean_activation_expanded = tf.reshape(mean_activation_per_channel, (1, 1, -1)) * np.ones((W_layer, H_layer, len(mean_activation_per_channel)))
            std_activation_expanded = tf.reshape(std_activation_per_channel, (1, 1, -1)) * np.ones((W_layer, H_layer, len(std_activation_per_channel)))

            # Compute binary map by thresholding
            thresholded_tensor[B, :, :, :] = tf.cast((activation > (mean_activation_expanded + feature_sigma * std_activation_expanded)), dtype=tf.int32)

        return thresholded_tensor

    def aggregate_binary_maps(self, binary_feature_map, original_feature_map):
        """
        Aggregate binary maps using the original feature map.

        Args:
            binary_feature_map (np.ndarray): Binary maps for each channel.
            original_feature_map (np.ndarray): Original feature map from the convolutional layer.

        Returns:
            np.ndarray: Aggregated feature map normalized to [0, 1].
        """
        batch_size, W_layer, H_layer, N_channels = original_feature_map.shape

        # Use the first sample from the batch
        original_feature_map = original_feature_map[0]
        binary_feature_map = binary_feature_map[0]

        # Compute mean activation weights for each channel
        channel_weights = np.mean(original_feature_map, axis=(0, 1))

        # Expand weights to match the shape of the activation map
        expanded_weights = self.expand_flat_values_to_activation_shape(channel_weights, W_layer, H_layer)

        # Apply the weights to the binary map and aggregate channels
        expanded_feat_map = np.multiply(expanded_weights, binary_feature_map)
        feat_map = np.sum(expanded_feat_map, axis=2)

        # Normalize the aggregated feature map
        if np.max(feat_map) != 0:
            feat_map /= np.max(feat_map)

        return feat_map

    def compute_saliency_map(self):
        """
        Compute the saliency map (FEM) for the input image.

        Returns:
            np.ndarray: Computed saliency map.
        """
        # Get the name of the last convolutional layer
        last_conv_layer_name = get_last_layer_name(self.model_name)

        # Remove the activation function of the last layer for linear outputs
        self.model.layers[-1].activation = None

        # Create a model to output the feature map from the last convolutional layer
        fem_model = tf.keras.models.Model(inputs=self.model.input, outputs=self.model.get_layer(last_conv_layer_name).output)

        # Generate the feature map for the input image
        feature_map = fem_model(self.img_array)

        # Compute binary feature maps
        binary_feature_map = self.compute_binary_maps(feature_map)

        # Aggregate binary maps to create the saliency map
        saliency_map = self.aggregate_binary_maps(binary_feature_map, feature_map)

        return saliency_map
