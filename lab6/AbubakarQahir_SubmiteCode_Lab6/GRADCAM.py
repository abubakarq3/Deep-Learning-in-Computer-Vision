import numpy as np
import tensorflow as tf
import cv2
from skimage.transform import resize
from tensorflow.keras.models import Model
from utils import normalise, get_last_layer_name

class GradCAM:
    """
    GradCAM class for generating activation maps that highlight regions important for class predictions.
    """

    def __init__(self, model, model_name, img_array, class_index):
        """
        Initialize GradCAM instance.

        Args:
            model (tf.keras.Model): The trained deep learning model.
            model_name (str): Name of the model.
            img_array (np.ndarray): Input image array.
            class_index (int): Class index for which saliency is computed.
        """
        self.model = model
        self.model_name = model_name
        self.img_array = img_array
        self.class_index = class_index
        self.weight_activation_maps = None
        self.last_conv_layer_output = None

    def get_model(self):
        """
        Create the GradCAM model by removing the softmax activation and adding the last conv layer.

        Returns:
            Model: The modified GradCAM model.
        """
        self.model.layers[-1].activation = None  # Remove softmax for gradients
        last_layer_name = get_last_layer_name(self.model_name)
        last_conv_layer = self.model.get_layer(last_layer_name)
        return Model(self.model.inputs, [self.model.output, last_conv_layer.output])

    def compute_gradients(self, grad_cam_model):
        """
        Compute gradients of class prediction w.r.t. the last conv layer output.

        Args:
            grad_cam_model (Model): The GradCAM model.

        Returns:
            tf.Tensor: Gradients tensor.
        """
        if self.last_conv_layer_output is not None:
            return self.last_conv_layer_output

        with tf.GradientTape() as tape:
            preds, last_conv_layer_output = grad_cam_model(self.img_array)
            score = preds[0][self.class_index]  # Class prediction score

        gradients = tape.gradient(score, last_conv_layer_output)
        self.last_conv_layer_output = last_conv_layer_output  # Store for reuse
        return gradients

    def pool_gradients(self, gradients):
        """
        Pool gradients globally for each channel.

        Args:
            gradients (tf.Tensor): Gradients tensor.

        Returns:
            list: Pooled gradient values.
        """
        pooled_gradients = [tf.keras.layers.GlobalAveragePooling2D()(
            tf.expand_dims(gradients[:, :, :, i], axis=-1)) for i in range(gradients.shape[-1])]
        return pooled_gradients

    def weight_activation_map(self, pooled_gradients):
        """
        Weight activation maps by corresponding pooled gradient values.

        Args:
            pooled_gradients (list): List of pooled gradient values.
        """
        if self.weight_activation_maps is not None:
            return self.weight_activation_maps

        shape = self.last_conv_layer_output.shape.as_list()[1:]
        weighted_maps = np.empty(shape)

        for i in range(len(pooled_gradients)):
            weighted_maps[:, :, i] = np.squeeze(
                self.last_conv_layer_output.numpy()[:, :, :, i], axis=0) * pooled_gradients[i]

        self.weight_activation_maps = weighted_maps  # Store for reuse

    def apply_relu(self):
        """
        Apply ReLU to set negative values in the weighted maps to zero.
        """
        if self.weight_activation_maps is not None:
            self.weight_activation_maps[self.weight_activation_maps < 0] = 0

    def apply_dimension_average_pooling(self):
        """
        Perform global average pooling along the channel dimension.

        Returns:
            np.ndarray: Averaged saliency map.
        """
        return np.mean(self.weight_activation_maps, axis=2)

    def compute_saliency_map(self):
        """
        Compute the saliency map using GradCAM approach.

        Returns:
            np.ndarray: Normalized and smoothed saliency map.
        """
        grad_cam_model = self.get_model()
        gradients = self.compute_gradients(grad_cam_model)
        pooled_gradients = self.pool_gradients(gradients)
        self.weight_activation_map(pooled_gradients)
        self.apply_relu()
        saliency_map = self.apply_dimension_average_pooling()
        saliency_map = normalise(saliency_map)

        # Apply Gaussian blur for smoothing
        self.saliency_map = cv2.GaussianBlur(saliency_map, (0, 0), 0.5)
        return self.saliency_map
