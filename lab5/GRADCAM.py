# Import necessary libraries and modules
import numpy as np
import tensorflow as tf
from skimage.transform import resize
from tensorflow.keras.models import Model

# Import custom utility functions from 'utils' module
from utils import create_classifier, get_last_conv_layer

class GradCAM:
    """
    Implements GradCAM to compute and visualize activation maps.
    """

    def __init__(self, model_name, img_array):
        # Initialize model and image array
        self.model_name = model_name
        self.img_array = self.resize_array(img_array)
        self.weight_activation_maps = 0
        self.last_conv_layer_output = 0

    def get_decode_predictions(self):
        # Get the decode_predictions function for the specified model
        if self.model_name == 'Xception':
            return tf.keras.applications.xception.decode_predictions
        elif self.model_name == 'ResNet':
            return tf.keras.applications.resnet_v2.decode_predictions

    def get_preprocess_input(self):
        # Get the preprocess_input function for the specified model
        if self.model_name == 'Xception':
            return tf.keras.applications.xception.preprocess_input
        elif self.model_name == 'ResNet':
            return tf.keras.applications.resnet_v2.preprocess_input

    def get_model(self):
        # Create a GradCAM model without softmax activation in the last layer
        model = create_classifier(self.model_name)
        model.layers[-1].activation = None

        # Get the last convolutional layer
        last_layer_name = get_last_conv_layer(self.model_name)

        # Build a model that outputs the last conv layer and predictions
        grad_cam_model = Model(model.inputs, 
                               [model.get_layer(last_layer_name).output, model.output])

        return grad_cam_model

    def resize_array(self, img_array):
        # Resize input image array according to the model's input size
        if self.model_name == 'Xception':
            size = (299, 299)
        elif self.model_name == 'ResNet':
            size = (224, 224)
        return resize(img_array, size)

    def compute_gradients(self, grad_cam_model, class_name):
        # Compute gradients for the target class with respect to the last conv layer
        preprocess_input = self.get_preprocess_input()
        img_array = tf.expand_dims(preprocess_input(self.img_array), axis=0)

        # Use GradientTape to compute gradients
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_cam_model(img_array)
            sorted_preds = tf.sort(preds, direction='DESCENDING')

            decode_predictions = self.get_decode_predictions()
            labels = decode_predictions(np.asarray(preds), top=1000)[0]

            # Find index of the target class
            class_index = [index for index, x in enumerate(labels) if x[1] == class_name][0]

            # Calculate score for the target class
            score = sorted_preds[:, class_index]

        # Compute gradients of the score with respect to last conv layer
        gradients = tape.gradient(score, last_conv_layer_output)
        self.last_conv_layer_output = last_conv_layer_output

        return gradients

    def pool_gradients(self, gradients):
        # Perform global average pooling on the gradients
        pooled_gradients = []
        for channel_index in range(gradients.shape[-1]):
            pooled_value = tf.keras.layers.GlobalAveragePooling2D()(
                tf.expand_dims(gradients[:, :, :, channel_index], axis=-1))
            pooled_gradients.append(pooled_value)

        return pooled_gradients

    def weight_activation_map(self, pooled_gradients):
        # Weight the activation maps by the gradients
        shape = self.last_conv_layer_output.shape.as_list()[1:]
        weighted_maps = np.empty(shape)

        if self.last_conv_layer_output.shape[-1] != len(pooled_gradients):
            print('Error: size mismatch')
        else:
            for i in range(len(pooled_gradients)):
                weighted_maps[:, :, i] = np.squeeze(self.last_conv_layer_output.numpy()
                                                    [:, :, :, i], axis=0) * pooled_gradients[i]

        # Store weighted activation maps
        self.weight_activation_maps = weighted_maps

    def apply_relu(self):
        # Apply ReLU to weighted activation maps
        self.weight_activation_maps[self.weight_activation_maps < 0] = 0

    def apply_dimension_average_pooling(self):
        # Perform average pooling across channels
        return np.mean(self.weight_activation_maps, axis=2)
