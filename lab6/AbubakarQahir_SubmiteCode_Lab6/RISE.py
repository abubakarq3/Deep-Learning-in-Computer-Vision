# RISE.py
import tensorflow as tf
from skimage.transform import resize
import numpy as np
from constants import size  # Ensure this is defined elsewhere

class RISE:
    def __init__(self, model, img_array, class_index, n_masks, mask_size, threshold):
        """
        Initialize the RISE class with model parameters and configurations.

        Args:
            model (tf.keras.Model): The pre-trained model to be used.
            img_array (numpy.ndarray): The input image array.
            class_index (int): The class index for which the saliency map is generated.
            n_masks (int): The number of random masks to generate.
            mask_size (int): The size of each mask.
            threshold (float): The threshold for mask generation.
        """
        self.model = model
        self.img_array = img_array
        self.class_index = class_index
        self.n_masks = n_masks
        self.mask_size = mask_size
        self.threshold = threshold

        self.perturbed_images = None  # Store generated perturbed images
        self.masks = None  # Store generated masks
        self.scores = None  # Store prediction scores

    def generate_masks(self):
        """
        Generates a set of random binary masks for the input image, applies the masks to the image,
        and creates a set of perturbed images.

        Returns:
            tuple: The perturbed images and corresponding masks.
        """
        if self.perturbed_images is not None and self.masks is not None:
            return self.perturbed_images, self.masks  # Return if already generated

        H, W = size  # Get image dimensions
        masks = np.empty((self.n_masks, H, W))  # Initialize masks array
        perturbed_images = np.empty((self.n_masks, H, W, 3))  # Initialize perturbed images array

        # Generate random masks and create perturbed images
        for i in range(self.n_masks):
            grid = (np.random.rand(1, self.mask_size, self.mask_size) < self.threshold).astype("float32")

            # Resize mask to image size
            C_H, C_W = np.ceil(H / self.mask_size), np.ceil(W / self.mask_size)
            h_new_mask, w_new_mask = (self.mask_size + 1) * C_H, (self.mask_size + 1) * C_W
            x, y = np.random.randint(0, C_H), np.random.randint(0, C_W)

            masks[i, :, :] = resize(
                grid[0],
                (h_new_mask, w_new_mask),
                order=1,
                mode="reflect",
                anti_aliasing=False,
            )[x: x + H, y: y + W]

            # Create 3-channel mask and apply it to the image
            mask_3d = masks[i, :, :][..., None].repeat(3, axis=2)
            perturbed_images[i, :, :, :] = mask_3d * self.img_array

        # Store generated masks and perturbed images
        self.perturbed_images = perturbed_images
        self.masks = masks

    def obtain_prediction_scores(self):
        """
        Computes prediction scores for all perturbed images using the pre-trained model.

        Returns:
            list: List of prediction scores for the specified class index.
        """
        if self.scores is not None:
            return self.scores  # Return if already computed

        scores = []
        for image_index in range(self.perturbed_images.shape[0]):
            image = self.perturbed_images[image_index, :, :, :]
            image = tf.expand_dims(image, axis=0)  # Add batch dimension

            # Predict and retrieve score for the specified class
            predictions = self.model.predict(image).flatten()
            score = predictions[self.class_index]
            scores.append(score)
            print(score)  # Optional: Print score for debugging

        # Store computed scores
        self.scores = scores

    def weight_saliency_maps(self):
        """
        Weights and aggregates saliency maps based on prediction scores.

        Returns:
            numpy.ndarray: The aggregated saliency map.
        """
        sum_of_scores = np.sum(self.scores)
        saliency_map = np.zeros(self.masks[0].shape, dtype=np.float64)

        # Aggregate saliency map by weighting each mask with its corresponding score
        for i, mask_i in enumerate(self.masks):
            score_i = self.scores[i]
            saliency_map += score_i * mask_i

        saliency_map /= sum_of_scores  # Normalize by the sum of scores
        return saliency_map

    def compute_saliency_map(self):
        """
        Computes the saliency map by generating masks, obtaining prediction scores, and
        aggregating the weighted saliency maps.

        Returns:
            numpy.ndarray: The final computed saliency map.
        """
        self.generate_masks()  # Generate masks and perturbed images
        self.obtain_prediction_scores()  # Compute prediction scores
        return self.weight_saliency_maps()  # Return the weighted saliency map
