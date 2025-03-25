# Import necessary libraries for LIME saliency generation
from skimage.segmentation import mark_boundaries
from lime import lime_image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def get_lime_explanation(model, img_array, pred_index, top_labels, hide_color, num_lime_features, num_samples):
    """
    Generate LIME explanation for a given image and model.

    Args:
        model (tf.keras.Model): The trained model to be explained.
        img_array (numpy.ndarray): The input image array.
        pred_index (int): The predicted class index.
        top_labels (int): Number of top labels for LIME to consider.
        hide_color (tuple/int): Color to hide the superpixels.
        num_lime_features (int): Number of LIME features.
        num_samples (int): Number of samples for LIME to generate.

    Returns:
        explanation (LimeImageExplainer): The LIME explanation object.
    """
    # Create LIME image explainer with a fixed random state for reproducibility
    explainer = lime_image.LimeImageExplainer(random_state=0)

    # Convert image array to float type
    img_array = img_array.numpy().astype(np.float64)

    # Generate LIME explanation
    explanation = explainer.explain_instance(
        img_array,
        model.predict,
        top_labels=top_labels,
        labels=(pred_index,),
        hide_color=hide_color,
        num_features=num_lime_features,
        num_samples=num_samples,
        random_seed=0  # Ensure consistent results
    )

    return explanation

def explain_with_lime(model, img_array, 
                      top_labels, hide_color, num_lime_features, num_samples,  # Explanation parameters
                      positive_only, negative_only, num_superpixels, hide_rest,  # Rendering parameters
                      rand_index):  # Hidden color choice
    """
    Generate and visualize LIME explanation for an image.

    Args:
        model (tf.keras.Model): The trained model.
        img_array (numpy.ndarray): The input image array.
        top_labels (int): Number of top labels to consider.
        hide_color (tuple/int): Color used for hiding superpixels.
        num_lime_features (int): Number of LIME features.
        num_samples (int): Number of samples for LIME to use.
        positive_only (bool): Show only positive regions.
        negative_only (bool): Show only negative regions.
        num_superpixels (int): Number of superpixels for visualization.
        hide_rest (bool): Hide non-explained parts of the image.
        rand_index (int): Index for hidden color selection.

    Returns:
        heatmap (numpy.ndarray): Heatmap generated from LIME explanation.
    """
    # Set hidden color based on provided index
    hidden_colour = ''
    if rand_index == 0:
        hidden_colour = 'No'
    elif rand_index == 1:
        hidden_colour = 'Red'
    elif rand_index == 2:
        hidden_colour = 'Green'
    elif rand_index == 3:
        hidden_colour = 'Blue'

    # Predict and find the class index with the highest probability
    preds = model.predict(img_array).flatten()
    pred_index = np.argmax(preds)

    # Get LIME explanation
    explanation = get_lime_explanation(model, img_array[0], pred_index, top_labels, hide_color, num_lime_features, num_samples)

    # Extract the top label and create a heatmap from the explanation
    index = explanation.top_labels[0]
    dict_heatmap = dict(explanation.local_exp[index])
    heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)

    # Generate image and mask for visualization
    temp, mask = explanation.get_image_and_mask(
        label=pred_index,
        positive_only=positive_only,
        negative_only=negative_only,
        num_features=num_superpixels,
        hide_rest=hide_rest
    )

    return heatmap
