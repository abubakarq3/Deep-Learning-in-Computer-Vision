

# Rendering Parameters for LIME Visualization
positive_only = True         # Display only features that positively impact the prediction
negative_only = False        # Display only features that negatively impact the prediction
num_superpixels = 15         # Number of superpixels to display in the visualization
hide_rest = True             # If True, hides areas not part of the selected superpixels


# Constants for RISE Method
low_res_mask_size = 8        # Size of the low-resolution masks for RISE
mask_number = 200            # Number of random masks to be generated
threshold = 0.4              # Threshold for determining mask binary values
size = (224, 224)            # Target input size for the images

# Constants for LIME Method
top_labels = 1               # Number of top labels to consider in the explanation
hide_color = [0, 0, 0]       # Color used to hide parts of the image; None uses the average color
num_lime_features = 100000#100#100000   # Number of feature groups in the LIME explanation
num_samples = 2000#3000 #2000          # Number of perturbed samples generated for LIME analysis
rand_index = 0               # Index for selecting specific color schemes for hidden areas

