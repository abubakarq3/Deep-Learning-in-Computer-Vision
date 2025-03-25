import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from utils import load_image, load_fixations, load_ground_truth, calculate_error_metrics, generate_saliency_map, normalise

def main(images_folder_path, fixations_folder_path, gdfm_folder_path, output_folder):
    errors = {}
    
    # Ensure output directory exists
    os.makedirs(output_folder + '/GFDM', exist_ok=True)
    os.makedirs(output_folder + '/blended', exist_ok=True)

    for filename in os.listdir(images_folder_path):
        # Construct filenames for fixation and GFDM files
        image_name = os.path.splitext(filename)[0]
        fixation_filename = filename.replace("_N_", "_GazeFix_N_").replace(".png", ".txt")
        gdfm_filename = filename.replace("_N_", "_GFDM_N_")

        # Construct file paths
        file_image_path = os.path.join(images_folder_path, filename)
        file_fixation_path = os.path.join(fixations_folder_path, fixation_filename)
        file_gdfm_path = os.path.join(gdfm_folder_path, gdfm_filename)

        # Load data
        image = load_image(file_image_path)
        fixation_points = load_fixations(file_fixation_path)
        gdfm_ground_truth = load_ground_truth(file_gdfm_path)

        # Generate saliency map
        saliency_map = generate_saliency_map(image, fixation_points)

        # Save the grayscale saliency map
        saliency_image_grey = Image.fromarray((saliency_map * 255).astype(np.uint8))
        saliency_image_grey.save(f'{output_folder}/GFDM/{filename}')

        # Generate a colored version of the saliency map
        colormap = plt.get_cmap('jet')
        saliency_map_colored = (colormap(saliency_map) * 255).astype(np.uint8)
        saliency_image_colored = Image.fromarray(saliency_map_colored)

        # Blend the saliency map with the original image
        blended_image = Image.blend(image.convert('RGBA'), saliency_image_colored.convert('RGBA'), alpha=0.4)  
        blended_image.save(f'{output_folder}/blended/{filename}')

        # Calculate error metrics
        metrics = calculate_error_metrics(normalise(gdfm_ground_truth), normalise(saliency_image_grey))
        errors[image_name] = metrics

    # Output average metrics
    average_mae = np.mean([error['MAE'] for error in errors.values()])
    print("Average MAE:", average_mae)   # note: i can  also calculate other error too like MSE ,PCC or SSIM

if __name__ == "__main__":
    images_folder_path = r'E:\Bordo\DL\Lab2\MexCulture142\images_train'
    fixations_folder_path = r'E:\Bordo\DL\Lab2\MexCulture142\fixations'
    gdfm_folder_path = r'E:\Bordo\DL\Lab2\MexCulture142\gazefixationsdensitymaps'
    output_folder = 'output_result'

     #'/net/ens/DeepLearning/DLCV2024/MexCulture142/fixations'
     #'/net/ens/DeepLearning/DLCV2024/MexCulture142/gazefixationsdensitymaps'

    main(images_folder_path, fixations_folder_path, gdfm_folder_path, output_folder)
