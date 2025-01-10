import os
import gc
from PIL import Image
import numpy as np
from tqdm import tqdm

class PatchExtractor:
    def __init__(self, patch_size, step_size, output_dir):
        """
        Initializes the PatchExtractor instance.

        Args:
            patch_size (tuple): (height, width) of each patch.
            step_size (tuple): Step size (vertical, horizontal) for sliding window.
            output_dir (str): Directory to save patches.
        """
        self.patch_size = patch_size
        self.step_size = step_size
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def extract_patches(self, image_path):
        """
        Extracts patches from an image using a sliding window approach.

        Args:
            image_path (str): Path to the image file.

        Returns:
            List of extracted patch filenames.
        """
        img = Image.open(image_path)  # Open the image
        img = np.array(img)  # Convert to NumPy array

        patches = []
        patch_height, patch_width = self.patch_size
        step_height, step_width = self.step_size
        img_height, img_width = img.shape[:2]  # Get the image dimensions

        patch_idx = 0  # Counter for patches
        for y in range(0, img_height, step_height):
            for x in range(0, img_width, step_width):
                # Extract patch
                patch = img[y:y + patch_height, x:x + patch_width]

                # Skip incomplete patches at the edges
                if patch.shape[0] != patch_height or patch.shape[1] != patch_width:
                    continue

                # Save the patch
                patch_name = f"{os.path.basename(image_path).split('.')[0]}_patch_{patch_idx}.png"
                patch_path = os.path.join(self.output_dir, patch_name)
                Image.fromarray(patch).save(patch_path)
                patches.append(patch_name)
                patch_idx += 1

        return patches

    def process_dataset(self, dataset_df, image_col, label_col, target_col):
        """
        Extracts patches for all images in a dataset.

        Args:
            dataset_df (pd.DataFrame): DataFrame containing image paths and labels.
            image_col (str): Column name for image paths.
            label_col (str): Column name for labels.
            target_col (str): Column name for targets.

        Returns:
            List of dictionaries mapping patch data to their labels and targets.
        """
        patches_data = []

        for idx, row in tqdm(dataset_df.iterrows(), total=dataset_df.shape[0], desc="Extracting Patches"):
            image_path = row[image_col]
            label = row[label_col]
            target = row[target_col]
            try:
                # Extract patches and save them
                patch_files = self.extract_patches(image_path)

                # Map patches to their labels
                for patch_file in patch_files:
                    patches_data.append({
                        'patch_name': patch_file,
                        'original_image': os.path.basename(image_path),
                        'label': label,
                        'target': target
                    })

                # Garbage collection to free memory
                gc.collect()

            except Exception as e:
                print(f"Error processing {image_path}: {e}")

        return patches_data
