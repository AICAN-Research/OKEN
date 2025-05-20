import os
import json
import argparse
import numpy as np
import pandas as pd
from cucim import CuImage
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import filters, morphology
from scipy.ndimage import binary_fill_holes
import onnxruntime as ort
import random

class WSIProcessor:
    def __init__(self, 
                 test_json_file_path, 
                 train_json_file_path,
                 store_tumor_only=False,
                 model_path=None,
                 p=0.5,
                 classifier_level=0):
        self.test_json_file_path = test_json_file_path
        self.train_json_file_path = train_json_file_path
        self.test_data = {}
        self.train_data = {}

        # For tumor-only storing
        self.store_tumor_only = store_tumor_only
        self.model_path = model_path
        self.p = p
        self.classifier_level = classifier_level

        # If we are storing tumor only, load the ONNX model session
        if self.store_tumor_only:
            if not self.model_path:
                raise ValueError("You must provide a valid --model_path when using --store_tumor_only.")
            self.ort_session = ort.InferenceSession(self.model_path)

    def create_resized_thumbnail(self, image_path, scale_factor=0.01):
        """
        Creates a reduced-size thumbnail of the WSI at a chosen level, 
        then resizes it to (scale_factor * full_dimensions).
        """
        img = CuImage(image_path)
        level_count = len(img.resolutions['level_dimensions'])
        # Choose an appropriate thumbnail level (avoid reading the largest resolution)
        thumbnail_level = min(level_count - 1, 7)
        thumbnail_dimensions = img.resolutions['level_dimensions'][thumbnail_level]
        
        thumbnail_region = img.read_region(location=(0, 0), level=thumbnail_level, size=thumbnail_dimensions)
        thumbnail_array = np.array(thumbnail_region)
        thumbnail = Image.fromarray(thumbnail_array)

        original_dimensions = img.resolutions['level_dimensions'][0]  # The dimensions at level 0 (full res)
        new_width = int(original_dimensions[0] * scale_factor)
        new_height = int(original_dimensions[1] * scale_factor)
        resized_thumbnail = thumbnail.resize((new_width, new_height), Image.LANCZOS)
        return resized_thumbnail.convert("RGB")

    def process_wsi(self, 
                    file_path, 
                    thumbnail_patch_size, 
                    scale_factor=0.01, 
                    tissue_threshold=0.5, 
                    save_dir='./Data', 
                    data_type='Test'):
        """
        Main function that:
        1) Creates a resized thumbnail
        2) Segments tissue using a gradient-based approach (binary mask)
        3) Generates patch positions from the binary mask
        4) (Optionally) Filters these patches with an ONNX classifier
        5) Saves an annotated thumbnail
        6) Stores results internally for JSON export
        """
        # 1) Create a resized thumbnail
        resized_thumbnail = self.create_resized_thumbnail(file_path, scale_factor)

        # 2) Get the tissue segmentation mask
        binary_mask = self.tissue_segmentation(resized_thumbnail)

        # 3) Generate patch positions from the mask
        patch_positions = self.generate_patches(resized_thumbnail, 
                                                binary_mask, 
                                                thumbnail_patch_size, 
                                                tissue_threshold)

        # Convert thumbnail-level positions to original resolution
        original_patch_positions = [(int(x / scale_factor), int(y / scale_factor)) 
                                    for x, y in patch_positions]
        original_patch_size = thumbnail_patch_size / scale_factor

        # 4) If store_tumor_only is True, filter these positions with the ONNX classifier
        if self.store_tumor_only:
            original_patch_positions = self.filter_tumor_patches(file_path, 
                                                                 original_patch_positions, 
                                                                 int(original_patch_size))

            # Convert back to thumbnail scale for drawing
            # Because now we only want to show the tumor patches in the thumbnail
            patch_positions = [(int(x * scale_factor), int(y * scale_factor)) 
                               for (x, y) in original_patch_positions]

        # Save results to the appropriate dictionary
        if data_type == 'Test':
            self.test_data[file_path] = {
                'patch_positions': patch_positions,
                'original_patch_positions': original_patch_positions,
                'original_patch_size': original_patch_size
            }
        else:
            self.train_data[file_path] = {
                'patch_positions': patch_positions,
                'original_patch_positions': original_patch_positions,
                'original_patch_size': original_patch_size
            }

        # 5) Save thumbnail with bounding boxes
        self.save_thumbnail_with_patches(file_path, 
                                         resized_thumbnail, 
                                         patch_positions, 
                                         thumbnail_patch_size, 
                                         save_dir)
        print(f"Processed {file_path} and saved results.")

    def tissue_segmentation(self, resized_thumbnail):
        """
        Perform tissue segmentation using a gradient-based approach.
        """
        gray = np.array(resized_thumbnail.convert('L'))
        gradient_magnitude = filters.sobel(gray)
        gradient_threshold = filters.threshold_otsu(gradient_magnitude)
        binary_mask = gradient_magnitude > gradient_threshold
        binary_mask = morphology.dilation(binary_mask, morphology.disk(2))
        binary_mask = morphology.erosion(binary_mask, morphology.disk(2))
        binary_mask = binary_fill_holes(binary_mask)
        return binary_mask.astype(np.uint8)

    def generate_patches(self, 
                         resized_thumbnail, 
                         binary_mask, 
                         thumbnail_patch_size, 
                         tissue_threshold=0.5):
        """
        Slide a window across the thumbnail, and keep positions that contain 
        enough 'tissue' based on tissue_threshold.
        """
        rows, cols = binary_mask.shape
        stride = thumbnail_patch_size
        patch_positions = []
        half_patch_size = thumbnail_patch_size // 2
        tissue_area_required = thumbnail_patch_size * thumbnail_patch_size * tissue_threshold

        for y in range(half_patch_size, rows - half_patch_size, stride):
            for x in range(half_patch_size, cols - half_patch_size, stride):
                patch_area = binary_mask[y - half_patch_size : y + half_patch_size,
                                         x - half_patch_size : x + half_patch_size]
                if np.sum(patch_area) >= tissue_area_required:
                    patch_positions.append((x - thumbnail_patch_size/2, 
                                            y - thumbnail_patch_size/2))
        return patch_positions

    def filter_tumor_patches(self, wsi_path, original_patch_positions, original_patch_size):
        """
        Use the ONNX classifier to filter out patches that are below a certain 
        probability threshold for tumor (class=1). Keep only tumor patches.
        """
        # Open the WSI once
        slide = CuImage(wsi_path)

        # Depending on your workflow, you can read from classifier_level
        division_factor = 2 ** self.classifier_level
        # Patch size at 'classifier_level'
        adjusted_patch_size = original_patch_size // division_factor

        filtered_positions = []
        for (x, y) in original_patch_positions:
            # Read the patch at the given classifier level
            patch_img = self.read_patch(slide, 
                                        x, 
                                        y, 
                                        int(adjusted_patch_size), 
                                        int(adjusted_patch_size), 
                                        self.classifier_level)
            # Classify
            prob_class_1 = self.classify_patch_onnx(patch_img)
            if prob_class_1 > self.p:
                filtered_positions.append((x, y))
        return filtered_positions

    def read_patch(self, slide, x, y, width, height, level):
        """
        Read a specific region from a slide (similar to your second code).
        Returns a 224x224 patch ready for classification.
        """
        region = slide.read_region(location=(x, y), size=(width, height), level=level)
        patch = np.array(region)
        patch = Image.fromarray(patch)
        # Resize to the model's expected dimension (224, 224)
        patch = patch.resize((224, 224), Image.LANCZOS)
        patch = np.array(patch)
        return patch

    def classify_patch_onnx(self, patch):
        """
        Classify a single patch using the loaded ONNX model session. 
        Returns the probability for class=1 (tumor).
        """
        # Prepare model input
        patch_input = np.expand_dims(patch, axis=0).astype(np.float32) / 255.0
        ort_inputs = {self.ort_session.get_inputs()[0].name: patch_input}
        ort_outs = self.ort_session.run(None, ort_inputs)
        # Probability of class=1 is ort_outs[0][0][1]
        return ort_outs[0][0][1]

    def save_thumbnail_with_patches(self, file_path, resized_thumbnail, patch_positions, 
                                    thumbnail_patch_size, save_dir='./Data'):
        """
        Save an image of the thumbnail with red bounding boxes drawn around each patch.
        """
        thumbnails_save_path = os.path.join(save_dir, 'PatchThumbnails', 
                                            os.path.basename(os.path.dirname(file_path)))
        os.makedirs(thumbnails_save_path, exist_ok=True)

        fig, ax = plt.subplots(1)
        ax.imshow(resized_thumbnail)
        for (x, y) in patch_positions:
            rect = patches.Rectangle((x, y), 
                                     thumbnail_patch_size, 
                                     thumbnail_patch_size, 
                                     linewidth=1, 
                                     edgecolor='r', 
                                     facecolor='none')
            ax.add_patch(rect)
        plt.axis("off")

        thumbnail_path = os.path.join(thumbnails_save_path, 
                                      os.path.basename(file_path) + '_thumbnail.png')
        plt.savefig(thumbnail_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    def save_positions_to_json(self):
        with open(self.test_json_file_path, 'w') as f:
            json.dump(self.test_data, f, indent=4)
        print(f"Test patch positions saved to {self.test_json_file_path}.")

        with open(self.train_json_file_path, 'w') as f:
            json.dump(self.train_data, f, indent=4)
        print(f"Training patch positions saved to {self.train_json_file_path}.")


def main():
    parser = argparse.ArgumentParser(description='Process WSI images and generate patch positions.')
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='Path to the Data directory containing Test and Training folders')
    parser.add_argument('--thumbnail_patch_size', type=int, default=9, 
                        help='Size of the thumbnail patch')
    parser.add_argument('--scale_factor', type=float, default=0.01, 
                        help='Scale factor for the thumbnail')
    parser.add_argument('--tissue_threshold', type=float, default=0.01, 
                        help='Threshold for tissue detection in patches')
    parser.add_argument('--save_dir', type=str, default='./Data', 
                        help='Base directory to save the patch positions and thumbnails')
    parser.add_argument('--test_json', type=str, default='test_positions.json', 
                        help='Filename for the Test JSON file')
    parser.add_argument('--train_json', type=str, default='train_positions.json', 
                        help='Filename for the Training JSON file')

    # New arguments for tumor-only approach
    parser.add_argument('--store_tumor_only', action='store_true', 
                        help='If set, only store patches classified as tumor using the ONNX model.')
    parser.add_argument('--model_path', type=str, default=None, 
                        help='Path to the ONNX model for tumor classification')
    parser.add_argument('--p', type=float, default=0.5, 
                        help='Probability threshold for tumor classification')
    parser.add_argument('--classifier_level', type=int, default=0, 
                        help='WSI level at which to read patches for classification (0 = full res).')

    args = parser.parse_args()

    # Check folder structure
    if not os.path.isdir(os.path.join(args.data_dir, 'Test')) or \
       not os.path.isdir(os.path.join(args.data_dir, 'Training')):
        print("Error: The Data directory must contain 'Test' and 'Training' folders.")
        return

    # Create processor with tumor-only parameters
    processor = WSIProcessor(
        test_json_file_path=args.test_json,
        train_json_file_path=args.train_json,
        store_tumor_only=args.store_tumor_only,
        model_path=args.model_path,
        p=args.p,
        classifier_level=args.classifier_level
    )

    # Process each tif file in Test and Training directories
    for folder in ['Test', 'Training']:
        folder_path = os.path.join(args.data_dir, folder)
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.tif'):
                file_path = os.path.join(folder_path, file_name)
                processor.process_wsi(
                    file_path=file_path,
                    thumbnail_patch_size=args.thumbnail_patch_size,
                    scale_factor=args.scale_factor,
                    tissue_threshold=args.tissue_threshold,
                    save_dir=args.save_dir,
                    data_type=folder
                )

    # Save final JSON results
    processor.save_positions_to_json()


if __name__ == '__main__':
    main()
