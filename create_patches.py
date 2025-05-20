import os
import argparse
import json
import numpy as np
import pandas as pd
from PIL import Image
import cucim
import onnxruntime as ort

def read_patch(slide, x, y, width, height, level):
    """Read a specific region from a slide."""
    region = slide.read_region(location=(x, y), size=(width, height), level=level)
    patch = np.array(region)
    patch = Image.fromarray(patch)
    patch = patch.resize((224, 224), Image.LANCZOS)
    patch = np.array(patch)
    return patch

def load_data_generator(json_path, patch_size, level):
    division_factor = 2 ** level  # Compute the division factor based on the level

    with open(json_path, 'r') as file:
        data = json.load(file)

    for image_path, details in data.items():
        print('Starting on', image_path)
        original_patch_positions = details['original_patch_positions']
        
        # Load the slide once per image
        slide = cucim.CuImage(image_path)

        for pos in original_patch_positions:
            x, y = pos
            adjusted_patch_size = patch_size // division_factor
            patch = read_patch(slide, x, y, adjusted_patch_size, adjusted_patch_size, level)
            yield patch, (x, y), image_path

def classify_and_save_patches(generator, model_path, output_dir, csv_path, p):
    # Load the CSV to match image paths to classes
    df = pd.read_csv(csv_path)

    # Load the ONNX model
    ort_session = ort.InferenceSession(model_path)

    # Ensure output directories exist
    os.makedirs(output_dir, exist_ok=True)
    for class_label in df['Class'].unique():
        os.makedirs(os.path.join(output_dir, str(class_label)), exist_ok=True)

    image_patch_count = {}

    for patch, (x, y), image_path in generator:
        # Ensure we do not save more than 10 patches per image
        if image_patch_count.get(image_path, 0) >= 20:
            continue

        # Prepare patch for model input
        patch_input = np.expand_dims(patch, axis=0).astype(np.float32) / 255.0

        # Run the ONNX model
        ort_inputs = {ort_session.get_inputs()[0].name: patch_input}
        ort_outs = ort_session.run(None, ort_inputs)
        
        # Get the probability of class 1
        prob_class_1 = ort_outs[0][0][1]

        # Save the patch if the probability is greater than the threshold 'p'
        if prob_class_1 > p:
            file_class = df.loc[df['File_name'] == image_path, 'Class'].values[0]
            save_dir = os.path.join(output_dir, str(file_class))
            save_path = os.path.join(save_dir, f"{os.path.basename(image_path)}_patch_{x}_{y}.png")
            Image.fromarray(patch).save(save_path)

            # Update count of saved patches
            image_patch_count[image_path] = image_patch_count.get(image_path, 0) + 1

    # Print the number of patches saved for each image
    for image, count in image_patch_count.items():
        print(f"Saved {count} patch(es) for {image}.")

def main():
    parser = argparse.ArgumentParser(description="Patch extractor and classifier from WSIs")
    parser.add_argument("--json_path", type=str, required=True, help="Path to the JSON file containing patch information")
    parser.add_argument("--patch_size", type=int, default=900, help="Size of the patches to be extracted")
    parser.add_argument("--level", type=int, default=2, help="Image level to extract the patches from")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the ONNX model")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the classified patches")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV file containing class information")
    parser.add_argument("--p", type=float, default=0.5, help="Probability threshold to save patches classified as class 1")

    args = parser.parse_args()

    generator = load_data_generator(args.json_path, args.patch_size, args.level)
    classify_and_save_patches(generator, args.model_path, args.output_dir, args.csv_path, args.p)

if __name__ == "__main__":
    main()
