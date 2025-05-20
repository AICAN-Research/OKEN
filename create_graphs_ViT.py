import argparse
import cucim  # Ensure cucim is installed
from PIL import Image
import numpy as np
import json
import networkx as nx
import os
import pickle
from tqdm import tqdm
import csv
import math
import gc

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Select GPU

# Import torch and related modules
import torch
import torchvision.transforms as transforms

# Ensure GPU memory management is handled appropriately
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# Import NearestNeighbors for efficient graph construction
from sklearn.neighbors import NearestNeighbors

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process images, extract features using ViT, and create graphs.")
    parser.add_argument('--matrix', required=True, help='Path to the transformation matrix file (.npy)')
    parser.add_argument('--params', required=True, help='Path to the transformation params file (.npy)')
    parser.add_argument('--json', required=True, help='Path to the JSON file with image details')
    parser.add_argument('--csv', required=True, help='Path to the CSV file with filenames and class labels')
    parser.add_argument('--output', required=True, help='Output directory to save the graphs and labels')
    parser.add_argument('--patch_size', type=int, default=900, help='Size of the patch to extract (default: 900)')
    parser.add_argument('--level', type=int, default=0, help='Level of the slide to extract the patches from (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing patches (default: 32)')
    return parser.parse_args()

def load_vit_model():
    model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
    model.eval()
    return model

def read_patch(slide, x, y, width, height, level):
    """Use CuImage to read a specific region from a slide."""
    region = slide.read_region(location=(x, y), size=(width, height), level=level)
    patch = Image.fromarray(np.array(region)).convert('RGB')
    return patch

def extract_features_batch(images, preprocess, model, device):
    """Extract features using ViT model in batch."""
    tensors = [preprocess(image).to(device) for image in images]
    input_batch = torch.stack(tensors)
    with torch.no_grad():
        outputs = model(input_batch)
    features = outputs.cpu().numpy()
    return features

def load_transformation_matrix(filepath):
    """Load the saved transformation matrix."""
    return np.load(filepath)

def load_transformation_data(matrix_filepath, params_filepath):
    """Load the saved transformation matrix and params."""
    matrix = np.load(matrix_filepath)
    params = np.load(params_filepath)
    return matrix, params

def linear_transform_features(features, matrix):
    """Apply the linear transformation to the features."""
    return features @ matrix


def transform_features(features, matrix, params):
    """Apply the transformation to the features."""
    return np.dot(features, matrix) * np.tanh(params[0]) + params[1]


def save_graph(graph, path):
    with open(path, 'wb') as f:
        pickle.dump(graph, f)

def load_csv_labels(csv_path):
    """Load the class labels from the CSV file."""
    labels = {}
    with open(csv_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            filename = os.path.basename(row[0])  # Get the base filename only
            class_label = int(row[1])
            labels[filename] = class_label
    return labels

def construct_graph(transformed_patches, positions):
    """Construct the graph from transformed patches and their positions."""
    G = nx.Graph()
    for i, (transformed, pos) in enumerate(zip(transformed_patches, positions)):
        G.add_node(i, features=transformed, position=pos)

    # Use NearestNeighbors for efficient nearest neighbors search
    positions_array = np.array(positions)
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(positions_array)
    distances, indices = nbrs.kneighbors(positions_array)

    for i, neighbors in enumerate(indices):
        for j in neighbors[1:]:  # Skip the first one (itself)
            G.add_edge(i, j)

    return G

def batch_generator(iterable, batch_size):
    """Yield batches from an iterable."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def process_image(image_path, details, transformation_matrix, transformation_params, patch_size, level, output_dir, csv_labels, preprocess, model, device, batch_size):
    """Process a single image to extract features, transform them, and construct a graph."""
    transformed_patches = []
    positions = []

    # Load the slide once
    slide = cucim.CuImage(image_path)

    # Compute the division factor based on the level
    division_factor = 2 ** level

    # Adjust patch size and positions for the specified level
    adjusted_patch_size = patch_size // division_factor
    adjusted_positions = [(x, y) for x, y in details['original_patch_positions']]

    # Create a generator for patches
    def patch_generator():
        for x, y in adjusted_positions:
            yield (read_patch(slide, x, y, adjusted_patch_size, adjusted_patch_size, level), (x, y))

    # Process patches in batches
    gen = patch_generator()
    for batch in batch_generator(gen, batch_size):
        patches, batch_positions = zip(*batch)

        # Extract features
        features = extract_features_batch(patches, preprocess, model, device)

        # Apply transformation and collect results
        for feature, pos in zip(features, batch_positions):
            transformed = linear_transform_features(feature, transformation_matrix) ## for linear case
            # transformed = transform_features(feature, transformation_matrix, transformation_params) ## for nonlinear case
            transformed_patches.append(transformed.flatten())
            positions.append(pos)

        # Clean up to free memory
        del patches, features
        gc.collect()

    G = construct_graph(transformed_patches, positions)

    image_filename = os.path.basename(image_path)
    base_filename = os.path.splitext(image_filename)[0]  # Get the base filename without extension
    class_label = csv_labels.get(image_filename)

    if class_label is not None:
        # Create class-specific subdirectory if it doesn't exist
        class_dir = os.path.join(output_dir, f"class_{class_label}")
        os.makedirs(class_dir, exist_ok=True)

        graph_filename = f"{base_filename}.pkl"  # Save with original filename but with .pkl extension
        save_graph(G, os.path.join(class_dir, graph_filename))
    else:
        print(f"Warning: No class label found for {image_filename}")

def apply_transformation_and_create_graphs(json_path, matrix_path, params_path, patch_size, level, output_dir, csv_path, preprocess, model, device, batch_size):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    transformation_matrix, transformation_params = load_transformation_data(matrix_path, params_path)
    csv_labels = load_csv_labels(csv_path)

    with open(json_path, 'r') as file:
        data = json.load(file)

    for image_index, (image_path, details) in tqdm(enumerate(data.items()), total=len(data)):
        process_image(
            image_path,
            details,
            transformation_matrix,
            transformation_params,
            patch_size,
            level,
            output_dir,
            csv_labels,
            preprocess,
            model,
            device,
            batch_size,
        )

if __name__ == "__main__":
    args = parse_arguments()

    # Load ViT model
    model = load_vit_model()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Ensure model is in evaluation mode
    model.eval()

    # Define preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.Resize(256),             # Resize the shorter side to 256 pixels
        transforms.CenterCrop(224),         # Center crop to 224x224 pixels
        transforms.ToTensor(),              # Convert the image to a PyTorch tensor
        transforms.Normalize(               # Normalize using ImageNet's mean and std
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    apply_transformation_and_create_graphs(
        json_path=args.json,
        matrix_path=args.matrix,
        params_path=args.params,
        patch_size=args.patch_size,
        level=args.level,
        output_dir=args.output,
        csv_path=args.csv,
        preprocess=preprocess,
        model=model,
        device=device,
        batch_size=args.batch_size,
    )
