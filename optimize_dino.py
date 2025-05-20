import numpy as np
import os
import torch
import random
import matplotlib.pyplot as plt
import argparse
from PIL import Image
from sklearn.metrics import silhouette_score
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Select GPU

# Argument parser for CLI input
def parse_arguments():
    parser = argparse.ArgumentParser(description="Run an evolutionary algorithm to optimize image clustering based on features extracted by DINO model.")
    parser.add_argument('--data_folder', type=str, required=True, help="Path to the folder containing images.")
    parser.add_argument('--output_folder', type=str, required=True, help="Directory to save the optimized parameters.")
    parser.add_argument('--output_dimension', type=int, required=True, help="Dimension to which the final vector is reduced.")
    parser.add_argument('--pop_size', type=int, default=200, help="Population size for the evolutionary algorithm.")
    parser.add_argument('--gen_max', type=int, default=100, help="Number of generations for the evolutionary algorithm.")
    parser.add_argument('--mutation_rate', type=float, default=0.8, help="Mutation rate for the evolutionary algorithm.")
    parser.add_argument('--crossover_rate', type=float, default=0.4, help="Crossover rate for the evolutionary algorithm.")
    parser.add_argument('--portion', type=float, default=1.0, help="Portion of the data to be used in each generation (0 < p <= 1).")
    return parser.parse_args()

# Function to get image paths and labels
def get_image_paths(directory):
    image_paths = []
    labels = []
    for label_dir in os.listdir(directory):
        label_path = os.path.join(directory, label_dir)
        if not os.path.isdir(label_path):
            continue
        for file in os.listdir(label_path):
            file_path = os.path.join(label_path, file)
            image_paths.append(file_path)
            labels.append(int(label_dir))
    return image_paths, labels

def load_images(image_paths):
    images = []
    for file_path in image_paths:
        img = Image.open(file_path).convert('RGB')  # Ensure RGB
        images.append(img)
    return images

def augment_images(images):
    augmented_images = []
    for img in images:
        # Random horizontal flip
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        # Random vertical flip
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        # Random rotation
        angle = random.choice([0, 90, 180, 270])
        img = img.rotate(angle)
        # Random crop and resize
        scale = random.uniform(0.8, 1.0)
        w, h = img.size
        new_w, new_h = int(w * scale), int(h * scale)
        left = random.randint(0, w - new_w) if w - new_w > 0 else 0
        top = random.randint(0, h - new_h) if h - new_h > 0 else 0
        img = img.crop((left, top, left + new_w, top + new_h))
        img = img.resize((w, h), Image.BILINEAR)
        # Random color jitter
        brightness = random.uniform(0.8, 1.2)
        contrast = random.uniform(0.8, 1.2)
        saturation = random.uniform(0.8, 1.2)
        hue = random.uniform(-0.05, 0.05)
        img = F.adjust_brightness(img, brightness)
        img = F.adjust_contrast(img, contrast)
        img = F.adjust_saturation(img, saturation)
        img = F.adjust_hue(img, hue)
        augmented_images.append(img)
    return augmented_images

def extract_features(images, preprocess, model, device):
    # Process images
    tensors = [preprocess(img).to(device) for img in images]
    input_batch = torch.stack(tensors)
    # Get the model outputs without computing gradients
    with torch.no_grad():
        outputs = model(input_batch)
    # Extract features
    features = outputs.cpu().numpy()  # Shape: (batch_size, feature_dim)
    return features

def initialize_population(pop_size, feature_dim, output_dim):
    return [{'matrix': np.random.randn(feature_dim, output_dim) * 0.01, 'params': np.random.rand(2)} for _ in range(pop_size)]

def evaluate(individual, features, labels):
    matrix = individual['matrix']
    params = individual['params']

    transformed = np.dot(features, matrix) ## linear
    # Apply transformation with nonlinear parameters
    # transformed = np.dot(features, matrix) * np.tanh(params[0]) + params[1]
    score = silhouette_score(transformed, labels)
    return score, transformed

def select(population, fitnesses):
    sorted_pop = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)
    return [ind for ind, fit in sorted_pop[:len(population)//2]]

def crossover(parent1, parent2):
    child_matrix = (parent1['matrix'] + parent2['matrix']) / 2
    child_params = (parent1['params'] + parent2['params']) / 2
    return {'matrix': child_matrix, 'params': child_params}

def mutate(individual, mutation_rate):
    if random.random() < mutation_rate:
        perturbation_matrix = np.random.normal(0, 0.01, size=individual['matrix'].shape)
        perturbation_params = np.random.normal(0, 0.1, size=individual['params'].shape)
        individual['matrix'] += perturbation_matrix
        individual['params'] += perturbation_params

def evolutionary_algorithm(args):
    # Configuration from arguments
    n_groups = len(os.listdir(args.data_folder))

    pop_size = args.pop_size
    gen_max = args.gen_max
    mutation_rate = args.mutation_rate
    crossover_rate = args.crossover_rate
    output_dim = args.output_dimension
    output_folder = args.output_folder

    # Load DINO model
    model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
    model.eval()  # Set model to evaluation mode

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

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

    # Get image paths and labels
    all_image_paths, all_labels = get_image_paths(args.data_folder)
    all_labels = np.array(all_labels)

    # Initialize population (will be adjusted after extracting features)
    population_initialized = False

    for generation in range(gen_max):
        # Sample a portion of the data
        num_samples = int(len(all_image_paths) * args.portion)
        indices = np.random.choice(len(all_image_paths), num_samples, replace=False)
        sampled_image_paths = [all_image_paths[i] for i in indices]
        sampled_labels = all_labels[indices]

        # Load images
        images = load_images(sampled_image_paths)
        labels = sampled_labels

        # Augment images
        augmented_images = augment_images(images.copy())

        # Extract features
        features = extract_features(augmented_images, preprocess, model, device)

        # Initialize population based on feature dimensions
        if not population_initialized:
            population = initialize_population(pop_size, features.shape[1], output_dim)
            population_initialized = True

        fitnesses = [evaluate(ind, features, labels)[0] for ind in population]
        best_index = np.argmax(fitnesses)
        best_individual = population[best_index]
        best_score, best_transform = evaluate(best_individual, features, labels)

        print(f"Generation {generation}: Best Silhouette Score = {best_score}")

        selected = select(population, fitnesses)
        next_gen = []

        while len(next_gen) < pop_size:
            p1, p2 = random.sample(selected, 2)
            if random.random() < crossover_rate:
                # Perform crossover
                child = crossover(p1, p2)
            else:
                # Clone one of the parents
                child = random.choice([p1, p2])
            mutate(child, mutation_rate)
            next_gen.append(child)

        population = next_gen

    # Save the final transformation matrix and parameters
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    np.save(os.path.join(output_folder, 'final_transformation_matrix.npy'), best_individual['matrix'])
    np.save(os.path.join(output_folder, 'final_transformation_params.npy'), best_individual['params'])

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()
    
    # Run the evolutionary_algorithm with the provided arguments
    evolutionary_algorithm(args)
