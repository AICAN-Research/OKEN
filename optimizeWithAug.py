import numpy as np
import os
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import argparse
from tensorflow.keras.applications import MobileNetV2, InceptionV3, ResNet50, DenseNet121, EfficientNetB0
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import silhouette_score
from saug import multi_lens_distortion

# Argument parser for CLI input
def parse_arguments():
    parser = argparse.ArgumentParser(description="Run a genetic algorithm to optimize image clustering based on features extracted by pre-trained models.")
    parser.add_argument('--data_folder', type=str, required=True, help="Path to the folder containing images.")
    parser.add_argument('--output_folder', type=str, required=True, help="Directory to save the optimized parameters.")
    parser.add_argument('--output_dimension', type=int, required=True, help="Dimension to which the final vector is reduced.")
    parser.add_argument('--pop_size', type=int, default=200, help="Population size for the genetic algorithm.")
    parser.add_argument('--gen_max', type=int, default=100, help="Number of generations for the genetic algorithm.")
    parser.add_argument('--mutation_rate', type=float, default=0.8, help="Mutation rate for the genetic algorithm.")
    parser.add_argument('--crossover_rate', type=float, default=0.4, help="Crossover rate for the genetic algorithm.")
    parser.add_argument('--model_type', type=str, required=True, choices=['m', 'e', 'r', 'i', 'd'], help="Type of model to use (m=MobileNetV2, e=EfficientNetB0, r=ResNet50, i=InceptionV3, d=DenseNet121).")
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

def load_images(image_paths, image_size):
    images = []
    for file_path in image_paths:
        img = load_img(file_path, target_size=image_size[:-1])
        img = img_to_array(img) / 255.0
        images.append(img)
    return np.array(images)

def extract_features(images, model):
    return model.predict(images)

def initialize_population(pop_size, feature_dim, output_dim):
    return [{'matrix': np.random.randn(feature_dim, output_dim) * 0.01, 'params': np.random.rand(50)} for _ in range(pop_size)]

def evaluate(individual, features, labels):
    matrix = individual['matrix']
    params = individual['params']

    # Apply transformation with dynamic parameters
    transformed = np.dot(features, matrix) * np.tanh(params[0]) + params[1]
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

def plot_clusters(transformed, labels, title, n_groups):
    plt.figure()
    for i in range(n_groups):
        cluster = transformed[labels == i]
        plt.scatter(cluster[:, 0], cluster[:, 1], label=f'Group {i}')
    plt.title(title)
    plt.legend()
    plt.show()

def augment_images(images, image_size):
    augmented_images = []
    for img in images:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.rot90(img, k=np.random.randint(0, 4))
        zoom_scale = np.random.uniform(0.8, 1.0)
        img = tf.image.central_crop(img, central_fraction=zoom_scale)
        img = tf.image.resize(img, (image_size[0], image_size[1]))  # Resize back to original size

        img = tf.image.random_brightness(img, max_delta=0.2)
        img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
        img = tf.image.random_hue(img, max_delta=0.05)
        img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
        
        # Apply a multi-lens distortion with random parameters
        num_lenses = np.random.randint(4, 8)
        radius_range = (10, 50)
        strength_range = (-0.5, 0.5)
        img = multi_lens_distortion(img.numpy(), num_lenses, radius_range, strength_range)

        # Convert numpy array back to tensor
        img = tf.convert_to_tensor(img, dtype=tf.float32)
        
        augmented_images.append(img)
    
    return np.array(augmented_images)

def genetic_algorithm(args):
    # Configuration from arguments
    n_groups = len(os.listdir(args.data_folder))
    
    pop_size = args.pop_size
    gen_max = args.gen_max
    mutation_rate = args.mutation_rate
    crossover_rate = args.crossover_rate
    output_dim = args.output_dimension
    output_folder = args.output_folder

    model_mapping = {
        'm': (MobileNetV2(weights='imagenet', include_top=False, pooling='avg'), (224, 224, 3)),
        'e': (EfficientNetB0(weights='imagenet', include_top=False, pooling='avg'), (240, 240, 3)),
        'r': (ResNet50(weights='imagenet', include_top=False, pooling='avg'), (224, 224, 3)),
        'i': (InceptionV3(weights='imagenet', include_top=False, pooling='avg'), (299, 299, 3)),
        'd': (DenseNet121(weights='imagenet', include_top=False, pooling='avg'), (224, 224, 3))
    }
    
    base_model, image_size = model_mapping[args.model_type]
    base_model.trainable = False

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
        images = load_images(sampled_image_paths, image_size)
        labels = sampled_labels

        # Augment images
        augmented_images = augment_images(images.copy(), image_size)
        features = extract_features(augmented_images, base_model)

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
    np.save(os.path.join(output_folder, 'final_transformation_matrix.npy'), best_individual['matrix'])
    np.save(os.path.join(output_folder, 'final_transformation_params.npy'), best_individual['params'])

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()
    
    # Run the genetic algorithm with the provided arguments
    genetic_algorithm(args)
