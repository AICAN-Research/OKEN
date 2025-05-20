import numpy as np
import os
import tensorflow as tf
# from tensorflow.keras.applications import DenseNet121
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.applications import EfficientNetB0
# from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import InceptionV3


from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import silhouette_score
import random
import matplotlib.pyplot as plt
import argparse

# Argument parser for CLI input
def parse_arguments():
    parser = argparse.ArgumentParser(description="Optimize a transformation matrix using a genetic algorithm.")
    parser.add_argument('--data_folder', type=str, required=True, help="Path to the folder containing N subdirectories (each representing a cluster).")
    parser.add_argument('--output_folder', type=str, required=True, help="Directory to save the best final transformation matrix.")
    parser.add_argument('--output_dimension', type=int, required=True, help="Dimension to which the final vector is reduced (e.g., 2).")
    parser.add_argument('--pop_size', type=int, default=200, help="Population size for the genetic algorithm.")
    parser.add_argument('--gen_max', type=int, default=100, help="Number of generations for the genetic algorithm.")
    parser.add_argument('--mutation_rate', type=float, default=0.8, help="Mutation rate for the genetic algorithm.")
    parser.add_argument('--crossover_rate', type=float, default=0.4, help="Crossover rate for the genetic algorithm.")
    return parser.parse_args()

def load_images(directory, image_size):
    images = []
    labels = []
    for label_dir in os.listdir(directory):
        for file in os.listdir(os.path.join(directory, label_dir)):
            img = load_img(os.path.join(directory, label_dir, file), target_size=image_size[:-1])
            img = img_to_array(img) / 255.0
            images.append(img)
            labels.append(int(label_dir))
    return np.array(images), np.array(labels)

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

def genetic_algorithm(args):
    # Configuration from arguments
    n_groups = len(os.listdir(args.data_folder))
    image_size = (299, 299, 3)  # MobileNetV2 expects these dimensions
    pop_size = args.pop_size
    gen_max = args.gen_max
    mutation_rate = args.mutation_rate
    crossover_rate = args.crossover_rate
    output_dim = args.output_dimension
    output_folder = args.output_folder

    # Load MobileNetV2
    base_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg', input_shape=image_size)
    base_model.trainable = False

    # Load training and test datasets
    images, labels = load_images(args.data_folder, image_size)
    print("Total images being used for optimization:", len(images))
    features = extract_features(images, base_model)

    population = initialize_population(pop_size, features.shape[1], output_dim)
    for generation in range(gen_max):
        fitnesses = [evaluate(ind, features, labels)[0] for ind in population]
        best_index = np.argmax(fitnesses)
        best_individual = population[best_index]
        best_score, best_transform = evaluate(best_individual, features, labels)

        print(f"Generation {generation}: Best Silhouette Score = {best_score}")

        # Uncomment to visualize the clusters at each generation
        # plot_clusters(best_transform, labels, f'Generation {generation} Clusters', n_groups)

        # Uncomment to save the transformation matrix and parameters at each generation
        # np.save(os.path.join(output_folder, f'transformation_matrix_gen_{generation}.npy'), best_individual['matrix'])
        # np.save(os.path.join(output_folder, f'transformation_params_gen_{generation}.npy'), best_individual['params'])

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