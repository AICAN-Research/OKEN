import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import argparse

def parse_arguments():
    """
    Parse command-line arguments for PCA-based dimensionality reduction.
    """
    parser = argparse.ArgumentParser(description="Perform PCA-based dimensionality reduction.")
    parser.add_argument('--data_folder', type=str, required=True,
                        help="Path to the folder containing N subdirectories (each representing a cluster).")
    parser.add_argument('--output_folder', type=str, required=True,
                        help="Directory to save the PCA transformation matrix and mean.")
    parser.add_argument('--output_dimension', type=int, required=True,
                        help="Dimension to which the final vector is reduced (e.g., 2).")
    return parser.parse_args()

def load_images(directory, image_size):
    """
    Loads images from the given directory. Each subdirectory is treated
    as one class/label (the folder name should be convertible to an integer).
    """
    images = []
    labels = []
    for label_dir in os.listdir(directory):
        label_path = os.path.join(directory, label_dir)
        if not os.path.isdir(label_path):
            continue  # Skip any non-directory items
        for file in os.listdir(label_path):
            file_path = os.path.join(label_path, file)
            if not os.path.isfile(file_path):
                continue
            img = load_img(file_path, target_size=image_size[:-1])
            img = img_to_array(img) / 255.0
            images.append(img)
            labels.append(int(label_dir))
    return np.array(images), np.array(labels)

def extract_features(images, model):
    """
    Given a pretrained model (e.g., DenseNet121), extract features
    from the images.
    """
    return model.predict(images)

def plot_clusters(transformed, labels, title):
    """
    Scatter plot of the transformed features for visualization.
    """
    plt.figure()
    unique_labels = np.unique(labels)
    for lab in unique_labels:
        cluster = transformed[labels == lab]
        plt.scatter(cluster[:, 0], cluster[:, 1], label=f'Group {lab}')
    plt.title(title)
    plt.legend()
    plt.show()

def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Configuration
    image_size = (224, 224, 3)  # DenseNet121 input size
    output_dim = args.output_dimension
    output_folder = args.output_folder

    # Make sure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # --- Load base model (DenseNet121) for feature extraction ---
    base_model = DenseNet121(weights='imagenet', include_top=False, pooling='avg', input_shape=image_size)
    base_model.trainable = False

    # --- Load images from the specified folder ---
    images, labels = load_images(args.data_folder, image_size)
    print(f"Total images loaded: {len(images)}")

    # --- Extract features ---
    features = extract_features(images, base_model)
    print(f"Feature shape: {features.shape}")

    # --- PCA dimensionality reduction ---
    print(f"Performing PCA to reduce dimension to {output_dim}...")
    pca = PCA(n_components=output_dim)
    pca.fit(features)
    transformed = pca.transform(features)  # shape: (num_samples, output_dim)

    # --- Evaluate using silhouette score ---
    score = silhouette_score(transformed, labels)
    print(f"PCA Silhouette Score: {score}")

    # --- (Optional) Plot the 2D clusters if output_dimension == 2 ---
    if output_dim == 2:
        plot_clusters(transformed, labels, "PCA Clusters")

    # --- Save the PCA transformation matrix and mean ---
    # pca.components_ has shape (output_dim, original_feature_dim)
    # You can transpose it if you prefer the shape (original_feature_dim, output_dim).
    np.save(os.path.join(output_folder, 'pca_components.npy'), pca.components_)
    np.save(os.path.join(output_folder, 'pca_mean.npy'), pca.mean_)

    print("PCA transformation matrix and mean have been saved.")

if __name__ == "__main__":
    main()