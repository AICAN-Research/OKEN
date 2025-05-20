# OKEN
Optimizable Kernel Encoded Network


## Your Directory Structure Should Be Like This:

- `project_root/`
  - `Data/`
    - `Test/`
      - `WSI(tif)`
      - `WSI(tif)`
      - `WSI(tif)`
      - `...`
    - `Training/`
      - `WSI(tif)`
      - `WSI(tif)`
      - `WSI(tif)`
      - `...`
  - `Test.csv`
  - `Training.csv`
  - `model.onnx`
  - `patch_positions.py`
  - `create_patches.py`
  - `optimize.py`
  - `create_graphs.py`
  - `train_and_test.py`
 
## Creating Patch Positions

To create patch positions that will be used later in the workflow, you can use the `patch_position.py` script via the command line interface (CLI).

#### Example Command:

```bash
python patch_positions.py --data_dir ./Data/ --thumbnail_patch_size 9 --scale_factor 0.01 --tissue_threshold 0.5 --save_dir ./Data/
```

#### Explanation:

- **`--data_dir ./Data/`**: Specifies the directory where your data is stored.
- **`--thumbnail_patch_size 9`**: Sets the size of the patch at the thumbnail level. In this example, the size is 9 units.
- **`--scale_factor 0.01`**: Determines the scale factor applied to the thumbnail patch size. A scale factor of 0.01 indicates that each unit at the thumbnail level corresponds to 100 pixels on the whole slide image (WSI).
- **`--tissue_threshold 0.5`**: This threshold defines the minimum percentage of tissue in a patch for it to be considered valid.
- **`--save_dir ./Data/`**: Specifies the directory where the generated patch positions will be saved.

#### Calculating the Original Patch Size on the WSI:

The original patch size on the WSI (at level 0) is calculated as:

```
Original Patch Size = thumbnail_patch_size / scale_factor
```

For example, with `--thumbnail_patch_size 9` and `--scale_factor 0.01`, the original patch size on the WSI would be:

```
9 / 0.01 = 900 which corresponds to 900 × 900 pixels at level 0.
```

#### True Patch Size When Reading Images:

When reading the images, you can select the desired level to capture the patch. The final patch size is adjusted based on the selected level. For instance:

- If you choose to read the positions at **level 2**, the patch size would be:

```
900 / 2^2 = 900 / 4 = 225 which corresponds to 225 × 225 pixels
```

Thus, selecting different levels allows you to obtain patches of varying sizes depending on the resolution you need.

#### Output

This code creates two JSON files with the full path of the images as items that contain positions for the patches. It also creates thumbnail images of the WSIs with illustrated patches on them.




## Creating Patches

The script extracts patches from WSIs based on coordinates provided in a JSON file. Each patch is classified as a tumor/non-tumor using an ONNX model, and if the classification probability exceeds a given threshold, the patch is saved in a directory corresponding to its class.

#### JSON Structure

The JSON file created from the previous code should contain a dictionary where each key is the path to a WSI, and the value is another dictionary with the following structure:

```json
{
    "./Data/Training/wsi1.tif": {
        "class": 1,
        "original_patch_positions": [
            [x1, y1],
            [x2, y2],
            ...
        ]
    },
    "./Data/Training/wsi2.tif": {
        "class": 2,
        "original_patch_positions": [
            [x1, y1],
            [x2, y2],
            ...
        ]
    }
}
```

#### CSV Structure

The CSV file should map each WSI file name to its corresponding class. It should have the following structure:

```csv
File_name,Class
"./Data/Training/wsi1.tif",1
"./Data/Training/wsi2.tif",2
...,...
```

#### Example Command

The script can be executed from the command line with the following arguments:

```bash
python create_patches.py --json_path train_positions.json --patch_size 900 --level 2 --model_path model.onnx --output_dir ./patches/ --csv_path Training.csv --p 0.85 --save_probability 0.05
```

#### Arguments:

- `--json_path`: Path to the JSON file containing patch information.
- `--patch_size`: Size of the patches to be extracted (default: 900).
- `--level`: Image level to extract the patches from (default: 2).
- `--model_path`: Path to the ONNX model for patch classification.
- `--output_dir`: Directory where classified patches will be saved.
- `--csv_path`: Path to the CSV file containing class information.
- `--p`: Probability threshold for classifying a patch as class 1 (default: 0.5).
- `--save_probability`: Probability of saving a patch after it meets the classification threshold (default: 0.5).


This command will process the WSIs described in `patches.json`, classify the extracted patches using the ONNX model, and save patches with a class 1 probability greater than 0.85 based on a 5% chance in the output directory organized by class.

#### Output Structure

The output directory will be structured as follows:

```
patches/
├── 1/
│   ├── wsi1_patch_100_200.png
│   └── wsi1_patch_150_250.png
|   └── ...
|
└── 2/
    ├── wsi2_patch_300_400.png
    └── wsi2_patch_350_450.png
    └── ...
.
.
.

```

Each patch is saved in a directory named after its class, with filenames indicating the original WSI and the patch's location.


## Evolutionary Algorithm for Optimizing Transformation Matrix

This code ([optimize.py](./optimize.py)) implements an evolutionary algorithm to optimize a transformation matrix that reduces high-dimensional feature vectors to a lower-dimensional space. The goal is to find a matrix that maximizes the silhouette score, a measure of how well clusters are separated.

There is an augmentation section integrated inside the optimization code. You can comment out the augmentation and take the feature extraction out of the loop if you do not require augmentation at this stage. Otherwise, you are highly recommended to keep the augmentation as it improves the optimization and generalizability of the method. 

#### Features

- **Evolutionary Algorithm**: Optimizes a transformation matrix and additional parameters.
- **Feature Extraction**: Uses a pre-trained MobileNetV2 model to extract features from images.
- **Cluster Visualization**: Visualizes the clusters formed in the reduced-dimensional space (commented out as default).
- **Configurable**: Allows customization of population size, number of generations, mutation rate, and more via command-line arguments.


The script accepts several command-line arguments to configure the evolutionary algorithm and file paths. Below is a sample usage:

#### Example Command
```bash
python optimize.py --data_folder ./patches/ --output_folder ./ --output_dimension 8 --pop_size 200 --gen_max 100 --mutation_rate 0.8 --crossover_rate 0.4 --model_type d
```

To try different versions of optimization such as with augmentaion, for DINO, and for PCA, try the proper Python files.

#### Command-Line Arguments

- `--data_folder`: (Required) Path to the folder containing N subdirectories, each representing a cluster.
- `--output_folder`: (Required) Directory to save the best final transformation matrix and parameters.
- `--output_dimension`: (Required) Dimension to which the final vector is reduced (e.g., 2).
- `--pop_size`: (Optional) Population size for the evolutionary algorithm. Default is 200.
- `--gen_max`: (Optional) Number of generations for the evolutionary algorithm. Default is 100.
- `--mutation_rate`: (Optional) Mutation rate for the evolutionary algorithm. Default is 0.8.
- `--crossover_rate`: (Optional) Crossover rate for the evolutionary algorithm. Default is 0.4.
- `--model_type`: (Required) choices=['m', 'e', 'r', 'i', 'd'] indicating the type of model to use for feature extraction (m=MobileNetV2, e=EfficientNetB0, r=ResNet50, i=InceptionV3, d=DenseNet121).

## Creating Graphs from the WSIs

This script (create_graphs.py) follows the steps listed below:

1. **Image Loading**: The script reads regions of interest from the slides using the saved positions by `CuImage` library.

2. **Feature Extraction**: It utilizes the MobileNetV2 model to extract high-dimensional features from the image patches.

3. **Feature Transformation**: The extracted features are then transformed using the pre-saved transformation matrix.

4. **Graph Construction**: The transformed features are used to construct a graph, where each node represents a patch of the image, and edges represent the spatial proximity of these patches.

5. **Saving Graphs**: The script then saves the constructed graphs into a specified output directory. The graphs are organized by class labels provided in the CSV file, making it easy to manage and access the data for later training.

Remember that the model type used here should be the same that was used during the optimization to achieve the best performance.

#### Example Command:

Training images:
```bash
python create_graphs.py --matrix final_transformation_matrix.npy --json train_positions.json --csv Training.csv --output ./training_graphs/ --patch_size 900 --level 2 --model_type d
```

Test images:
```bash
python create_graphs.py --matrix final_transformation_matrix.npy --json test_positions.json --csv Test.csv --output ./test_graphs/ --patch_size 900 --level 2 --model_type d
```

## Train and Test
This script (train_and_test.py) trains or evaluates a graph-based model for classification. The model uses node features from graph data to perform classification tasks.

The script performs the following tasks:

1. **Load Graph Data**: Reads graph data from the specified directory, focusing on node features.
2. **Model Definition**: Defines a neural network model using TensorFlow/Keras for graph-based classification.
3. **Train and Evaluate**: Trains the model on provided training data and evaluates its performance on a test set.
4. **Prediction**: Makes predictions on new graph data using a trained model.
5. **Accuracy Calculation**: Computes the accuracy of the model's predictions.

#### Script Arguments

The script accepts the following command-line arguments:

- `--data_dir`: **(Required for Training)** Directory containing the graph data for training. The graph data should be organized in subdirectories where each subdirectory corresponds to a class (e.g., `class_1`, `class_2`, etc.).
  
- `--test_dir`: **(Required for Testing)** Directory containing the graph data for testing. Similar to the training directory, the test directory should be organized by class.
  
- `--model_path`: **(Required)** Path to save the trained model or load a pre-trained model. When training, the model will be saved at this path. When testing, a model will be loaded from this path.

- `--train`: **(Optional)** Specify this flag to train the model. If not specified, the script will assume the model should be evaluated on the test data.

- `--epochs`: **(Optional)** Number of training epochs. Defaults to `20`.

- `--batch_size`: **(Optional)** Batch size for training. Defaults to `8`.

- `--max_nodes`: **(Optional)** Max number of nodes in the graphs. This is needed for evaluation, especially if the number of nodes in the test data varies.


#### Training and Testing the Classifier 

```bash
python train_and_test.py --data_dir ./training_graphs/ --model_path ./model/ --train --epochs 20 --batch_size 1
```

#### Testing the Classifier 

```bash
python train_and_test.py --test_dir ./test_graphs/ --model_path ./model/
```

