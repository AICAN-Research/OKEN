import argparse
import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling1D, Softmax
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
import numpy as np
import pickle
import csv
import tensorflow_addons as tfa
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Input, Reshape, Conv2D, MaxPooling2D, GlobalAveragePooling2D

import networkx as nx  # For reading adjacency from the pickled graph

# For classical ML
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
try:
    import xgboost as xgb
    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


# ---------------------------------------------------------
# 1) Simple function to normalize adjacency
# ---------------------------------------------------------
def normalize_adjacency(A, add_self_loops=True, eps=1e-9):
    """
    Compute normalized adjacency: A_hat = D^{-1/2} (A + I) D^{-1/2}.
    """
    if add_self_loops:
        A = A + np.eye(A.shape[0])
    D = np.sum(A, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D + eps))
    return D_inv_sqrt @ A @ D_inv_sqrt


# ---------------------------------------------------------
# 2) GCN layer that uses adjacency to propagate features
# ---------------------------------------------------------
class GraphConv(tf.keras.layers.Layer):
    """
    A simple Graph Convolution layer:
      H = ReLU( A * X * W + b )
    Where A is the normalized adjacency matrix (with self-loops),
    X is the node feature matrix, and W is trainable weights.
    """
    def __init__(self, units, activation=tf.nn.relu, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        # input_shape should be [(None, N, F), (None, N, N)]
        # where N = max_nodes, F = num_features
        feature_dim = input_shape[0][-1]
        self.w = self.add_weight(
            shape=(feature_dim, self.units),
            initializer='glorot_uniform',
            trainable=True,
            name='w'
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            name='b'
        )
        super().build(input_shape)

    def call(self, inputs):
        """
        inputs = (X, A)
          X: (batch_size, N, F)
          A: (batch_size, N, N)
        """
        X, A = inputs
        support = tf.matmul(X, self.w)  # XW shape => (batch_size, N, units)
        out = tf.matmul(A, support)     # A (XW) => (batch_size, N, units)
        out = tf.nn.bias_add(out, self.b)
        if self.activation:
            out = self.activation(out)
        return out


# ---------------------------------------------------------
# 3) Loading graph data (including adjacency)
# ---------------------------------------------------------
def load_graph_data(graph_path, label, use_adjacency=False, normalize_adj=True):
    """
    Reads a pickled NetworkX graph.
    Returns:
       If use_adjacency=True: (X, A, Y)
       Else: (X, Y)
    """
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)

    # Collect node features
    x = []
    for node, data in graph.nodes(data=True):
        if 'features' not in data:
            print(f"Skipping graph {graph_path}: 'features' missing in node {node}.")
            return None  # Skip
        x.append(data['features'])

    x = np.array(x, dtype=np.float32)

    y = np.array([label], dtype=np.int32)

    if use_adjacency:
        # Convert to adjacency matrix
        A = nx.to_numpy_array(graph, nodelist=graph.nodes())
        if normalize_adj:
            A = normalize_adjacency(A, add_self_loops=True)
        return x, A, y
    else:
        return x, y


# ---------------------------------------------------------
# 4) File path loading
# ---------------------------------------------------------
def load_file_paths_from_directory(directory, val_slide_names=None):
    train_file_paths = []
    train_labels = []
    val_file_paths = []
    val_labels = []
    for class_name in os.listdir(directory):
        class_dir = os.path.join(directory, class_name)
        if os.path.isdir(class_dir):
            try:
                # Shift label so class_1 -> 0, class_2 -> 1, etc.
                label = int(class_name.split('_')[-1]) - 1
            except ValueError:
                print(f"Skipping class directory: {class_name}, unable to extract label.")
                continue

            for graph_filename in os.listdir(class_dir):
                graph_path = os.path.join(class_dir, graph_filename)
                slide_name = os.path.splitext(graph_filename)[0]
                # Check if this slide should go to validation
                if (val_slide_names is not None) and (slide_name in val_slide_names):
                    val_file_paths.append(graph_path)
                    val_labels.append(label)
                else:
                    train_file_paths.append(graph_path)
                    train_labels.append(label)
    return train_file_paths, train_labels, val_file_paths, val_labels


def load_test_file_paths_from_directory(directory):
    test_file_paths = []
    test_labels = []
    for class_name in os.listdir(directory):
        class_dir = os.path.join(directory, class_name)
        if os.path.isdir(class_dir):
            try:
                label = int(class_name.split('_')[-1]) - 1
            except ValueError:
                print(f"Skipping class directory: {class_name}, unable to extract label.")
                continue

            for graph_filename in os.listdir(class_dir):
                graph_path = os.path.join(class_dir, graph_filename)
                test_file_paths.append(graph_path)
                test_labels.append(label)
    return test_file_paths, test_labels


# ---------------------------------------------------------
# 5) Model Definitions (Deep Learning)
# ---------------------------------------------------------
def create_1d_cnn_model(input_shape, num_classes):
    """
    Original 1D CNN model: Input shape -> (max_nodes, num_features)
    """
    inputs = Input(shape=input_shape)  # (N, F)
    x = Conv1D(32, kernel_size=3, activation='relu')(inputs)
    x = Conv1D(128, kernel_size=3, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(128, kernel_size=3, activation='relu')(x)
    x = Conv1D(64, kernel_size=3, activation='relu')(x)
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=inputs, outputs=outputs)


def create_2d_cnn_model(input_shape, num_classes):
    """
    2D CNN: we treat (max_nodes, num_features) as a 2D "image" with 1 channel.
    To avoid negative dimensions when num_features is small, we use
    `padding='same'` in convolution and pooling layers.
    """
    inputs = Input(shape=input_shape)  # (N, F)
    # Reshape to (N, F, 1) so we can apply 2D convolutions
    x = Reshape((input_shape[0], input_shape[1], 1))(inputs)  # => (batch, N, F, 1)

    # Use 'same' padding to avoid negative dimensions:
    x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    x = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = GlobalAveragePooling2D()(x)

    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=inputs, outputs=outputs)


def create_graph_cnn_model(max_nodes, num_features, num_classes):
    """
    Simple adjacency-based GCN:
      1) GraphConv -> ReLU
      2) GraphConv -> ReLU
      3) Global average across nodes
      4) Dense softmax
    Inputs: [X, A]
      X: (batch_size, N, F)
      A: (batch_size, N, N)
    """
    X_in = Input(shape=(max_nodes, num_features), name='X_in')
    A_in = Input(shape=(max_nodes, max_nodes), name='A_in')

    x = GraphConv(8, activation=tf.nn.relu)([X_in, A_in])
    x = GraphConv(4, activation=tf.nn.relu)([x, A_in])
    
    # Aggregate across nodes
    x = tf.reduce_mean(x, axis=1)  # Global average pooling across N dimension
    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=[X_in, A_in], outputs=outputs)


# ---------------------------------------------------------
# 6) Generators (Deep Learning)
# ---------------------------------------------------------
class BalancedDataGenerator(tf.keras.utils.Sequence):
    def __init__(self,
                 file_paths,
                 labels,
                 batch_size,
                 max_nodes,
                 num_classes,
                 model_type='1dcnn'):
        """
        model_type: 
          - 'gcn' => return (X, A)
          - otherwise => return X only
        """
        self.file_paths = np.array(file_paths)
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.max_nodes = max_nodes
        self.num_classes = num_classes
        self.model_type = model_type

        self.indexes = np.arange(len(self.file_paths))
        # Per-class indices for balanced sampling
        self.class_indices = {}
        for c in np.unique(self.labels):
            self.class_indices[c] = np.where(self.labels == c)[0]

    def __len__(self):
        return int(np.floor(len(self.file_paths) / self.batch_size))

    def __getitem__(self, index):
        # Balanced sampling
        batch_indices = self.__get_balanced_indices()
        batch_file_paths = self.file_paths[batch_indices]
        batch_labels = self.labels[batch_indices]
        return self.__data_generation(batch_file_paths, batch_labels)

    def __get_balanced_indices(self):
        indices = []
        samples_per_class = self.batch_size // self.num_classes
        for c in self.class_indices:
            idx_c = np.random.choice(self.class_indices[c], samples_per_class, replace=True)
            indices.extend(idx_c)
        # If batch_size not perfectly divisible:
        remaining = self.batch_size - samples_per_class * self.num_classes
        if remaining > 0:
            extra = np.random.choice(self.indexes, remaining, replace=True)
            indices.extend(extra)
        np.random.shuffle(indices)
        return indices

    def __data_generation(self, batch_file_paths, batch_labels):
        X_list = []
        A_list = []
        y_list = []

        use_adjacency = (self.model_type == 'gcn')

        for file_path, label in zip(batch_file_paths, batch_labels):
            if use_adjacency:
                data = load_graph_data(file_path, label, use_adjacency=True)
                if data is None:
                    continue
                x, A, y = data
            else:
                data = load_graph_data(file_path, label, use_adjacency=False)
                if data is None:
                    continue
                x, y = data

            # Truncate / pad features to max_nodes
            if x.shape[0] > self.max_nodes:
                x = x[:self.max_nodes, :]
            else:
                pad_len = self.max_nodes - x.shape[0]
                x = np.pad(x, ((0, pad_len), (0, 0)), 'constant')

            # Truncate / pad adjacency if GCN
            if use_adjacency:
                if A.shape[0] > self.max_nodes:
                    A = A[:self.max_nodes, :self.max_nodes]
                else:
                    pad_len = self.max_nodes - A.shape[0]
                    A_padded = np.zeros((self.max_nodes, self.max_nodes), dtype=np.float32)
                    A_padded[:A.shape[0], :A.shape[1]] = A
                    A = A_padded
                A_list.append(A)

            X_list.append(x)
            y_list.append(y[0])  # shape(1,) -> scalar

        X_array = np.array(X_list)
        y_array = tf.keras.utils.to_categorical(y_list, num_classes=self.num_classes)

        if use_adjacency:
            A_array = np.array(A_list)
            return [X_array, A_array], y_array
        else:
            return X_array, y_array


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self,
                 file_paths,
                 labels,
                 batch_size,
                 max_nodes,
                 num_classes,
                 model_type='1dcnn'):
        self.file_paths = np.array(file_paths)
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.max_nodes = max_nodes
        self.num_classes = num_classes
        self.model_type = model_type

    def __len__(self):
        return int(np.ceil(len(self.file_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_file_paths = self.file_paths[index*self.batch_size:(index+1)*self.batch_size]
        batch_labels = self.labels[index*self.batch_size:(index+1)*self.batch_size]
        return self.__data_generation(batch_file_paths, batch_labels)

    def __data_generation(self, batch_file_paths, batch_labels):
        X_list = []
        A_list = []
        y_list = []

        use_adjacency = (self.model_type == 'gcn')

        for file_path, label in zip(batch_file_paths, batch_labels):
            if use_adjacency:
                data = load_graph_data(file_path, label, use_adjacency=True)
                if data is None:
                    continue
                x, A, y = data
            else:
                data = load_graph_data(file_path, label, use_adjacency=False)
                if data is None:
                    continue
                x, y = data

            # Truncate / pad X
            if x.shape[0] > self.max_nodes:
                x = x[:self.max_nodes, :]
            else:
                pad_len = self.max_nodes - x.shape[0]
                x = np.pad(x, ((0, pad_len), (0, 0)), 'constant')

            if use_adjacency:
                if A.shape[0] > self.max_nodes:
                    A = A[:self.max_nodes, :self.max_nodes]
                else:
                    pad_len = self.max_nodes - A.shape[0]
                    A_padded = np.zeros((self.max_nodes, self.max_nodes), dtype=np.float32)
                    A_padded[:A.shape[0], :A.shape[1]] = A
                    A = A_padded
                A_list.append(A)

            X_list.append(x)
            y_list.append(label)

        X_array = np.array(X_list)
        y_array = tf.keras.utils.to_categorical(y_list, num_classes=self.num_classes)

        if use_adjacency:
            A_array = np.array(A_list)
            return [X_array, A_array], y_array
        else:
            return X_array, y_array


# ---------------------------------------------------------
# 7a) Training/Evaluating (Deep Learning)
# ---------------------------------------------------------
def train_and_evaluate_dl(train_dir,
                          val_slide_names,
                          model_path,
                          epochs=20,
                          batch_size=8,
                          model_type='1dcnn'):
    # 1) Load file paths
    train_file_paths, train_labels, val_file_paths, val_labels = load_file_paths_from_directory(
        train_dir, val_slide_names
    )

    # 2) Determine max_nodes + num_features from a few samples
    sample_files = train_file_paths[:min(10, len(train_file_paths))]
    max_nodes = 0
    num_features = None
    for file_path in sample_files:
        data = load_graph_data(file_path, 0, use_adjacency=False)
        if data is not None:
            x, _ = data
            if x.shape[0] > max_nodes:
                max_nodes = x.shape[0]
            if num_features is None:
                num_features = x.shape[1]

    if max_nodes == 0:
        max_nodes = 100
    if num_features is None:
        num_features = 128

    num_classes = len(set(train_labels))

    # 3) Create data generators
    val_generator = DataGenerator(
        val_file_paths,
        val_labels,
        batch_size,
        max_nodes,
        num_classes,
        model_type=model_type
    )
    train_generator = BalancedDataGenerator(
        train_file_paths,
        train_labels,
        batch_size,
        max_nodes,
        num_classes,
        model_type=model_type
    )

    # 4) Build model
    if model_type == '1dcnn':
        model = create_1d_cnn_model((max_nodes, num_features), num_classes)
    elif model_type == '2dcnn':
        model = create_2d_cnn_model((max_nodes, num_features), num_classes)
    elif model_type == 'gcn':
        model = create_graph_cnn_model(max_nodes, num_features, num_classes)
    else:
        raise ValueError("Unknown DL model_type. Choose from ['1dcnn', '2dcnn', 'gcn'].")

    # 5) Compile model (with f1 metric from tfa)
    f1_metric = tfa.metrics.F1Score(num_classes=num_classes, average='macro', name='f1_score')
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy', 'AUC', f1_metric]
    )

    # 6) Setup callbacks
    if model_path:
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        checkpoint_path = os.path.join(model_path, 'best_model')
        csv_log_path = os.path.join(model_path, 'training_log.csv')
    else:
        checkpoint_path = 'best_model'
        csv_log_path = 'training_log.csv'

    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        save_format='tf',
        verbose=1
    )
    early_stopping = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )
    csv_logger = CSVLogger(csv_log_path, separator=",", append=False)
    callbacks = [checkpoint, early_stopping, csv_logger]

    # 7) Train
    model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )

    # 8) Save final model
    if model_path:
        model.save(os.path.join(model_path, 'final_model'), save_format='tf')

    # 9) Save max_nodes for inference
    if model_path:
        max_nodes_path = os.path.join(model_path, 'max_nodes.txt')
    else:
        max_nodes_path = 'max_nodes.txt'
    with open(max_nodes_path, 'w') as f:
        f.write(str(max_nodes))

    return model, max_nodes, num_classes


# ---------------------------------------------------------
# 7b) Training/Evaluating (Classical ML)
# ---------------------------------------------------------
def load_classical_dataset(file_paths, labels):
    """
    Load all (x,y) in memory and aggregate node features by mean across nodes.
    Adjacency is ignored for classical methods in this simple example.
    Returns X, y where X.shape => (n_samples, n_features).
    """
    X_data = []
    Y_data = []
    for (fp, lbl) in zip(file_paths, labels):
        data = load_graph_data(fp, lbl, use_adjacency=False)
        if data is None:
            continue
        x, _ = data  # x.shape => (#nodes, #features)
        # Simple aggregator: mean over nodes
        x_agg = np.mean(x, axis=0)
        X_data.append(x_agg)
        Y_data.append(lbl)
    return np.array(X_data), np.array(Y_data)


def train_classical_and_evaluate(train_dir,
                                 val_slide_names,
                                 model_path,
                                 model_type='dt'):
    """
    Train a classical ML model (DT, RF, XGB, SVM) on aggregated node features.
    """
    # 1) Load file paths
    train_file_paths, train_labels, val_file_paths, val_labels = load_file_paths_from_directory(
        train_dir, val_slide_names
    )

    # 2) Load in-memory aggregated data
    X_train, y_train = load_classical_dataset(train_file_paths, train_labels)
    X_val, y_val = load_classical_dataset(val_file_paths, val_labels)

    num_classes = len(set(train_labels))

    # 3) Create/Train model
    if model_type == 'dt':
        clf = DecisionTreeClassifier()
    elif model_type == 'rf':
        clf = RandomForestClassifier()
    elif model_type == 'xgb':
        if not HAVE_XGB:
            raise ImportError("xgboost is not installed. Please install xgboost or choose another model_type.")
        clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    elif model_type == 'svm':
        clf = SVC(probability=True)
    else:
        raise ValueError("Unknown classical model_type. Choose from ['dt','rf','xgb','svm']")

    clf.fit(X_train, y_train)

    # 4) Evaluate on validation
    if len(X_val) > 0:
        val_preds = clf.predict(X_val)
        val_probs = clf.predict_proba(X_val)

        acc = accuracy_score(y_val, val_preds)
        f1 = f1_score(y_val, val_preds, average='macro')
        # For multi-class AUC, we use 'ovr' approach
        try:
            auc = roc_auc_score(y_val, val_probs[:, 1], multi_class='ovr')
        except ValueError:
            # If there's only one class in val, handle gracefully
            auc = float('nan')

        print(f"Validation Accuracy: {acc:.4f}")
        print(f"Validation F1 (macro): {f1:.4f}")
        print(f"Validation AUC (macro-ovr): {auc:.4f}")
    else:
        print("No validation data provided.")

    # 5) Save model if path is given
    if model_path:
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        joblib.dump(clf, os.path.join(model_path, 'classical_model.pkl'))

    # For consistency, define max_nodes= None for classical (we won't need it).
    # We'll just store num_classes so we know how many we have at test time.
    return clf, None, num_classes


# ---------------------------------------------------------
# 7c) Common test evaluation function
# ---------------------------------------------------------
def test_model(args, model_type, model, max_nodes, num_classes):
    """
    Evaluate on test data and write CSV predictions.
    For DL => model is a tf.keras model
    For classical => model is scikit-like model
    """
    test_file_paths, test_labels = load_test_file_paths_from_directory(args.test_data_dir)

    # If no test data, just return
    if len(test_file_paths) == 0:
        print("No test data found. Exiting.")
        return

    # Evaluate
    if model_type in ['1dcnn', '2dcnn', 'gcn']:
        # Build the Keras generator
        test_generator = DataGenerator(
            test_file_paths,
            test_labels,
            args.batch_size,
            max_nodes,
            num_classes,
            model_type=model_type
        )
        results = model.evaluate(test_generator, verbose=1)
        print("Test performance:")
        for name, value in zip(model.metrics_names, results):
            print(f"{name}: {value}")

        # Predict probabilities
        predictions = model.predict(test_generator, verbose=1)

    else:
        # Classical: load in-memory data
        X_test, y_test = load_classical_dataset(test_file_paths, test_labels)
        if len(X_test) == 0:
            print("No valid test samples found.")
            return

        # Evaluate
        pred_labels = model.predict(X_test)
        pred_probs = model.predict_proba(X_test)

        acc = accuracy_score(y_test, pred_labels)
        f1 = f1_score(y_test, pred_labels, average='macro')

        # For multi-class problems, you'd pass the entire `pred_probs` to roc_auc_score 
        # with `multi_class='ovr'`. If it’s purely binary, you could do pred_probs[:, 1].
        try:
            auc_ = roc_auc_score(y_test, pred_probs[:, 1], multi_class='ovr')
        except ValueError:
            auc_ = float('nan')

        print("Test performance (classical):")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 (macro): {f1:.4f}")
        print(f"AUC (macro-ovr): {auc_:.4f}")

        predictions = pred_probs  # shape => (N, num_classes)

    # Write predictions to CSV
    if args.model_path:
        csv_file_path = os.path.join(args.model_path, 'test_predictions.csv')
    else:
        csv_file_path = 'test_predictions.csv'

    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ['ID', 'true_label'] + [f'prob_class_{i}' for i in range(num_classes)]
        writer.writerow(header)
        for i, file_path in enumerate(test_file_paths):
            sample_id = os.path.basename(file_path)
            true_label = test_labels[i]
            prob_values = predictions[i]
            writer.writerow([sample_id, true_label] + list(prob_values))

    print(f"Test predictions saved to {csv_file_path}")


# ---------------------------------------------------------
# 8) Main
# ---------------------------------------------------------
def main(args):
    # Read validation slide names if provided
    if args.val_slide_list:
        with open(args.val_slide_list, 'r') as f:
            val_slide_names = [line.strip() for line in f]
    else:
        val_slide_names = None

    # We branch by model type
    if args.train:
        # Training
        if args.model_type in ['1dcnn', '2dcnn', 'gcn']:
            # Deep-learning path
            model, max_nodes, num_classes = train_and_evaluate_dl(
                args.data_dir,
                val_slide_names,
                args.model_path,
                epochs=args.epochs,
                batch_size=args.batch_size,
                model_type=args.model_type
            )
        elif args.model_type in ['dt', 'rf', 'xgb', 'svm']:
            # Classical path
            model, max_nodes, num_classes = train_classical_and_evaluate(
                args.data_dir,
                val_slide_names,
                args.model_path,
                model_type=args.model_type
            )
        else:
            raise ValueError("Unknown model_type. Choose from DL ['1dcnn','2dcnn','gcn'] or classical ['dt','rf','xgb','svm'].")
    else:
        # Not training => load the model
        if not args.model_path:
            raise ValueError("Please specify --model_path if you are not training.")

        # Distinguish DL vs. classical
        if args.model_type in ['1dcnn', '2dcnn', 'gcn']:
            # Load a Keras model
            custom_objects = {'F1Score': tfa.metrics.F1Score, 'GraphConv': GraphConv}
            model = tf.keras.models.load_model(os.path.join(args.model_path, 'final_model'),
                                               custom_objects=custom_objects)

            # Load max_nodes
            max_nodes_path = os.path.join(args.model_path, 'max_nodes.txt')
            if os.path.exists(max_nodes_path):
                with open(max_nodes_path, 'r') as f:
                    max_nodes = int(f.read())
            else:
                raise ValueError("max_nodes.txt not found in model directory. Cannot pad/truncate properly.")
            # We have to guess num_classes from the model output layer
            num_classes = model.output_shape[-1]

        elif args.model_type in ['dt', 'rf', 'xgb', 'svm']:
            # Load a classical model
            model = joblib.load(os.path.join(args.model_path, 'classical_model.pkl'))
            # We set max_nodes = None for classical
            max_nodes = None
            # If you want to store num_classes, we might guess from model.n_classes_ or so:
            num_classes = int(model.n_classes_) if hasattr(model, 'n_classes_') else 2
        else:
            raise ValueError("Unknown model_type for loading.")

    # Test if provided
    if args.test_data_dir:
        test_model(args, args.model_type, model, max_nodes, num_classes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test a graph-based model (DL or classical).")
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing the graph data for training.')
    parser.add_argument('--val_slide_list', type=str,
                        help='Path to a text file containing slide names to use for validation.')
    parser.add_argument('--model_path', type=str,
                        help='Path to save or load the model.')
    parser.add_argument('--train', action='store_true',
                        help='Specify this flag to train the model.')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs (DL only).')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size (DL only).')
    parser.add_argument('--test_data_dir', type=str,
                        help='Directory containing the graph data for testing.')
    parser.add_argument('--model_type', type=str, default='1dcnn',
                        choices=['1dcnn', '2dcnn', 'gcn', 'dt', 'rf', 'xgb', 'svm'],
                        help="Choose model architecture: "
                             "'1dcnn', '2dcnn', 'gcn' (Deep Learning) "
                             "OR 'dt', 'rf', 'xgb', 'svm' (Classical ML).")

    args = parser.parse_args()
    main(args)
