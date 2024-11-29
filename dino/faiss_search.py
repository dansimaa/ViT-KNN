import argparse
import os
import json
from collections import Counter
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import faiss


def read_json(input_path):
    """Reads a JSON file from the input path."""
    with open(input_path, "r") as f:
        return json.load(f)


def save_json(data, output_path):
    """Saves a dictionary to a JSON file."""
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)


def build_training_index(dataset_config, imgpath_to_idx):
    """
    Builds a FAISS index used for "training" the KNN pseudo-labeling
    algorithm using the training and validation manually labeled splits.
    Args:
        dataset_config (dict): Labeled datasets' splits dictionary.
        imgpath_to_idx (dict): Mapping of image paths to indices.
    Returns:
        faiss.IndexFlatL2: The FAISS index.
        dict: Mapping of new indices to image paths.
    """
    faiss_train_index = faiss.IndexFlatL2(1536)
    new_idx = 0
    new_idx_to_imgpath = {}

    for split in ["train", "val"]:
        for img_path in dataset_config[split]:
            idx = imgpath_to_idx[img_path]
            vector_path = f"dino/dino_features/{int(idx):05d}.npy"
            if not os.path.exists(vector_path):
                continue

            vector = np.load(vector_path)
            faiss.normalize_L2(vector)
            faiss_train_index.add(vector)

            new_idx_to_imgpath[new_idx] = img_path
            new_idx += 1

    return faiss_train_index, new_idx_to_imgpath


def evaluate_performance(dataset_config, 
                         imgpath_to_idx, 
                         new_idx_to_imgpath, 
                         faiss_train_index, 
                         k):
    """
    Evaluates performance on test data.
    """
    true_labels = []
    predicted_labels = []

    for img_path in dataset_config["test"]:
        idx = imgpath_to_idx.get(img_path)
        if idx is None:
            continue

        vector_path = f"dino_features/{int(idx):05d}.npy"
        if not os.path.exists(vector_path):
            continue
        vector = np.load(vector_path)
        faiss.normalize_L2(vector)

        # Retrieve top-K neighbors
        dists, closest_neighbor_idxs = faiss_train_index.search(vector, k)
        closest_neighbor_idxs = closest_neighbor_idxs[0]

        true_label = dataset_config["test"][img_path][1]
        k_predicted_labels = []
        for closest_idx in closest_neighbor_idxs:
            img_path_train = new_idx_to_imgpath.get(closest_idx)
            if img_path_train in dataset_config["train"]:
                k_predicted_labels.append(dataset_config["train"][img_path_train][1])
            elif img_path_train in dataset_config["val"]:
                k_predicted_labels.append(dataset_config["val"][img_path_train][1])

        predicted_label = Counter(k_predicted_labels).most_common(1)[0][0]

        true_labels.append(true_label)
        predicted_labels.append(predicted_label)

    # Map and compute metrics
    label_mapping = {"l": "aligned", "o": "not_aligned"}
    true_labels = [label_mapping.get(label, label) for label in true_labels]
    predicted_labels = [label_mapping.get(label, label) for label in predicted_labels]

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, pos_label="aligned", average="binary")
    recall = recall_score(true_labels, predicted_labels, pos_label="aligned", average="binary")
    f1 = f1_score(true_labels, predicted_labels, pos_label="aligned", average="binary")
    conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=["aligned", "not_aligned"])

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (aligned): {precision:.4f}")
    print(f"Recall (aligned): {recall:.4f}")
    print(f"F1 Score (aligned): {f1:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")


def argument_parser():
    parser = argparse.ArgumentParser(description="Pseudo labeling with knn")
    parser.add_argument(
        "--dataset_config", 
        type=str, 
        required=True, 
        help="Path to dataset split JSON."
    )
    parser.add_argument(
        "--idx_to_imgpath", 
        type=str, 
        default="dino/mappings/idx_to_imgpath.json", 
        help="Path to idx_to_imgpath mapping."
    )
    parser.add_argument(
        "--imgpath_to_idx", 
        type=str, 
        default="dino/mappings/imgpath_to_idx.json", 
        help="Path to imgpath_to_idx mapping."
    )
    parser.add_argument(
        "--k", 
        type=int, 
        default=3, 
        help="Number of nearest neighbors for KNN FAISS search."
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default="dataset/pseudo_labeled_data.json", 
        help="Path to save the pseudo labels JSON."
    )    
    return parser.parse_args()


def main(args):
    dataset_config = read_json(args.dataset_config)
    idx_to_imgpath = read_json(args.idx_to_imgpath)
    imgpath_to_idx = read_json(args.imgpath_to_idx)

    faiss_train_index, new_idx_to_imgpath = build_training_index(dataset_config, imgpath_to_idx)

    # Pseudo labeling
    pseudo_labels = {}
    for idx, img_path in idx_to_imgpath.items():
        vector_path = f"dino/dino_features/{int(idx):05d}.npy"
        if not os.path.exists(vector_path):
            continue

        vector = np.load(vector_path)
        faiss.normalize_L2(vector)

        dists, closest_neighbor_idxs = faiss_train_index.search(vector, args.k)
        closest_neighbor_idxs = closest_neighbor_idxs[0]

        # K nearest neighbors
        k_predicted_labels = []
        for closest_idx in closest_neighbor_idxs:
            img_path_train = new_idx_to_imgpath.get(closest_idx)
            if img_path_train in dataset_config["train"]:
                k_predicted_labels.append(dataset_config["train"][img_path_train][1])
            elif img_path_train in dataset_config["val"]:
                k_predicted_labels.append(dataset_config["val"][img_path_train][1])

        predicted_label = Counter(k_predicted_labels).most_common(1)[0][0]
        pseudo_labels[img_path] = predicted_label

    label_mapping = {"l": "aligned", "o": "not_aligned"}
    for key, label in pseudo_labels.items():
        pseudo_labels[key] = label_mapping[label]
    
    save_json(pseudo_labels, args.output_path)

    # Evaluate performance on test data
    evaluate_performance(
        dataset_config, 
        imgpath_to_idx, 
        new_idx_to_imgpath, 
        faiss_train_index, 
        args.k
    )


if __name__ == "__main__":    
    args = argument_parser()
    main(args)
