import argparse
import os
import json
import random


def read_json(input_path):
    """Read and parse a JSON file."""
    with open(input_path, "r") as file:
        input_file = json.load(file)
    return input_file


def get_img_paths(img_dir):
    """Get full paths of all images in a directory."""
    return [os.path.join(img_dir, img) for img in os.listdir(img_dir)]


def get_labeled_subset(data_dir):
    """
    Get 'given' labeled subset image paths and their associated labels.

    Args:
        data_dir (str): Path to the dataset directory.

    Returns:
        dict: Dictionary mapping image paths to their labels.
    """
    labeled_data_dir = os.path.join(data_dir, "example_set")

    aligned_img_dir = os.path.join(labeled_data_dir, "aligned")
    not_aligned_img_dir = os.path.join(labeled_data_dir, "not_aligned")

    aligned_img_paths = get_img_paths(aligned_img_dir)
    not_aligned_img_paths = get_img_paths(not_aligned_img_dir)

    images_dict = {}
    for image_path in aligned_img_paths + not_aligned_img_paths:
        images_dict[image_path] = os.path.basename(os.path.dirname(image_path))

    return images_dict


def get_labeled_trainset(labels_dict, unlabeled_data_dir):
    """
    Map 'manually' labeled data image paths and their associated labels.

    Args:
        labels_dict (dict): Dictionary of labeled data.
        unlabeled_data_dir (str): Path to the unlabeled data directory.

    Returns:
        dict: Dictionary mapping full image paths to their labels.
    """
    images_dict = {}
    for img_name in labels_dict:
        images_dict[os.path.join(unlabeled_data_dir, img_name)] = labels_dict[img_name][1]

    return images_dict


def stratified_split(data, train_ratio, val_ratio, test_ratio):
    """
    Perform a stratified split of the data.

    Args:
        data (dict): Dictionary of data to split.
        train_ratio (float): Ratio of training data.
        val_ratio (float): Ratio of validation data.
        test_ratio (float): Ratio of test data.

    Returns:
        tuple: Train, validation, and test dictionaries.
    """
    def split_dict(data_dict, train_ratio, val_ratio):
        """Helper to split a dictionary by ratios."""
        keys = list(data_dict.keys())
        random.shuffle(keys)

        train_size = int(len(keys) * train_ratio)
        val_size = int(len(keys) * val_ratio)

        train_keys = keys[:train_size]
        val_keys = keys[train_size:train_size + val_size]
        test_keys = keys[train_size + val_size:]

        return (
            {k: data_dict[k] for k in train_keys},
            {k: data_dict[k] for k in val_keys},
            {k: data_dict[k] for k in test_keys}
        )

    aligned = {k: v for k, v in data.items() if v == "aligned"}
    not_aligned = {k: v for k, v in data.items() if v == "not_aligned"}

    train_aligned, val_aligned, test_aligned = split_dict(aligned, train_ratio, val_ratio)
    train_not_aligned, val_not_aligned, test_not_aligned = split_dict(not_aligned, train_ratio, val_ratio)

    train_set = {**train_aligned, **train_not_aligned}
    val_set = {**val_aligned, **val_not_aligned}
    test_set = {**test_aligned, **test_not_aligned}

    return train_set, val_set, test_set
  
    
def parse_arguments():
    parser = argparse.ArgumentParser(description="Split labeled data into train/val/test.")
    parser.add_argument(
        "--data_dir",
        type=str, 
        required=True, 
        help="Path to the dataset directory."
    )
    parser.add_argument(
        "--good_light_labels", 
        type=str,
        default="dataset/labeled/good_light_labels.json", 
        help="Path to the good light labels JSON file."
    )
    parser.add_argument(
        "--bad_light_labels",
        type=str, 
        default="dataset/labeled/bad_light_labels.json", 
        help="Path to the bad light labels JSON file."
    )
    parser.add_argument(
        "--output_path", 
        default="dataset/split.json", 
        help="Output path for the split JSON file."
    )
    parser.add_argument(
        "--train_ratio", 
        type=float, 
        default=0.7, 
        help="Train set ratio."
    )
    parser.add_argument(
        "--val_ratio", 
        type=float, 
        default=0.15, 
        help="Validation set ratio."
    )
    parser.add_argument(
        "--test_ratio", 
        type=float, 
        default=0.15, 
        help="Test set ratio."
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    labeled_imgs = get_labeled_subset(args.data_dir)
    
    unlabeled_data_dir = os.path.join(args.data_dir, "train_set") # Manually labeled dataset
    good_light_labels_dict = read_json(args.good_light_labels)
    bad_light_labels_dict = read_json(args.bad_light_labels)

    good_light_imgs = get_labeled_trainset(good_light_labels_dict, unlabeled_data_dir)
    bad_light_imgs = get_labeled_trainset(bad_light_labels_dict, unlabeled_data_dir)

    all_labeled_imgs = {**labeled_imgs, **good_light_imgs, **bad_light_imgs} 

    train_set, val_set, test_set = stratified_split(
        all_labeled_imgs,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio
    )

    split_output = {"train": train_set, "val": val_set, "test": test_set}
    with open(args.output_path, "w") as json_file:
        json.dump(split_output, json_file, indent=4)


if __name__ == "__main__":
    main()
