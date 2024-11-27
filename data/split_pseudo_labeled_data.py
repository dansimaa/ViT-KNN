import os
import random
import json
import argparse


def read_json(input_path):
    """Reads a JSON file and returns the loaded data."""
    with open(input_path, "r") as f:
        return json.load(f)


def save_json(output_path, data):
    """Saves data to a JSON file."""
    with open(output_path, "w") as json_file:
        json.dump(data, json_file, indent=4)


def add_and_split_new_data(
    labeled_split, 
    pseudo_labeled_data, 
    train_ratio=0.7, 
    val_ratio=0.15, 
    test_ratio=0.15
):
    """
    Updates the labeled dataset splits with the pseudo-labeled data, ensuring no overlap, 
    and splits the new data into train, validation, and test sets based on specified ratios.

    Args:
        labeled_split (dict): Existing labeled dataset splits with keys 'train', 'val', and 'test'.
        pseudo_labeled_data (dict): New pseudo-labeled data to be added.
        train_ratio (float): Ratio of training data.
        val_ratio (float): Ratio of validation data.
        test_ratio (float): Ratio of test data.

    Returns:
        dict: Updated dataset splits.
    """

    train, val, test = {}, {}, {}
 
    # Handle overlapping data 
    for image_path, label in pseudo_labeled_data.items():
        if image_path in labeled_split["train"]:
            continue
        elif image_path in labeled_split["val"]:
            continue
        elif image_path in labeled_split["test"]:
            continue
        else:
            if (image_path not in train 
                and image_path not in val 
                and image_path not in test):
                train[image_path] = label

    # Split non-overlapping data into train/val/test
    non_overlapping_data = list(train.items())  
    random.shuffle(non_overlapping_data) 

    n_new_data = len(non_overlapping_data)
    train_end = int(n_new_data * train_ratio)
    val_end = train_end + int(n_new_data * val_ratio)

    for idx, (image_path, label) in enumerate(non_overlapping_data):
        if idx < train_end:
            labeled_split["train"][image_path] = label
        elif idx < val_end:
            labeled_split["val"][image_path] = label
        else:
            labeled_split["test"][image_path] = label

    return labeled_split


def parse_arguments():
    parser = argparse.ArgumentParser(description="Split augmented dataset into train/val/test.")
    parser.add_argument(
        "--labeled_split_path",
        type=str, 
        required=True, 
        help="Path to the labeled dataset splits JSON file."
    )
    parser.add_argument(
        "--pseudo_labeled_data_path",
        type=str, 
        required=True, 
        help="Path to the pseudo data JSON file."
    )
    parser.add_argument(
        "--output_path",
        type=str, 
        default="dataset/augmented_split.json", 
        help="Path to save the augmented dataset splits JSON file."
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

    labeled_split_path = os.path.abspath(args.labeled_split_path)
    pseudo_labeled_data_path = os.path.abspath(args.pseudo_labeled_data_path)
    output_path = os.path.abspath(args.output_path)

    labeled_split = read_json(labeled_split_path)
    pseudo_labeled_data = read_json(pseudo_labeled_data_path)

    augmented_split = add_and_split_new_data(
        labeled_split, 
        pseudo_labeled_data, 
        args.train_ratio, 
        args.val_ratio, 
        args.test_ratio
    )

    save_json(output_path, augmented_split)
    print(f"Processed split saved to: {output_path}")


if __name__ == "__main__":
    main()
