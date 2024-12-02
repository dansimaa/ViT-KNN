import argparse
import os
import json


def save_mapping(path, mapping):
    """Save the mapping dictionary to a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True) 
    with open(path, "w") as json_file:
        json.dump(mapping, json_file, indent=4)


def get_image_paths(directory):
    """Retrieve all image file paths from a given directory."""
    return [os.path.join(directory, img_path) for img_path in os.listdir(directory)]


def get_image_set(base_dir, sub_dirs):
    """Get all image paths from specified subdirectories."""
    image_paths = []
    for sub_dir in sub_dirs:
        directory = os.path.join(base_dir, sub_dir)
        image_paths.extend(get_image_paths(directory))
    return image_paths


def main(args):
    example_set_dirs = ["example_set/aligned", "example_set/not_aligned"] # 'Example' set paths
    example_set_img_paths = get_image_set(args.data_dir, example_set_dirs)

    train_set_dirs = ["train_set/good_light", "train_set/bad_light"] # 'Train' set paths
    train_set_img_paths = get_image_set(args.data_dir, train_set_dirs)

    img_paths = example_set_img_paths + train_set_img_paths

    # Create and save mappings
    idx_to_imgpath = {i: img_path for i, img_path in enumerate(img_paths)}
    imgpath_to_idx = {v: k for k, v in idx_to_imgpath.items()}

    save_mapping(
        os.path.join(args.output_dir, "idx_to_imgpath.json"), idx_to_imgpath
    )
    save_mapping(
        os.path.join(args.output_dir, "imgpath_to_idx.json"), imgpath_to_idx
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build mapping between paths and indexes")
    parser.add_argument(
        "--data_dir", 
        type=str, 
        required=True, 
        help="Path to the dataset directory"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        help="Output path for the map JSON", 
        default="./dino/mappings"
    )
    args = parser.parse_args()
    main(args)
    