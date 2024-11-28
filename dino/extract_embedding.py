import argparse
import json
import os
from tqdm import tqdm
from PIL import Image
import faiss
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel


def read_json(input_path):
    """Reads a JSON file and returns a dictionary."""
    with open(input_path, "r") as f:
        input_file = json.load(f)
    return input_file


def save_feature_vector(idx, vector, save_dir="./dino/dino_features"):
    """Saves a feature vector to a .npy file."""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{idx:05d}.npy")
    np.save(save_path, vector)


def argument_parser():
    parser = argparse.ArgumentParser(description="DINO features extraction")
    parser.add_argument(
        "--data_config_path", 
        type=str, 
        help="Path to the JSON mapping file", 
        required=True
    )
    parser.add_argument(
        "--index_path", 
        type=str, 
        help="Output path for the FAISS index", 
        required=True
    )
    parser.add_argument(
        "--dino_model", 
        type=str, 
        help="DINOv2 model 'dimension'", 
        default="dinov2-giant"
    )
    parser.add_argument(
        "--embedding_size",
        type=int,
        help="Size of the embeddings",
        default=1536
    )
    return parser.parse_args()


def main(args):
    """Main function to process images and generate embeddings."""
    data_config = read_json(args.data_config_path) 

    # Load the DINOv2 model and processor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")
    processor = AutoImageProcessor.from_pretrained(f"facebook/{args.dino_model}")
    model = AutoModel.from_pretrained(f"facebook/{args.dino_model}").to(device)

    # FAISS index
    faiss_index = faiss.IndexFlatL2(args.embedding_size)
        
    for idx, img_path in tqdm(data_config.items(), desc="Processing images"):
        img = Image.open(img_path).convert("RGB")

        # Generate embeddings
        with torch.no_grad():
            inputs = processor(images=img, return_tensors="pt").to(device)
            outputs = model(**inputs)
    
        features = outputs.last_hidden_state
        embedding = features.mean(dim=1)
        vector = embedding.detach().cpu().numpy().astype(np.float32)

        # Save and normalize the vector
        save_feature_vector(int(idx), vector)
        faiss.normalize_L2(vector)
        faiss_index.add(vector)

    faiss.write_index(faiss_index, args.index_path)  


if __name__ == "__main__":    
    args = argument_parser()
    main(args)
