import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, fbeta_score, confusion_matrix
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.DufercoDataset import DufercoDataset
from data.transforms import test_transforms
from models.EfficientNet import EfficientNetBinaryClassifier


def prepare_test_loader(data_config_path, batch_size):
    """
    Prepares the test data loader.
    
    Args:
        data_config_path (str): Path to dataset configuration file.
        batch_size (int): Batch size for the test DataLoader.
        
    Returns:
        DataLoader: DataLoader for the test dataset.
    """
    test_dataset = DufercoDataset(
        data_config_path, 
        split="test", 
        transform=test_transforms
    )
    return DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )


def load_model(model_path, device):
    """
    Loads a trained model from a checkpoint.
    
    Args:
        model_path (str): Path to the trained model checkpoint.
        device (torch.device): Device to load the model on.
        
    Returns:
        nn.Module: Loaded model.
    """
    try:
        model = EfficientNetBinaryClassifier()
        model = nn.DataParallel(model)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model.to(device)
    except FileNotFoundError:
        raise ValueError(f"Model checkpoint not found at {model_path}")
    except KeyError:
        raise ValueError("Checkpoint is missing the 'model_state_dict' key")


def evaluate_model(model, test_loader, device, beta=0.5):
    """
    Evaluates the model on the test dataset and computes metrics.
    
    Args:
        model (nn.Module): The trained model to evaluate.
        test_loader (DataLoader): DataLoader for the test dataset.
        device (torch.device): Device for computation.
        beta (float): Beta value for the F-beta score.
    """
    model.eval()
    all_labels, all_predictions = [], []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            labels = labels.unsqueeze(1).to(torch.float32)

            outputs = model(images)
            predictions = (outputs > 0.5).float()

            all_labels.extend(labels.cpu().numpy().astype(int))
            all_predictions.extend(predictions.cpu().numpy().astype(int))

    all_labels = np.array(all_labels).flatten()
    all_predictions = np.array(all_predictions).flatten()

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    fbeta = fbeta_score(all_labels, all_predictions, beta=beta)
    confusion = confusion_matrix(all_labels, all_predictions)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"F-beta Score (beta={beta}): {fbeta:.4f}")
    print(f"Confusion Matrix:\n{confusion}")


def argument_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate EfficientNet on Duferco test dataset"
    )
    parser.add_argument(
        "--data_config_path", 
        type=str, 
        required=True, 
        help="Path to dataset JSON"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.5,
        help="Beta value for F-beta score"
    )
    return parser.parse_args()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on: {device}")

    test_loader = prepare_test_loader(
        args.data_config_path,
        args.batch_size
    )
    model = load_model(args.model_path, device)
    
    evaluate_model(model, test_loader, device, beta=args.beta)


if __name__ == "__main__":
    args = argument_parser()
    main(args)
