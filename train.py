import os
import sys
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, models
from sklearn.metrics import fbeta_score

from data.DufercoDataset import DufercoDataset
from data.transforms import train_transforms, test_transforms
from models.EfficientNet import EfficientNetBinaryClassifier
from models.trainer import train_model


def prepare_dataloaders(data_config_path, batch_size):
    """
    Prepares training and validation data loaders.
    
    Args:
        data_config_path (str): Path to the dataset split JSON.
        batch_size (int): Batch size for data loading.
        
    Returns:
        tuple: Training and validation data loaders.
    """
    train_dataset = DufercoDataset(
        data_config_path, 
        split="train", 
        transform=train_transforms
    )
    val_dataset = DufercoDataset(
        data_config_path, 
        split="val", 
        transform=test_transforms
    )

    sample_weights = train_dataset.get_sample_weights()
    sampler = WeightedRandomSampler(
        weights=sample_weights, 
        num_samples=len(sample_weights), 
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=sampler
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, val_loader


def configure_model(device):
    """
    Configures the model for training.
    
    Args:
        device (torch.device): Device to use for computation.
        
    Returns:
        nn.Module: The configured model.
    """
    model = EfficientNetBinaryClassifier()
    model = nn.DataParallel(model)  # Enable multi-GPU support
    return model.to(device)


def configure_training(model, learning_rate, weight_decay):
    """
    Configures the loss function and optimizer for training.
    
    Args:
        model (nn.Module): The model to train.
        learning_rate (float): Learning rate for the optimizer.
        
    Returns:
        tuple: The loss function and optimizer.
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay
    )
    return criterion, optimizer


def create_logging_directory(base_path):
    """
    Creates a timestamped logging directory for TensorBoard and checkpoints.
    
    Args:
        base_path (str): Base path for logs and checkpoints.
        
    Returns:
        tuple: Log directory and checkpoint path.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("runs", f"experiment_{timestamp}")
    checkpoint_path = os.path.join(base_path, timestamp)
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)
    
    return log_dir, checkpoint_path


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    train_loader, val_loader = prepare_dataloaders(
        args.data_config_path, 
        args.batch_size
    )

    model = configure_model(device)
    criterion, optimizer = configure_training(
        model, 
        args.learning_rate,
        args.weight_decay
    )
    log_dir, checkpoint_path = create_logging_directory(args.checkpoint_path)
    writer = SummaryWriter(log_dir=log_dir)
   
    train_model(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        args.num_epochs,
        writer, 
        device, 
        checkpoint_path
    )
    writer.close()  


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train EfficientNet on Duferco dataset")
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
        "--num_epochs",
        type=int,
        default=20,
        help="Number of epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Number of epochs"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="Loss weight decay"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Checkpoint path"
    )
    return parser.parse_args()


if __name__ == "__main__":    
    args = parse_arguments()
    main(args)
