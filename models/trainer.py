import time
from tqdm import tqdm
from sklearn.metrics import fbeta_score
import torch


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs,
    writer,
    device,
    checkpoint_path,
    beta=0.5,
    pred_threshold=0.5,
    save_each_n_epoch=5,
):
    """
    Train and evaluate the model.

    Args:
        model (nn.Module): The PyTorch model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimization algorithm.
        num_epochs (int): Number of training epochs.
        writer (SummaryWriter): TensorBoard writer.
        device (torch.device): Device to use for training.
        checkpoint_path (str): Path to save model checkpoints.
        beta (float, optional): Beta value for F-beta score. Defaults to 0.5.
        pred_threshold (float, optional): Threshold for prediction probability.
        save_each_n_epoch (int): Save each n epochs the model checkpoints

    Returns:
        None
    """    
    model.train() 

    for epoch in range(num_epochs):
        # Initialize epoch metrics
        running_loss = 0.0
        start_time = time.time()
        all_labels = []
        all_predictions = []

        # Training loop
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            labels = labels.unsqueeze(1).float()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss and collect predictions
            running_loss += loss.item()
            predicted = (outputs > pred_threshold).float()
            all_labels.extend(labels.cpu().numpy().astype(int))
            all_predictions.extend(predicted.cpu().numpy().astype(int))

        # Compute training metrics
        epoch_loss = running_loss / len(train_loader)
        train_fbeta = fbeta_score(all_labels, all_predictions, beta=beta, average="binary")
        epoch_time = time.time() - start_time

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, "
            f"F-beta Score: {train_fbeta:.4f}, Time: {epoch_time:.2f}s"
        )

        # Log training metrics to TensorBoard
        writer.add_scalar("Training/Loss", epoch_loss, epoch)
        writer.add_scalar("Training/F-beta Score", train_fbeta, epoch)

        # Validation phase
        val_loss, val_fbeta = validate_model(
            model, val_loader, criterion, device, beta
        )

        # Log validation metrics to TensorBoard
        writer.add_scalar("Validation/Loss", val_loss, epoch)
        writer.add_scalar("Validation/F-beta Score", val_fbeta, epoch)

        print(f"Validation Loss: {val_loss:.4f}, F-beta Score: {val_fbeta:.4f}")

        # Save model checkpoint every 5 epochs
        if (epoch + 1) % save_each_n_epoch == 0 and checkpoint_path:
            save_checkpoint(
                model, optimizer, epoch + 1, epoch_loss, checkpoint_path
            )


def validate_model(
        model, 
        val_loader, 
        criterion, 
        device, 
        beta=0.5, 
        pred_threshold=0.5
):
    """
    Perform validation on the model checkpoint.

    Args:
        model (nn.Module): The PyTorch model to validate.
        val_loader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to use for computation.
        beta (float, optional): Beta value for F-beta score. Defaults to 0.5.

    Returns:
        tuple: Validation loss and F-beta score.
    """
    model.eval()
    val_running_loss = 0.0
    val_all_labels = []
    val_all_predictions = []

    with torch.no_grad():
        for val_images, val_labels in val_loader:
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            val_labels = val_labels.unsqueeze(1).float()

            # Forward pass
            val_outputs = model(val_images)
            val_loss = criterion(val_outputs, val_labels)

            # Accumulate loss and collect predictions
            val_running_loss += val_loss.item()
            val_predicted = (val_outputs > pred_threshold).float()
            val_all_labels.extend(val_labels.cpu().numpy().astype(int))
            val_all_predictions.extend(val_predicted.cpu().numpy().astype(int))

    # Compute validation metrics
    val_epoch_loss = val_running_loss / len(val_loader)
    val_fbeta = fbeta_score(
        val_all_labels, val_all_predictions, beta=beta, average="binary"
    )

    return val_epoch_loss, val_fbeta


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path):
    """
    Save the model checkpoint.

    Args:
        model (nn.Module): The PyTorch model to save.
        optimizer (Optimizer): Optimization algorithm.
        epoch (int): Current epoch number.
        loss (float): Current loss value.
        checkpoint_path (str): Path to save the checkpoint.

    Returns:
        None
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    checkpoint_file = f"{checkpoint_path}/model_epoch_{epoch}.pt"
    torch.save(checkpoint, checkpoint_file)
    print(f"Checkpoint saved at: {checkpoint_file}")
