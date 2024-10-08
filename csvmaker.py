from data_preparation import load_dataset
from model import SParamCNN
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
import pickle
import numpy as np
from sklearn.metrics import r2_score

def train_model(train_dataloader, val_dataloader, model, criterion, optimizer, scheduler, num_epochs, device):
    # Extracting the number of frequencies from the first batch of data in the train dataset
    num_frequencies = train_dataloader.dataset.dataset.tensors[1].size(2)
    
    # Initialize lists to store epoch-wise average training and validation loss
    epoch_train_losses = []
    epoch_val_losses = []
    
    # Initialize lists to store R^2 scores for all outputs, for all epochs
    train_r2_scores = []
    val_r2_scores = []

    # Initialize lists to store losses for each frequency separately
    train_losses_per_frequency = [[] for _ in range(num_frequencies)]
    val_losses_per_frequency = [[] for _ in range(num_frequencies)]
    train_losses_per_frequency_recon = [[] for _ in range(num_frequencies)]
    val_losses_per_frequency_recon = [[] for _ in range(num_frequencies)]

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        true_train, preds_train = [], []
        for i, (inputs, outputs) in enumerate(train_dataloader):
            inputs, outputs = inputs.to(device), outputs.to(device)
            optimizer.zero_grad()
            predictions, predictions_recon = model(outputs)
            
            # Calculate loss for each output and frequency separately
            loss = torch.tensor(0.0, device=device)
            loss_recon = torch.tensor(0.0, device=device)

            # Reconstruction Loss
            for i in range(inputs.size(1)):  # Loop over outputs
                for j in range(inputs.size(2)):  # Loop over frequencies within each output
                    current_loss_recon = criterion(predictions_recon[:, i, j], inputs[:, i, j])
                    loss_recon += current_loss_recon
                    train_losses_per_frequency_recon[j].append(current_loss_recon.item())  # Append loss for this frequency

            # Main Loss
            for i in range(outputs.size(1)):  # Loop over outputs
                for j in range(outputs.size(2)):  # Loop over frequencies within each output
                    current_loss = criterion(predictions[:, i, j], outputs[:, i, j])
                    loss += current_loss
                    train_losses_per_frequency[j].append(current_loss.item())  # Append loss for this frequency

            # Combine losses with scaling factors
            alpha = 0.5  # Weight for reconstruction loss
            beta = 1.0   # Weight for main loss
            total_loss = alpha * loss_recon + beta * loss
            total_loss.backward()

            optimizer.step()
            scheduler.step()

            running_loss += total_loss.item() * inputs.size(0)
            
            true_train.append(outputs.detach().cpu().numpy())
            preds_train.append(predictions.detach().cpu().numpy())

        epoch_train_loss = running_loss / len(train_dataloader.dataset)
        epoch_train_losses.append(epoch_train_loss)

        model.eval()
        running_loss = 0.0
        true_val, preds_val = [], []
        with torch.no_grad():
            for inputs, outputs in val_dataloader:
                inputs, outputs = inputs.to(device), outputs.to(device)
                predictions, predictions_recon = model(outputs)
                
                # Calculate loss for each output and frequency separately
                loss = torch.tensor(0.0, device=device)
                for i in range(outputs.size(1)):  # Loop over outputs
                    for j in range(outputs.size(2)):  # Loop over frequencies within each output
                        current_loss = criterion(predictions[:, i, j], outputs[:, i, j])
                        loss += current_loss
                        val_losses_per_frequency[j].append(current_loss.item())  # Append loss for this frequency
                
                running_loss += loss.item() * inputs.size(0)
                
                true_val.append(outputs.cpu().numpy())
                preds_val.append(predictions.cpu().numpy())

        epoch_val_loss = running_loss / len(val_dataloader.dataset)
        epoch_val_losses.append(epoch_val_loss)

        true_train = np.concatenate(true_train, axis=0)
        preds_train = np.concatenate(preds_train, axis=0)
        true_val = np.concatenate(true_val, axis=0)
        preds_val = np.concatenate(preds_val, axis=0)

        # Calculate R^2 scores for the current epoch and append them to their respective lists
        epoch_train_r2 = [[r2_score(true_train[:, j, i], preds_train[:, j, i]) for i in range(num_frequencies)] for j in range(len(true_train[0]))]
        epoch_val_r2 = [[r2_score(true_val[:, j, i], preds_val[:, j, i]) for i in range(num_frequencies)] for j in range(len(true_val[0]))]
        train_r2_scores.append(epoch_train_r2)
        val_r2_scores.append(epoch_val_r2)

        print(f"\nEpoch {epoch+1}/{num_epochs}, Training Loss: {epoch_train_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}")
    
    print("\nTraining complete")

    # Save the loss for each frequency for each output for the final epoch
    final_epoch_losses = {}
    for j in range(num_frequencies):
        final_epoch_losses[f'Frequency_{j+1}'] = {
            'Train_Losses': train_losses_per_frequency[j],
            'Val_Losses': val_losses_per_frequency[j]
        }

    with open("final_epoch_losses.pkl", "wb") as f:
        pickle.dump(final_epoch_losses, f)

    print("Final epoch losses saved.")

    # Save the results
    results = {
        "train_loss": epoch_train_losses,
        "val_loss": epoch_val_losses,
        "train_r2_scores": train_r2_scores,
        "val_r2_scores": val_r2_scores,
    }

    with open("training_results.pkl", "wb") as f:
        pickle.dump(results, f)

    print("Training results saved.")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    full_dataset = load_dataset()

    # Split dataset into training and validation
    dataset_size = len(full_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Prepare data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=10)

    # Initialize the model and move it to the designated device
    model = SParamCNN().to(device)

    # Define the loss function and optimizer
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Number of epochs to train the model
    num_epochs = 100

    # Train the model
    train_model(train_dataloader, val_dataloader, model, criterion, optimizer, scheduler, num_epochs, device)

    # Save the model
    torch.save(model.state_dict(), 'statedict_rounding.pt')

if __name__ == "__main__":
    main()
