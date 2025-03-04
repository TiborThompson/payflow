import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
from transaction import Transaction

class TransactionDataset(Dataset):
    """Custom Dataset for loading Transaction data."""

    def __init__(self, transactions: List[Transaction], labels: List[int]) -> None:
        """
        Initialize the dataset with transactions and labels.

        :param transactions: List of Transaction objects.
        :param labels: List of corresponding labels (0 for normal, 1 for fraud).
        """
        self.transactions = transactions
        self.labels = labels

    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.transactions)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample by index.

        :param idx: Index of the sample to retrieve.
        :return: A tuple of (transaction_vector, label).
        """
        transaction = self.transactions[idx]
        label = self.labels[idx]
        transaction_vector = transaction.to_vector()
        return transaction_vector, label

class FraudDetectionModel(nn.Module):
    """Feedforward neural network model for fraud detection."""

    def __init__(self, input_size: int) -> None:
        """
        Initialize the neural network layers.

        :param input_size: The size of the input feature vector.
        """
        super(FraudDetectionModel, self).__init__()
        self.input_size = input_size
        # Define network layers
        self.fc1 = nn.Linear(input_size, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Define the forward pass of the network.

        :param x: Input tensor.
        :return: Output tensor after passing through the network.
        """
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out

def train_model(model: nn.Module, train_loader: DataLoader, criterion: any, optimizer: any, num_epochs: int) -> None:
    """
    Train the fraud detection model.

    :param model: The neural network model to train.
    :param train_loader: DataLoader for training data.
    :param criterion: Loss function.
    :param optimizer: Optimizer for updating model parameters.
    :param num_epochs: Number of epochs for training.
    """
    device = torch.device('cpu')
    model.to(device)
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device).float().view(-1, 1)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

def evaluate_model(model: nn.Module, test_loader: DataLoader) -> float:
    """
    Evaluate the model's performance on test data.

    :param model: The trained model to evaluate.
    :param test_loader: DataLoader for test data.
    :return: Accuracy of the model on the test set.
    """
    device = torch.device('cpu')
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            predicted = (outputs >= 0.5).float().view(-1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy