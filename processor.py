# processor.py
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from transaction import Transaction
from fraud_detection_model import FraudDetectionModel, train_model, evaluate_model

def prepare_data(transactions: List[Transaction], labels: List[int]) -> Tuple[DataLoader, DataLoader]:
    """
    Prepare the data for training and testing.

    :param transactions: List of Transaction objects.
    :param labels: List of corresponding labels (0 for normal, 1 for fraud).
    :return: A tuple containing the training DataLoader and testing DataLoader.
    """
    # Convert all transactions to vectors
    all_vectors = torch.stack([transaction.to_vector() for transaction in transactions])

    # Compute mean and std for normalization
    mean = torch.mean(all_vectors, dim=0)
    std = torch.std(all_vectors, dim=0)

    # Handle zero std to avoid division by zero
    std[std == 0] = 1.0

    # Define a normalization function
    def normalize_vector(vector: torch.Tensor) -> torch.Tensor:
        return (vector - mean) / std

    # Create a custom dataset that applies normalization
    class NormalizedTransactionDataset(torch.utils.data.Dataset):
        """Dataset that normalizes transaction vectors."""

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
            :return: A tuple of (normalized_transaction_vector, label).
            """
            transaction = self.transactions[idx]
            label = self.labels[idx]
            vector = transaction.to_vector()
            normalized_vector = normalize_vector(vector)
            return normalized_vector, label

    # Create the dataset
    dataset = NormalizedTransactionDataset(transactions, labels)

    # Split the dataset into training and testing sets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoaders for training and testing
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader