from datetime import datetime
import torch
from typing import Any

class Transaction:
    """Represents a payment transaction with attributes like amount, sender, receiver, timestamp, and fraud status."""
    
    def __init__(self, amount: float, sender: str, receiver: str, timestamp: datetime, is_fraud: bool = False) -> None:
        """
        Initialize a new Transaction instance.
        
        :param amount: The amount of the transaction.
        :param sender: The sender's identifier.
        :param receiver: The receiver's identifier.
        :param timestamp: The timestamp when the transaction occurred.
        :param is_fraud: Flag indicating whether the transaction is fraudulent.
        """
        # Set the amount of the transaction
        self.amount: float = amount
        
        # Set the sender of the transaction
        self.sender: str = sender
        
        # Set the receiver of the transaction
        self.receiver: str = receiver
        
        # Set the timestamp of the transaction
        self.timestamp: datetime = timestamp
        
        # Set the fraud status of the transaction
        self.is_fraud: bool = is_fraud
        
    def to_vector(self) -> torch.Tensor:
        """
        Convert the transaction's numerical features into a torch tensor.
        
        :return: A torch tensor containing the numerical representation of the transaction.
        """
        # Extract numerical features: amount and timestamp converted to a numerical value
        amount = self.amount
        timestamp_value = self.timestamp.timestamp()  # Convert datetime to timestamp (float)
        
        # Create a tensor from the numerical features
        vector = torch.tensor([amount, timestamp_value], dtype=torch.float32)
        
        return vector