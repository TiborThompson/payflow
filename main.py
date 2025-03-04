from typing import List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from transaction import Transaction
from data_generator import generate_transactions
from fraud_detection_model import FraudDetectionModel, train_model, evaluate_model
from processor import prepare_data

def main() -> None:
    """
    Main function to run data generation, preparation, model initialization, training, and evaluation.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Fraud Detection Model Training and Evaluation")
    parser.add_argument('--num_transactions', type=int, default=1000, help='Number of transactions to generate')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    args = parser.parse_args()

    print("Starting Payflow fraud detection system...")
    
    # Generate synthetic transaction data
    print(f"Generating {args.num_transactions} synthetic transactions...")
    transactions, labels = generate_transactions(args.num_transactions)
    
    # Prepare the data
    print("Preparing data for training and testing...")
    train_loader, test_loader = prepare_data(transactions, labels)
    
    # Initialize the model
    input_size = transactions[0].to_vector().size(0)
    model = FraudDetectionModel(input_size)
    
    # Define the loss criterion and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    print(f"Training model for {args.num_epochs} epochs...")
    train_model(model, train_loader, criterion, optimizer, num_epochs=args.num_epochs)
    
    # Evaluate the model
    print("Evaluating model performance...")
    accuracy = evaluate_model(model, test_loader)
    print(f"Model accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()