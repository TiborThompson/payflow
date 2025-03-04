#!/bin/bash
# This script sets up and runs the Payment Transaction Fraud Detection System.
# It simulates payment transactions and detects fraudulent ones using a neural network.

# Set up the environment using the provided setup script
chmod +x setup_env.sh
./setup_env.sh

# Run the main script to start the fraud detection process
python main.py

# You can also specify custom parameters:
# python main.py --num_transactions 1000 --num_epochs 10