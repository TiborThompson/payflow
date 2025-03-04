# Payment Transaction Fraud Detection System

## Overview

This project simulates a payment transaction system and implements a simple fraud detection model using a neural network built with PyTorch. The system generates random transaction data, including both normal and fraudulent transactions, to create a dataset for training and testing the fraud detection model. The goal is to detect fraudulent transactions based on various attributes of each transaction.

## File Structure

- **transaction.py**: Defines a `Transaction` class representing a payment transaction with attributes such as amount, sender, receiver, and timestamp. It includes a method to convert the transaction into a numerical vector suitable for input into the neural network model.

- **data_generator.py**: Programmatically generates random transaction data, including both normal and fraudulent transactions. This simulates a realistic dataset for testing and training the fraud detection model.

- **fraud_detection_model.py**: Defines a simple neural network model using PyTorch. The model is designed to detect fraudulent transactions based on transaction data by learning patterns and anomalies.

- **processor.py**: Processes transactions by preparing data for the model. It handles data loading, preprocessing, splitting into training and testing sets, training the fraud detection model, and provides functions to evaluate the model's performance.

- **main.py**: The main script that runs the payment simulation. It orchestrates the workflow by generating transaction data, training the fraud detection model, evaluating it, and outputting the results.

## Getting Started

### Prerequisites

Make sure you have the following installed on your system:

- Python 3.6 or later
- PyTorch
- NumPy
- pandas

### Installation

1. **Clone the repository:**

   ```sh
   git clone https://github.com/tiborthompson/payflow.git
   cd payflow
   ```

2. **Install required packages:**

   Install the necessary Python packages using pip:

   ```sh
   pip install -r requirements.txt
   ```

   Ensure that the `requirements.txt` file includes all dependencies:

   ```
   torch
   numpy
   pandas
   ```

### Running the Project

1. **Generate Transaction Data:**

   The data generation is integrated into the main script, but you can run `data_generator.py` separately if you wish to generate the dataset alone.

   ```sh
   python data_generator.py
   ```

   This will generate a dataset file (e.g., `transactions.csv`) containing both normal and fraudulent transactions.

2. **Train and Evaluate the Model:**

   Run the main script to train the fraud detection model and evaluate its performance.

   ```sh
   python main.py
   ```

   This script will:

   - Generate transaction data if not already present.
   - Prepare and preprocess the data.
   - Train the neural network model.
   - Evaluate the model using test data.
   - Output the evaluation results, including metrics like accuracy, precision, and recall.

## Detailed Description

### transaction.py

Defines the `Transaction` class representing a single payment transaction.

**Attributes:**

- `amount`: The monetary amount of the transaction.
- `sender`: Identifier for the sender account.
- `receiver`: Identifier for the receiver account.
- `timestamp`: The date and time when the transaction occurred.
- Additional attributes as needed (e.g., transaction type, location).

**Methods:**

- `to_vector()`: Converts the transaction attributes into a numerical vector for input into the neural network.

### data_generator.py

Creates synthetic transaction data to simulate a real-world payment system.

- Generates a mixture of normal and fraudulent transactions.
- Allows customization of the number of transactions and the ratio of fraudulent transactions.
- Saves the generated data to a CSV file for later use in training and evaluation.

### fraud_detection_model.py

Defines the neural network model for fraud detection using PyTorch.

**Features:**

- Comprises input, hidden, and output layers suitable for binary classification.
- Utilizes activation functions such as ReLU and Sigmoid.
- Defines a loss function (e.g., Binary Cross-Entropy) and an optimizer (e.g., Adam).

### processor.py

Handles data processing tasks necessary for training the model.

**Functions:**

- `load_data()`: Loads transaction data from a file.
- `preprocess_data()`: Cleans and preprocesses the data, including encoding categorical variables and normalizing numerical features.
- `split_data()`: Splits the dataset into training and testing sets.
- `train_model()`: Trains the neural network model using the training data.
- `evaluate_model()`: Evaluates the trained model on the test data and calculates performance metrics.

### main.py

The central script that runs the entire fraud detection process.

- Calls `data_generator.py` to generate transaction data if needed.
- Uses functions from `processor.py` to prepare data and train the model.
- Utilizes `fraud_detection_model.py` to define and initialize the model.
- Outputs results and performance metrics to the console or an output file.

## Usage Notes

- **Customization:** You can adjust parameters such as the number of transactions, proportion of fraud cases, and model hyperparameters by modifying variables in the scripts.
- **Data Persistence:** Generated data and trained models can be saved for reuse to save time on subsequent runs.
- **Logging:** The scripts may include logging statements to help track the progress of data generation, training, and evaluation.

## Example Output

After running `main.py`, you should see output similar to:

```
Generating transaction data...
Data generation complete. Total transactions: 10000
Preprocessing data...
Data split into training and testing sets.
Training the fraud detection model...
Training complete. Model saved to 'fraud_model.pth'.
Evaluating model performance...
Accuracy: 0.98
Precision: 0.97
Recall: 0.95
F1 Score: 0.96
```

## Support

If you encounter any issues or have questions, please open an issue on the GitHub repository or contact the project maintainers.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

*Note: This project is intended for educational purposes to demonstrate the process of fraud detection using machine learning techniques. The generated data and model are simplistic and should not be used for real-world fraud detection without significant enhancements and validations.*