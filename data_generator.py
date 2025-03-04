# data_generator.py
from transaction import Transaction
from typing import List, Tuple
import random
from datetime import datetime, timedelta

def generate_transactions(num_transactions: int) -> Tuple[List[Transaction], List[int]]:
    """
    Generate a list of random Transaction objects and their corresponding labels.

    :param num_transactions: The number of transactions to generate.
    :return: A tuple containing a list of Transaction objects and a list of labels (1 for fraud, 0 for normal).
    """
    # Initialize lists to hold transactions and labels
    transactions: List[Transaction] = []
    labels: List[int] = []

    # Get the current time to use as a reference for timestamps
    current_time = datetime.now()

    # Loop over the number of transactions to generate each one
    for i in range(num_transactions):
        # Generate a random amount between $1 and $10,000
        amount = round(random.uniform(1.0, 10000.0), 2)

        # Generate random sender and receiver IDs (random 10-character strings)
        sender = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=10))
        receiver = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=10))

        # Generate a random timestamp within +/- 1 day from the current time
        time_offset = timedelta(seconds=random.randint(-86400, 86400))
        timestamp = current_time + time_offset

        # Determine if the transaction is fraudulent based on the amount exceeding $9,000
        is_fraud = amount > 9000.0

        # Create a new Transaction object with the generated data
        transaction = Transaction(
            amount=amount,
            sender=sender,
            receiver=receiver,
            timestamp=timestamp,
            is_fraud=is_fraud
        )

        # Append the transaction and its label to the respective lists
        transactions.append(transaction)
        labels.append(1 if is_fraud else 0)

    # Return the list of transactions and their labels
    return transactions, labels