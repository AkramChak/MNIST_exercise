# from tests import _PATH_DATA
import pytest
import torch
import os
from src.data.make_dataset import mnist

def test_dataset_creation():
    # Step 1: Trigger the function that creates the .pt files
    mnist()

    # Step 2: Check if the .pt files exist
    assert os.path.exists(r'C:\School\Minor\MLOPS\MNIST_exercise\data\processed\train_data_tensor.pt'), "Training data file not created."
    assert os.path.exists(r'C:\School\Minor\MLOPS\MNIST_exercise\data\processed\test_data_tensor.pt'), "Test data file not created."

    # Step 3: Load the .pt files and verify their contents
    # Load the .pt files and verify their contents
    training_data = torch.load(r'C:\School\Minor\MLOPS\MNIST_exercise\data\processed\train_data_tensor.pt')
    test_data = torch.load(r'C:\School\Minor\MLOPS\MNIST_exercise\data\processed\test_data_tensor.pt')

    # Updated assertions
    assert isinstance(training_data, torch.utils.data.TensorDataset), "Training data is not a TensorDataset."
    assert isinstance(test_data, torch.utils.data.TensorDataset), "Test data is not a TensorDataset."
