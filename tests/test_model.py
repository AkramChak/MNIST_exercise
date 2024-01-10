import torch
from src.models.model import MyNeuralNet  # adjust the import path if necessary

def test_model_output_shape():
    # Create a dummy input tensor of shape [batch_size, 784]. Let's use a batch_size of 1 for simplicity
    dummy_input = torch.randn(1, 784)

    # Instantiate the model
    model = MyNeuralNet()

    # Get the output by passing the dummy input through the model
    output = model(dummy_input)

    # Assert that the output shape is [1, 10]
    assert output.shape == (1, 10)

# You can then run this test using a testing framework like pytest
