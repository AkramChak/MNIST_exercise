import click
import torch
from torch import nn, optim
# from src.data import make_dataset
import matplotlib.pyplot as plt
from src.models.model import MyNeuralNet

# Moving data to GPU if possible for higher computational speed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@click.group()
def cli():
    """Command line interface."""
    pass

"""
    This is a decorator provided by the click library, 
    which is used to define an option for a command-line interface. 
    It allows you to specify command-line arguments that can be passed to your script.
"""

"""
    Model_checkpoint is used to pass the saved model you want to evaluate
"""    
@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    # RUN CODE: python evaluate_model.py evaluate {name of torch.save file}
    print(model_checkpoint)

    # Loading the model.pt file that is made and save in the train function
    model = torch.load(model_checkpoint)
    # Taking only the second return of mnist() function ignoring the first one
    test_set = torch.load(r'C:\School\Minor\MLOPS\MNIST_exercise\data\processed\test_data_tensor.pt')
    # Creating dataloader called testloader for test_data
    testloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)
    
    # Lists to save results
    test_preds = [ ]
    test_labels = [ ]
     
    # Disabling autograd
    with torch.no_grad():
        # Looping through each batch of the testloader
        for batch in testloader:
            batch_data, batch_labels = batch
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            logits = model(batch_data)
            test_preds.append(logits.argmax(dim=1).cpu())
            test_labels.append(batch_labels.cpu())

    # Stack data vertically
    test_preds = torch.cat(test_preds, dim=0)
    test_labels = torch.cat(test_labels, dim=0)

    print('Accuracy:', (test_preds == test_labels).float().mean())


cli.add_command(evaluate)
if __name__ == "__main__":
    cli()