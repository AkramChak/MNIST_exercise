import click
import torch
from torch import nn, optim
# from src.data import make_dataset
import matplotlib.pyplot as plt
from models.model import MyNeuralNet


# Moving data to GPU if possible for higher computational speed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_and_save_training_curve(history, filename='training_curve.png', folder=r'C:\School\Minor\MLOPS\MNIST_exercise\reports\figures'):
    """
    Plots and saves the training curve from the training history.

    :param history: A dictionary containing 'loss' and 'accuracy' as keys with lists of their values per epoch.
    :param filename: Name of the file to save the plot.
    :param folder: The directory where the plot will be saved.
    """
    plt.figure()
    plt.plot(history['loss'], label='Loss')
    plt.title('Training Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(f"{folder}/{filename}")
    plt.close()

@click.group()
def cli():
    """Command line interface."""
    pass

"""
    This is a decorator provided by the click library, 
    which is used to define an option for a command-line interface. 
    It allows you to specify command-line arguments that can be passed to your script.
"""

@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
@click.option("--batch_size", default=64, help="batch size for training")
@click.option("--epochs", default=10, help="number of training epochs")

# # Hyperparameters
# lr = 1e-3
# batch_size = 64
# epochs = 10

def train(lr, batch_size, epochs):
    # RUN CODE: python train_model.py train --lr 0.001 --batch_size 64 --epochs 10
    print('Training')
    print('learning rate', lr)
    print('batch size', batch_size)
    print('Epochs', epochs)

    # Able dropout during training 
    model = MyNeuralNet()
    
    """
        Function mnist() returns two values, 
        and you're interested in capturing only the first of these returned values into the variable train_set. 
        The underscore _ is used as a placeholder for the second value which you're choosing to ignore.
    """
        
    """
        Dataset object train_set containing training data, which have been created using 'TensorDataset' 
        The DataLoader object is assigned to the variable trainloader. This object is iterable. 
        When you iterate over trainloader in a training loop, it will yield batches of data from train_set, 
        each batch being of size batch_size.
        
        You would iterate over trainloader to get your data in batches. For each iteration, 
        the trainloader provides a batch of data that you can feed into your model for training.
        This way of loading data is very efficient and helps in managing memory usage, 
        as only a portion of the dataset is loaded and processed at a time.
    """
    train_set = torch.load(r'C:\School\Minor\MLOPS\MNIST_exercise\data\processed\train_data_tensor.pt')
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False)
    
    # Type of loss function
    criterion = nn.NLLLoss()
    # Optimization algorithm used for training neural networks, by updating the weights and biases
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Initialize the history dictionary
    history = {'loss': []}
    
    for e in range(epochs):
        model.train()
        # The passing each time one batch of the trainloader, part of the whole data
        for batch in trainloader:
            optimizer.zero_grad()
            batch_data, batch_labels = batch
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            logits = model(batch_data)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {e} Loss {loss}")
        # Append the loss of this epoch to the history
        history['loss'].append(loss.item())
        
    torch.save(model, r'C:\School\Minor\MLOPS\MNIST_exercise\models\trained_model.pt')
    plot_and_save_training_curve(history)
    
cli.add_command(train)
if __name__ == "__main__":
    cli()