import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from tqdm import tqdm
import os



# Create Network (extended torch.Module)
class LinearNetwork(nn.Module):
    """
    Create a custom fully-connected neural network with 2 layers of 1000 neurons.
    """

    def __init__(self, dataset):
        super(LinearNetwork, self).__init__()
        x, y = dataset[0]   # grab an image from the dataset
        c, h, w = x.size()  # channel, height, width of image
        out_dim = 10        # number of classification classes

        self.net = nn.Sequential(nn.Linear(c * h * w, 1000),
                                nn.ReLU(),
                                nn.Linear(1000, out_dim))

    def forward(self, x):
        n, c, h, w = x.size()   # batch, channels, height, width
        flattened = x.view(n, c*h*w)
        return self.net(flattened)


# Dataset class for FashionMNIST (extended torch.utils.data Dataset class)
class FashionMNISTProcessedDataset(Dataset):
    def __init__(self, root, train=True):
        self.data = datasets.FashionMNIST(root,
                                     train=train,
                                     transform=transforms.ToTensor(),
                                     download=True)

    def __getitem__(self, i):
        x, y = self.data[i]     #x is the image as a tensor, y is the class label (int)
        return x, y

    def __len__(self):
        return len(self.data)   #number of items in our dataset


def plot_training_and_validation(train_losses, validation_losses):
    """
    Create a plot of your loss over time.

    Parameters
        - train_losses (list) : training loss values for each epoch
        - validation_losses (list) : validation loss values for each epoch

    Returns
        - fig (fig) : plot of validation and training over time
    """

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(len(validation_losses)),
            validation_losses, label="Validation Loss")
    ax.plot(range(len(train_losses)), train_losses, label='Training Loss')
    plt.legend()
    plt.ylabel("loss")
    plt.xlabel("epoch")

    return fig


def main(path_to_fashion_MNIST_dataset='/tmp/fashionmnist',
        num_epochs=100,
        visualize_training=True,
        plot_save_path=""
        ):
    """
    Load dataset (FashionMNIST), build a linear network, and run num_epochs of training and validation
    to classify 10 image classes. Optionally plot and save the training and validation curves.

    Parameters
        - path_to_fashion_MNIST_dataset (str) : path where fashion mnist dataset will be saved. Default is tmp/fashiomnist
            assume you have a folder in the current working directory.
        - num_epochs (int) : number of epochs to train the model. Default is 100.
        - visualize_training (bool) : flag for whether to generate a plot of the training and validation. Default True.
        - plot_save_path (str) : string for where to save the generated output path. Default is current working directory.

    Return
        - None
    """

    # Instantiate the train and validation sets
    train_dataset = FashionMNISTProcessedDataset(
        path_to_fashion_MNIST_dataset, train=True)
    val_dataset = FashionMNISTProcessedDataset(
        path_to_fashion_MNIST_dataset, train=False)

    # Instantiate data loaders
    train_loader = DataLoader(train_dataset, batch_size=42, pin_memory=True)
    validation_loader = DataLoader(val_dataset, batch_size=42)

    # Instantiate your model and loss and optimizer functions
    model = LinearNetwork(train_dataset)
    model = model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=1e-4)
    objective = torch.nn.CrossEntropyLoss()  # objective function

    # Set up some other pieces
    cntr = 0        # counter
    train_losses = []
    validation_losses = []

    # Include a progress bar for training
    loop = tqdm(total=len(train_loader)*num_epochs, position=0)

    # Run training / validation loops
    for epoch in range(num_epochs):

        # Reset batch counter
        batch = 0

        for x, y_truth in train_loader:

            # learn
            x, y_truth = x.cuda(), y_truth.cuda()
            optimizer.zero_grad()

            y_hat = model(x)  # make prediction
            loss = objective(y_hat, y_truth)  # check prediction

            # Validation every 10 epochs at the start of
            if epoch % 10 == 0 and batch == 0:
                train_losses.append(loss.item())
                validation_loss_list = []
            for val_x, val_y_truth in validation_loader:
                val_x, val_y_truth = val_x.cuda(), val_y_truth.cuda()
                val_y_hat = model(val_x)
                # Calculate the loss for the validation
                validation_loss_list.append(objective(val_y_hat, val_y_truth))

            validation_losses.append(
                (sum(validation_loss_list)/float(len(validation_loss_list))).item())

            cntr += 1

            loop.set_description(
                f'epoch:{epoch} batch:{batch} loss:{loss.item()} val_loss:{validation_losses[-1]}')

            # Backpropagation
            loss.backward()   # backpropagate the accumulated losses
            optimizer.step()  # take a step in the direction of the gradient

            # Updated batch counter
            batch += 1

    loop.close()

    # Visualizie the training and validation if requested
    if visualize_training == True:
        plot_fig = plot_training_and_validation(
            train_losses, validation_losses)
        plot_fig.savefig(os.path.join(plot_save_path, "test_and_validation"))

    pass



if __name__ == "__main__":
    # Check whether we have cuda (This code presupposes we can use cuda.)
    assert torch.cuda.is_available()
    
    # Do deep learning
    main()
