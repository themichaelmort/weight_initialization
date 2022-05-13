import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, utils, datasets
from tqdm import tqdm
from torch.nn.parameter import Parameter


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




class Conv2d(nn.Module):
    """Custom Convolutional Layer Class. 
    
    Allows for different initialization schemes:
    - Uniform : Weights randomly initialized based on a uniform distribution
    - Xe (aka Xavier) : Weights randomly initialized based on a uniform 
        distribution, scaled by the square root of the number of input channels
        (https://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization)
    - Orthogonal : Weights form an orthogonal set. Orthogonal set created via 
        singular value decomposition. (https://arxiv.org/abs/1312.6120)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None, initialization='Uniform'):
        
        # locals(), a dictionary of all the local variables and their values
        # update adds the locals() dict to the self.__dict__() dictionary of object attributes
        self.__dict__.update(locals())
        super(Conv2d, self).__init__()

        self.weight = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
        self.bias = Parameter(torch.Tensor(out_channels))

        #Initialize the data in the weight parameter
        if initialization == "Uniform":
            self.weight.data.uniform_(0.0,1.0).float() 
        if initialization == "Xe":
            weight_array = np.random.normal(loc=0.0, scale=np.sqrt(in_channels), size=(out_channels, in_channels, *kernel_size))
            self.weight.data = torch.from_numpy(weight_array).float()
        if initialization == "Orthogonal":
            random_array = np.random.rand(out_channels, in_channels, *kernel_size)
            u, s, weight_array = np.linalg.svd(random_array, full_matrices=True, compute_uv=True, hermitian=False)
            self.weight.data = torch.from_numpy(weight_array).float()

        #Initialize bias uniformly
        self.bias.data.uniform_(0,0)

  
    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)




class ConvNetwork(nn.Module):
    """Custom class for the Convolutional Neural Network (CNN)

    Architecture (3 layers)
    - Channels x 10 2d convolution layer with 3x3 kernel & 1 pixel of padding
    - 10 x 15 2d convolution layer with 3x3 kernel & 1 pixel of padding
    - 15 x output_size 2d convolution layer with a 28x28 kernel and no padding
        Note: This last layer reduces a 28x28 image to one value that can be 
        used for classification.    
    """

    def __init__(self, dataset, initialization="Uniform"):
        self.initialization=initialization
        super(ConvNetwork, self).__init__()   #run __init__ for parent
        x, y = dataset[0]   #data (help with size), an output label
        c, h, w = x.size()
        output = 10 #Force to 10 because 10 output labels
        

        # My custom conv2d version of the network
        self.net = nn.Sequential(
            Conv2d(c, 10, (3,3), padding=(1,1), initialization=self.initialization),
            nn.LeakyReLU(),
            Conv2d(10, 15, (3,3), padding=(1,1), initialization=self.initialization),
            nn.LeakyReLU(),
            Conv2d(15, output, (28, 28), padding=(0,0), initialization=self.initialization))

    def forward(self, x):
        return self.net(x).squeeze(2).squeeze(2) 


class CrossEntropyLoss(nn.Module):
    """
    Custom implementation of the Cross-Entropy Loss function.
    """
    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        super(CrossEntropyLoss, self).__init__()
        self.__dict__.update(locals())
        pass

    def forward(self, y_hat, y_truth):
        batch_idx = torch.arange(y_hat.size(0)) #1 number per batch in this vector
        estimate_truth = torch.logsumexp(y_hat, 1)
        loss = -y_hat[batch_idx, y_truth] + estimate_truth         
        return torch.mean(loss)



def main():

    data = FashionMNISTProcessedDataset("/")

    # Initialize Datasets
    train_dataset = FashionMNISTProcessedDataset('/tmp/fashionmnist', train=True)
    val_dataset = FashionMNISTProcessedDataset('/tmp/fashionmnist', train=False)

    # Initialize DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, pin_memory=True, shuffle=True)
    validation_loader = DataLoader(val_dataset, batch_size=16, pin_memory=True, shuffle=True)

    # NOTE:
    # Changing the batch size to something that doesn't divide happily into
    # the size of your dataset could lead to errors. Use a droplast=True (check
    # exact kwarg) to circumvent the issue.

    # Hyperparameters
    num_epochs = 1

    #Options: "Uniform", "Xe", "Orthogonal"
    init_strategies = ["Uniform", "Xe", "Orthogonal"]

    for init_strategy in init_strategies:

        # Initialize Model
        model = ConvNetwork(train_dataset, initialization=init_strategy)
        model = model.cuda()


        ## Initialize Objective and Optimizer and other parameters

        # Optimizer
        # optimizer = optim.SGD(model.parameters(), lr=1e-4)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        # Objective Function
        #objective = nn.CrossEntropyLoss()            #This is pytorch's implementation
        objective = CrossEntropyLoss()              #This is my implementation

        # Lists for saving the loss values as we go
        losses = []
        validations = []
        train_accuracy = []
        validation_accuracy = []

        # Run your training and validation loop and collect stats

        for epoch in range(3):

            # Display progress bar
            loop = tqdm(total=len(train_loader), position=0, leave=False)

            for batch, (x, y_truth) in enumerate(train_loader):
                x, y_truth = x.cuda(), y_truth.cuda()

                optimizer.zero_grad()
                y_hat = model(x)

                #Calculate the loss
                loss = objective(y_hat, y_truth)

                #Backpropagate the loss through the network
                loss.backward()

                #Take a step and update weights and biases
                optimizer.step()

                #Get the value of the loss
                loss_value = loss.item()

                losses.append(loss_value)
                train_accuracy_percent = (torch.softmax(y_hat, 1).argmax(1) == y_truth).float().mean()
                train_accuracy.append(train_accuracy_percent.item()) 

                
                loop.set_description( f'epoch:{epoch} loss:{loss_value} accuracy:{train_accuracy_percent.item()}' )
                loop.update(True)


                # Periodically check the validation            
                if batch % 700 == 0:

                    val = np.mean([objective(model(x.cuda()), y.cuda()).item() for x, y in validation_loader])
                    validations.append((len(losses), val))

                    temp_list = []
                    for x, y in validation_loader:
                        x, y = x.cuda(), y.cuda()
                        y_hat = model(x)
                        best_guess = y_hat.argmax(1)
                        compare = (y==best_guess).float()
                        accuracy = torch.sum(compare).item()/16
                        temp_list.append(accuracy)

                    validation_accuracy.append( (len(losses), np.mean(np.array(temp_list))) )

            loop.close()
        

        # Plot & Save Results
        plot_training_and_accuracy(init_strategy, losses, validations, train_accuracy, validation_accuracy)

    pass



def plot_training_and_accuracy(init_strategy, losses, validations, train_accuracy, validation_accuracy):
    
    # Training & Validation Plot
    a, b = zip(*validations)  #create 2 lists by unzipping
    plt.plot(losses, label='train')
    plt.plot(a, b, label='val')
    plt.legend()
    plt.xlabel("Training Time")
    plt.ylabel("Loss")
    plt.title(f"{init_strategy} Initialization Loss")
    plt.savefig(f"{init_strategy}_Initalization_Training")

    #Accuracy Plot
    fig, ax = plt.subplots()
    c, d = zip(*validation_accuracy)
    ax.plot(train_accuracy, label='train')
    ax.plot(c, d, label='val')
    ax.legend()
    ax.set_xlabel("Training Time")
    ax.set_ylabel("% Accuracy")
    ax.set_ylim([0,1])
    ax.grid(True)
    ax.set_title(f"{init_strategy} Initialization Accuracy")
    plt.savefig(f"{init_strategy}_Initalization_Accuracy")

    pass





if __name__ == "__main__":
    # Check whether we have cuda (This code presupposes we can use cuda.)
    assert torch.cuda.is_available()
    
    # Do deep learning
    main()