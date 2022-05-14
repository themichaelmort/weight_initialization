# Weight Initialization
The journey of a thousand steps could be a heck of a lot shorter if you'd just start from the right place. 

In this project, I explored how 3 methods of initializing weights from deep learning literature affected the training of a convolutional neural network performing a simple classification task (labeling pictures of clothes according to 10 categories.) I compared loss curves and accuracy curves over the course of training with each of the 3 weight initialization strategies. See the results for yourself below.

## At a Glance

* Dataset - FashionMNIST, a collection of color images of clothes, labeled according to 10 categoris
* Model - Custom convultional neural network (CNN) consisting of 3 layers.     
    * Channels x 10 2d convolution layer with 3x3 kernel & 1 pixel of padding
    * 10 x 15 2d convolution layer with 3x3 kernel & 1 pixel of padding
    * 15 x output_size 2d convolution layer with a 28x28 kernel and no padding
        * Note: This last layer reduces a 28x28 image to one value that can be used for classification.  
* Optimizer - Adam (although the code is set up to work with SGD as well)
* Loss - Pytorch's CrossEntropyLoss
* Result Visualization - Plots and saves the loss for training and validation and the accuracy during training. Examples shown below.

## Weight Initialization Strategies
* Uniform : Weights randomly initialized based on a uniform distribution
* Xe (aka Xavier) : Weights randomly initialized based on a uniform distribution, scaled by the square root of the number of input channels        (https://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization)
* Orthogonal : Weights form an orthogonal set. Orthogonal set created via 
        singular value decomposition. (https://arxiv.org/abs/1312.6120)

# Results
Loss plots on the left and accuracy plots on the right (if you are viewing this on a sufficiently wide screen.)

## Uniform

![](https://github.com/themichaelmort/weight_initialization/blob/main/Uniform%20Initialization%20Loss.png)
![](https://github.com/themichaelmort/weight_initialization/blob/main/Uniform%20Initialization%20Accuracy.png)

## Xe (or Xavier)

![](https://github.com/themichaelmort/weight_initialization/blob/main/Xe%20Initialization%20Loss.png)
![](https://github.com/themichaelmort/weight_initialization/blob/main/Xe%20Initialization%20Accuracy.png)

## Orthogonal

![](https://github.com/themichaelmort/weight_initialization/blob/main/Orthogonal%20Initialization%20Loss.png)
![](https://github.com/themichaelmort/weight_initialization/blob/main/Orthogonal%20Initialization%20Accuracy.png)
