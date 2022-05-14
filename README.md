# Weight Initialization
The journey of a thousand steps could be a heck of a lot shorter if you'd just start from the right place. 

In this project, I explored how 3 methods of initializing weights from deep learning literature affected the training of a convolutional neural network performing a simple classification task (labeling pictures of clothes according to 10 categories.) I compared loss curves and accuracy curves over the course of training with each of the 3 weight initialization strategies. See the results for yourself below.

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
