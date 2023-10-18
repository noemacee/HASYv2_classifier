
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import TensorDataset, DataLoader


class MLP(nn.Module):
    """
    An MLP network which does classification.

    It should not use any convolutional layers.
    """
 
    def __init__(self, input_size, n_classes, nbLayer = 3 , activationFunction = F.relu):
        """
        Initialize the network.

        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_size, n_classes, my_arg=32)

        Arguments:
            input_size (int): size of the input
            n_classes (int): number of classes to predict
        """
        super().__init__()
        ##
        ###
        # WRITE YOUR CODE HERE!
        ###
        ##
        self.input_size = input_size
        self.activationFunction = activationFunction
        self.n_classes = n_classes
        self.nbForward = nbLayer -1 
        
        self.layerDim = [0.] * nbLayer

        self.layerDim[0] = input_size

        for i in range(1, nbLayer -1) :
            self.layerDim[i] = 2048
        self.layerDim[-1] = n_classes

        self.layer = [nn.Linear] * (self.nbForward)

        for i in range(self.nbForward): 
            self.layer[i] = nn.Linear(self.layerDim[i], self.layerDim[i+1])

         

        self.params = nn.ParameterList(self.layer)





        

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, D)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        ##
        ###
        # WRITE YOUR CODE HERE!
        ###
        ##
        
    
        for j in range(self.nbForward-1):
            x = self.activationFunction(self.layer[j](x))

        preds = self.layer[self.nbForward-1](x)
        return preds


class CNN(nn.Module):
    """
    A CNN which does classification.

    It should use at least one convolutional layer.
    """

    # Added filters as argument to make it easier to change the number of filters
    # Added padding as argument to make it easier to change the padding
    # Added kernel_size : the size of the convolution kernel : maybe change to a list to have different kernel sizes
    # Added stride : the stride of the convolution : maybe change to a list to have different strides
    # Added max_pooling_kernel : the size of the max_pooling kernel : maybe change to a list to have different kernel sizes
    # Added max_pooling_number : the number of max_pooling layers
    # Added mlp_size : the size of the linear layers

    # Only need to fix the number of filters and the layers of the MLP

    def __init__(self, input_channels, n_classes, filters=[16, 32, 64], conv_kernel_size=3, max_pooling_kernel=3, linear_layers_size=[100, 50,  20], image_width_height=32):
        """
        Initialize the network.

        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_channels, n_classes, my_arg=32)

        Arguments:
            input_channels (int): number of channels in the input
            n_classes (int): number of classes to predict
        """
        super(CNN, self).__init__()

        self.input_channels = input_channels
        self.n_classes = n_classes
        self.filters = filters
        self.conv_kernel_size = conv_kernel_size
        self.max_pooling_kernel = max_pooling_kernel
        self.linear_layers_size = linear_layers_size

        self.convs = nn.ModuleList()
        prev_channels = input_channels
        for filter_size in filters:
            self.convs.append(nn.Conv2d(
                prev_channels, filter_size, conv_kernel_size, padding=conv_kernel_size//2))
            prev_channels = filter_size

        self.linears = nn.ModuleList()

        prev_size = filters[-1] * ((image_width_height //
                                   (max_pooling_kernel ** len(filters))) ** 2)
        for linear_size in linear_layers_size:
            self.linears.append(nn.Linear(prev_size, linear_size))
            prev_size = linear_size

    def forward(self, x, max_pooling_kernel=2):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        for conv in self.convs:
            x = F.relu(conv(x))
            x = F.max_pool2d(x, self.max_pooling_kernel)

        x = x.view(x.size(0), -1)

        for linear in self.linears[:-1]:
            x = F.relu(linear(x))

        preds = self.linears[-1](x)

        return preds


class Trainer(object):
    """
    Trainer class for the deep networks.

    It will also serve as an interface between numpy and pytorch.
    """

    def __init__(self, model, lr, epochs, batch_size):
        """
        Initialize the trainer object for a given model.

        Arguments:
            model (nn.Module): the model to train
            lr (float): learning rate for the optimizer
            epochs (int): number of epochs of training
            batch_size (int): number of data points in each batch
        """
        self.lr = lr
        self.epochs = epochs
        self.model = model
        self.batch_size = batch_size

        self.criterion = nn.CrossEntropyLoss()
        # Creates a state-less Stochastic Gradient Descent. Which one could be the best ?
        self.optimizer = torch.optim.SGD(
            model.parameters(), lr=lr)  # WRITE YOUR CODE HERE

    def train_all(self, dataloader):
        """
        Fully train the model over the epochs. 

        In each epoch, it calls the functions "train_one_epoch". If you want to
        add something else at each epoch, you can do it here.

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        for ep in range(self.epochs):
            self.train_one_epoch(dataloader)

            # WRITE YOUR CODE HERE if you want to do add else at each epoch

    def train_one_epoch(self, dataloader):
        """
        Train the model for ONE epoch.

        Should loop over the batches in the dataloader. (Recall the exercise session!)
        Don't forget to set your model to training mode, i.e., self.model.train()!

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        ##
        ###
        # WRITE YOUR CODE HERE!
        ###
        ##

        self.model.train()  # set model to training mode
        for it, batch in enumerate(dataloader):
            # Fet the inputs and labels
            inputs, labels = batch

            # Run the forward pass
            logits = self.model.forward(inputs)

            # Compute the loss
            loss = self.criterion(logits, labels)

            # Compute the gradients
            loss.backward()

            # Update the weights
            self.optimizer.step()

            # Reset the gradients
            self.optimizer.zero_grad()

    def predict_torch(self, dataloader):
        """
        Predict the validation/test dataloader labels using the model.

        Hints:
            1. Don't forget to set your model to eval mode, i.e., self.model.eval()!
            2. You can use torch.no_grad() to turn off gradient computation, 
            which can save memory and speed up computation. Simply write:
                with torch.no_grad():
                    # Write your code here.

        Arguments:
            dataloader (DataLoader): dataloader for validation/test data
        Returns:
            pred_labels (torch.tensor): predicted labels of shape (N,),
                with N the number of data points in the validation/test data.
        """
        ##
        ###
        # WRITE YOUR CODE HERE!
        ###
        ##
        self.model.eval()  # set model to evaluation mode
        pred_labels = torch.tensor([]).long()
        with torch.no_grad():
            for it, batch in enumerate(dataloader):
                x = batch[0]
                pred = self.model(x)
                pred_labels = torch.cat((pred_labels, pred))
        pred_labels = torch.argmax(pred_labels, axis=1)

        return pred_labels
# filter_values = [[16, 32, 64], [32, 64, 128], [64, 128, 256], [128, 256, 512], [256, 512, 1024]]
# learning_rate = 0.001
# epochs = 10

# best_filters = None
# best_accuracy = 0

# # Grid search
# for filters in filter_values:
#     model = YourModel(input_channels, n_classes, filters, conv_kernel_size, max_pooling_kernel, linear_layers_size, image_width_height)

#     model.train(train_data, val_data, epochs, learning_rate)

#     accuracy = model.evaluate(val_data)

#     if accuracy > best_accuracy:
#         best_accuracy = accuracyÅÅÅÅÅÅÅ
#         best_filters = filters

# print(f'Best filters: {best_filters}, accuracy: {best_accuracy}')

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        This serves as an interface between numpy and pytorch.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """
        # First, prepare data for pytorch
        train_dataset = TensorDataset(torch.from_numpy(training_data).float(),
                                      torch.from_numpy(training_labels))
        train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True)

        self.train_all(train_dataloader)

        return self.predict(training_data)

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        This serves as an interface between numpy and pytorch.

        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        # First, prepare data for pytorch
        test_dataset = TensorDataset(torch.from_numpy(test_data).float())
        test_dataloader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False)

        pred_labels = self.predict_torch(test_dataloader)

        # We return the labels after transforming them into numpy array.
        return pred_labels.numpy()
