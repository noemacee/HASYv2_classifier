import argparse

import numpy as np
import random as rd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from src.data import load_data
from src.methods.dummy_methods import DummyClassifier
from src.methods.logistic_regression import LogisticRegression
from src.methods.pca import PCA
from src.methods.deep_network import MLP, CNN, Trainer
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, get_n_classes, split_train_test


def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end 
                          of this file). Their value can be accessed as "args.argument".
    """
    # 1. First, we load our data and flatten the images into vectors
    xtrain, xtest, ytrain, ytest = load_data(args.data)
    # 80     20     80       20
    xtrain = xtrain.reshape(xtrain.shape[0], -1)
    xtest = xtest.reshape(xtest.shape[0], -1)
    data_size = len(xtrain) + len(xtest)

    # 2. Then we must prepare it. This is were you can create a validation set,
    #  normalize, add bias, etc.

    # Make a validation set
    if not args.test:
        xtrain, xtest, ytrain, ytest = split_train_test(
            xtrain, ytrain, test_size=0.2)

    print(
        f"[INFO] Data loaded: xtrain.shape = {xtrain.shape} - ytrain.shape = {ytrain.shape}")
    print(
        f"[INFO] Data loaded: xtest.shape = {xtest.shape} - ytest.shape = {ytest.shape}")
    print(
        f"[INFO] Data composition: train = {len(xtrain)/data_size:.2f} - test = {len(xtest)/data_size:.2f}")

    # WRITE YOUR CODE HERE to do any other data processing

    # Dimensionality reduction (MS2)
    if args.use_pca:
        # PCA for plot
        pca_obj = PCA(d=3)
        pca_obj.find_principal_components(xtrain)
        xtrain_plot = pca_obj.reduce_dimension(xtrain)
        xtest_plot = pca_obj.reduce_dimension(xtest)
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        colors = [color for color in mcolors.CSS4_COLORS]
        rd.shuffle(colors)
        for i, point in enumerate(xtest_plot):
            ax.scatter3D(point[0], point[1], point[2], color=colors[ytest[i]])
        plt.show()

        # PCA for classification
        pca_obj = PCA(d=args.pca_d)
        pca_obj.find_principal_components(xtrain)
        xtrain = pca_obj.reduce_dimension(xtrain)
        xtest = pca_obj.reduce_dimension(xtest)
        # WRITE YOUR CODE HERE: use the PCA object to reduce the dimensionality of the data

    # 3. Initialize the method you want to use.
    if args.method == "logistic_regression":
        if not args.test:  # use the validation to find the best lr
            # 20 values between 10^-6 and 10^0
            learning_rates = np.logspace(-6, 0, 20)
            accuracies = []
            for lr in learning_rates:
                model = LogisticRegression(lr=lr, max_iters=args.max_iters)
                model.fit(xtrain, ytrain)
                preds = model.predict(xtest)
                acc = accuracy_fn(preds, ytest)
                accuracies.append(acc)
            best_lr = learning_rates[np.argmax(accuracies)]
            print(f"Best learning rate: {best_lr}")

        if args.test:  # Train k-means model using best k on 20 iteration to find the best among random start
            best_lr = args.lr
            model = LogisticRegression(lr=best_lr, max_iters=args.max_iters)
            preds_train = model.fit(xtrain, ytrain)
            acc = accuracy_fn(preds_train, ytrain)
            macrof1 = macrof1_fn(preds_train, ytrain)
            print(
                f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

            preds = model.predict(xtest)
            acc = accuracy_fn(preds, ytest)
            macrof1 = macrof1_fn(preds, ytest)
            print(
                f"Test set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    # Neural Networks (MS2)
    if args.method == "nn":

        print("Using deep network")

        # Prepare the model (and data) for Pytorch
        # Note: you might need to reshape the image data depending on the network you use!
        n_classes = get_n_classes(ytrain)
        if args.nn_type == "mlp":
            nb_hidden = 10

            if not args.test:
                tab = [F.relu, F.tanh, F.sigmoid]
                train_acc = np.empty(len(tab) * nb_hidden)
                val_acc = np.empty(len(tab) * nb_hidden)  # use the validation to find the best lr
                
                for i in range(len(tab)):
                    for j in range(3 ,nb_hidden): # number of hideen layer maybe add a command line argument for this
                        model = MLP(xtrain.shape[1], n_classes, j ,tab[i])  # WRITE YOUR CODE HERE
                        method_obj = Trainer(
                                model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size)
                        # Model prediction
                        index = i*j+j
                        preds_train = method_obj.fit(xtrain, ytrain)
                        train_acc[index] = accuracy_fn(preds_train, ytrain) # need a separation function for the validation set
                        preds_val = method_obj.predict(xtrain)
                        val_acc[index] = accuracy_fn(preds_val, ytest)

                bestModel = np.argmax(val_acc)
                bestActivation = tab[bestModel / len(tab)]
                bestHidden = bestModel % len(tab)
                
                print("Train accuracy = ", train_acc)
                print("Validation accuracy = ", val_acc)
                print("Best activation function = ", bestActivation)
                print("Best number of hidden layer = ", bestHidden)
            else :
                model = MLP(32*32, n_classes)
                summary(model)

        elif args.nn_type == "cnn": 
            xtrain = xtrain.reshape(xtrain.shape[0], 1, 32, 32)
            xtest = xtest.reshape(xtest.shape[0], 1, 32, 32)

            if not args.test:  # use the validation to find the best lr

                filters = [(2, 4, 8), (4, 8, 16), (5, 10, 20),
                           (10, 20, 40), (16, 32, 64), (32, 64, 128)]
                train_acc = np.empty(len(filters))
                val_acc = np.empty(len(filters))

                for it, filter in enumerate(filters):
                    print("Running one filter version of CNN with filter size:", filter)
                    # Model fit and train
                    model = CNN(xtrain.shape[1], n_classes, filters=filter)
                    method_obj = Trainer(
                        model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size)
                    # Model prediction
                    preds_train = method_obj.fit(xtrain, ytrain)
                    train_acc[it] = accuracy_fn(preds_train, ytrain)
                    preds_val = method_obj.predict(xtrain)
                    val_acc[it] = accuracy_fn(preds_val, ytrain) # TODO ytest not ytrain

                bestModel = np.argmax(val_acc)
                print("Train accuracy = ", train_acc)
                print("Validation accuracy = ", val_acc)

                plt.plot(train_acc, color="blue", marker='x')
                plt.xticks(range(len(filters)), labels=[
                           '(2,4,8)', '(4,8,16)', '(5,10,20)', '(10, 20, 40)', '(16, 32, 64)', '(32, 64, 128)'])
                plt.title("CNN training accuracies")
                plt.xlabel('Filters')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.savefig('trainingaccuracies.png')
                plt.show()

                plt.plot(val_acc, color="red", marker='o')
                plt.xticks(range(len(filters)), labels=[
                           '(2,4,8)', '(4,8,16)', '(5,10,20)', '(10, 20, 40)', '(16, 32, 64)', '(32, 64, 128)'])
                plt.title("CNN validation accuracies")
                plt.ylabel('Accuracy')
                plt.xlabel('Filters')
                plt.legend()
                plt.savefig('validationaccuracies.png')
                plt.show()
                model = CNN(xtrain.shape[1], n_classes,
                            filters=filters[bestModel])
                summary(model)

            else:
                model = CNN(1, n_classes)
                summary(model)

        # Trainer object
        method_obj = Trainer(
            model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size)

    # Follow the "DummyClassifier" example for your methods (MS1)
    elif args.method == "dummy_classifier":
        method_obj = DummyClassifier(arg1=1, arg2=2)

    # 4. Train and evaluate the method

    # Fit (:=train) the method on the training data
    preds_train = method_obj.fit(xtrain, ytrain)

    # Predict on unseen data
    preds = method_obj.predict(xtest)

    # Report results: performance on train and valid/test sets
    acc = accuracy_fn(preds_train, ytrain)
    macrof1 = macrof1_fn(preds_train, ytrain)
    print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    acc = accuracy_fn(preds, ytest)
    macrof1 = macrof1_fn(preds, ytest)
    print(f"Test set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    # WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.


if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="dataset_HASYv2", type=str,
                        help="the path to wherever you put the data, if it's in the parent folder, you can use ../dataset_HASYv2")
    parser.add_argument('--method', default="dummy_classifier", type=str,
                        help="dummy_classifier / kmeans / logistic_regression / svm / nn (MS2)")
    parser.add_argument('--K', type=int, default=10,
                        help="number of clusters for K-Means")
    parser.add_argument('--lr', type=float, default=1e-5,
                        help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=100,
                        help="max iters for methods which are iterative")
    parser.add_argument('--test', action="store_true",
                        help="train on whole training data and evaluate on the test data, otherwise use a validation set")
    parser.add_argument('--svm_c', type=float, default=1.,
                        help="Constant C in SVM method")
    parser.add_argument('--svm_kernel', default="linear",
                        help="kernel in SVM method, can be 'linear' or 'rbf' or 'poly'(polynomial)")
    parser.add_argument('--svm_gamma', type=float, default=1.,
                        help="gamma prameter in rbf/polynomial SVM method")
    parser.add_argument('--svm_degree', type=int, default=1,
                        help="degree in polynomial SVM method")
    parser.add_argument('--svm_coef0', type=float, default=0.,
                        help="coef0 in polynomial SVM method")

    # WRITE YOUR CODE HERE: feel free to add more arguments here if you need!

    # MS2 arguments
    parser.add_argument('--use_pca', action="store_true", help="to enable PCA")
    parser.add_argument('--pca_d', type=int, default=200,
                        help="output dimensionality after PCA")
    parser.add_argument('--nn_type', default="mlp",
                        help="which network to use, can be 'mlp' or 'cnn'")
    parser.add_argument('--nn_batch_size', type=int,
                        default=64, help="batch size for NN training")

    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)
