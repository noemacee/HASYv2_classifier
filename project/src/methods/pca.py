import numpy as np
from matplotlib import pyplot as plt

## MS2

class PCA(object):
    """
    PCA dimensionality reduction class.
    
    Feel free to add more functions to this class if you need,
    but make sure that __init__(), find_principal_components(), and reduce_dimension() work correctly.
    """

    def __init__(self, d):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            d (int): dimensionality of the reduced space
        """
        self.d = d
        
        # the mean of the training data (will be computed from the training data and saved to this variable)
        self.mean = None 
        # the principal components (will be computed from the training data and saved to this variable)
        self.W = None
        print("[INFO] PCA initialized with d = {}".format(self.d))

    def find_principal_components(self, training_data):
        """
        Finds the principal components of the training data and returns the explained variance in percentage.

        IMPORTANT: 
            This function should save the mean of the training data and the kept principal components as
            self.mean and self.W, respectively.

        Arguments:
            training_data (array): training data of shape (N,D)
        Returns:
            exvar (float): explained variance of the kept dimensions (in percentage, i.e., in [0,100])
        """
        # Setup
        print("[INFO] Finding principal components...")
        training_data = training_data.astype(float)
        dimension = training_data.shape[1]
        #print("[INFO] Training data #samples: {}, #dimensions: {} -> images of shape {}".format(training_data.shape[0], training_data.shape[1], (int(np.sqrt(training_data.shape[1])), int(np.sqrt(training_data.shape[1])))))
        training_data = (training_data - np.mean(training_data, axis=0)) / np.std(training_data, axis=0)
        
        # Compute the covariance matrix and eigenvecs
        covariance_matrix = np.cov(training_data.T)
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # Prepare for usage
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        # Save the mean and the principal components
        self.mean = np.mean(training_data, axis=0)
        self.W = sorted_eigenvectors[:, :self.d]

        # Compute cummulatives
        total_variance = np.sum(sorted_eigenvalues)
        cumulative_explained_variance = np.cumsum(sorted_eigenvalues) / total_variance * 100

        # Plot cumulative explained variance
        plt.plot(cumulative_explained_variance)
        plt.xlabel('Number of components')
        plt.ylabel('Cumulative explained variance')
        plt.plot(self.d, cumulative_explained_variance[self.d - 1], 'ro')
        plt.show()

        # Output the explained variance
        print("[INFO] Explained variance of the kept {} dimensions: {}%".format(self.d, cumulative_explained_variance[self.d - 1]))
        return cumulative_explained_variance[self.d - 1]

    def reduce_dimension(self, data):
        """
        Reduce the dimensionality of the data using the previously computed components.

        Arguments:
            data (array): data of shape (N,D)
        Returns:
            data_reduced (array): reduced data of shape (N,d)
        """
        # Setup
        print("[INFO] Reducing dimensionality...")
        data = data.astype(float)
        data = (data - self.mean) / np.std(data, axis=0)
        data_reduced = np.dot(data, self.W)
        print("[INFO] Data reduced to shape {}".format(data_reduced.shape))
        return data_reduced
        

