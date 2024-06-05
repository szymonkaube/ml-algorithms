import numpy as np


class CustomKMeans():
    """
    Custom implementation of the K-means clustering algorithm.
    
    Attributes:
        k (int): Number of clusters.
        init (str): The type of initialization.
        n_init (int): Number of times the algorithm will run with different centroid seeds.
        tol (float): Tolerance to declare convergence.
        n_iter (int): Maximum number of iterations for a single run.
    """

    def __init__(self, k: int, init: str, n_init: int=10, tol: float=1e-3, n_iter: int=300):
        """
        Initialize the K-means algorithm with specified parameters.
        
        Args:
            k (int): Number of clusters to form.
            init (str): The type of cluster initialization: "kmeans++" or "naive".
            n_init (int, optional): Number of times the algorithm will run with different centroid seeds. Default is 10.
            tol (float, optional): Tolerance to declare convergence. Default is 1e-3.
            n_iter (int, optional): Maximum number of iterations for a single run. Default is 300.
        """
        self.k = k
        self.init = init
        self.n_init = n_init
        self.tol = tol
        self.n_iter = n_iter


    def __naive_init_centroids(self, X: np.array):
        """
        Initialize centroids by randomly selecting k data points from the dataset.
        
        Args:
            X (numpy.ndarray): The dataset from which centroids are initialized.
        
        Returns:
            numpy.ndarray: An array of initial centroids.
        """
        centroids_idx = np.random.choice(len(X) - 1, size=self.k, replace=False)
        return X[centroids_idx, :]
    

    def __plusplus_init_centroids(self, X):
        """
        Initializes cluster centroids for k-means clustering using the k-means++ algorithm.

        Args:
            X (numpy.ndarray): The input data matrix of shape (n_samples, n_features).

        Returns:
            numpy.ndarray: An array of shape (k, n_features) containing the initialized centroids.
        """
        n_samples, n_features = X.shape
        # Initialize first centroid
        centroids = np.zeros((self.k, n_features))
        centroid_idx = np.random.choice(n_samples)
        centroids[0] = X[centroid_idx]
        # Calculate the distances to the first centroid
        distances = self.__calc_distance(X, centroids[0, None]).reshape(-1)
        for i in range(1, self.k):
            # Calculate the probabilities of being the new centroid
            probs = distances / distances.sum()
            # Randomly choose the new centroid according to the probabilities
            new_centroid_idx = np.random.choice(n_samples, p=probs)
            centroids[i] = X[new_centroid_idx]
            # Set distances to be the minimal distance to all centroids
            distances = np.minimum(distances, self.__calc_distance(X, centroids[i, None]).reshape(-1))
        return centroids
    

    def __calc_distance(self, X: np.array, Y: np.array):
        """
        Calculate the Euclidean distance between two sets of points.
        
        Args:
            X (numpy.ndarray): First set of points.
            Y (numpy.ndarray): Second set of points.
        
        Returns:
            numpy.ndarray: A distance matrix.
        """
        X_norms = np.diag(X @ X.T)
        Y_norms = np.diag(Y @ Y.T)
        return X_norms[:, np.newaxis] + Y_norms[np.newaxis, :] - 2 * X @ Y.T
    

    def __calc_inertia(self, X: np.array, centroids: np.array, labels: np.array):
        """
        Calculate the inertia (within-cluster sum of squares) for the given clustering.
        
        Args:
            X (numpy.ndarray): The dataset.
            centroids (numpy.ndarray): Current centroids.
            labels (numpy.ndarray): Labels indicating the cluster assignment for each data point.
        
        Returns:
            float: The inertia value.
        """
        inertia = 0
        for i in range(self.k):
            inertia += self.__calc_distance(X[labels == i], centroids[i, None]).sum()
        return inertia


    def fit(self, X: np.array):
        """
        Fit the K-means algorithm to the dataset X.
        
        Args:
            X (numpy.ndarray): The dataset to be clustered.
        
        Returns:
            CustomKMeans: The instance of the class with the best clustering result.
        """
        self.best_inertia = np.inf
        self.best_centroids = np.zeros((self.k, X.shape[1]))
        self.best_labels = np.zeros(X.shape[0])
        self.best_n_iter_to_converge = 0
        
        # Run k-means n_init of times and choose the result with the best inertia
        for _ in range(self.n_init):
            # Initialize centroids
            if self.init == "kmeans++":
                centroids = self.__plusplus_init_centroids(X)
            else:
                centroids = self.__naive_init_centroids(X)
            change = np.inf
            n_iter_to_converge = 0
            while change > self.tol and n_iter_to_converge < self.n_iter:
                # Calculate distances to centroids
                dists_to_centroids = self.__calc_distance(X, centroids)
                # Assign points to the closest centroid
                labels = np.argmin(dists_to_centroids, axis=1)
                # Update centroids
                new_centroids = np.array([X[labels == label].mean(axis=0) for label in np.unique(labels)])
                change = self.__calc_distance(centroids, new_centroids).diagonal().sum()
                centroids = new_centroids

                n_iter_to_converge += 1

            inertia = self.__calc_inertia(X, centroids, labels)
            if inertia < self.best_inertia:
                self.best_inertia = inertia
                self.best_centroids = centroids
                self.best_labels = labels
                self.best_n_iter_to_converge = n_iter_to_converge
        return self