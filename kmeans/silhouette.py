import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calc_distance(X, Y):
    """
    Calculate the Euclidean distance between each pair of the two collections of inputs.

    Parameters:
        X (numpy.ndarray): An m x n array where each row represents a point in n-dimensional space.
        Y (numpy.ndarray): A p x n array where each row represents a point in n-dimensional space.

    Returns:
        numpy.ndarray: An m x p array where the element at [i, j] represents the Euclidean distance
            between the i-th point in X and the j-th point in Y.
    """
    X_norms = np.diag(X @ X.T)
    Y_norms = np.diag(Y @ Y.T)
    return np.sqrt(X_norms[:, np.newaxis] + Y_norms[np.newaxis, :] - 2 * X @ Y.T)


def silhouette_coefficients(X, labels):
    """
    Calculate the silhouette coefficients for each point in a dataset.

    Parameters:
        X (numpy.ndarray): An m x n array where each row represents a point in n-dimensional space.
        labels (numpy.ndarray): An array of length m where each element is the cluster label for the corresponding point in X.

    Returns:
        numpy.ndarray: An array of length m where the i-th element is the silhouette coefficient for the i-th point in X.
    """
    distance_matrix = calc_distance(X, X)

    silhouette_vals = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        # Get the points that belong to the same cluster as point i
        same_cluster = labels == labels[i]
        # Get the points that belong to different clusters from point i
        other_clusters = labels != labels[i]
        a_i = np.mean(distance_matrix[i, same_cluster])
        b_i = np.min([np.mean(distance_matrix[i, labels == l]) for l in np.unique(labels[other_clusters])])
        # Calculate the silhouette coefficient for point i
        silhouette_vals[i] = (b_i - a_i) / max(a_i, b_i)
        
    return silhouette_vals


def plot_silhouette(silhouette_coeffs, labels, ax):
    """
    Plot the silhouette coefficients for each point in a dataset.

    Parameters:
        silhouette_coeffs (numpy.ndarray): An array of silhouette coefficients for each point in the dataset.
        labels (numpy.ndarray): An array of cluster labels for each point in the dataset.
        ax (matplotlib.axes.Axes): The axes on which to plot the silhouette plot.
    
    Returns:
        matplotlib.axes.Axes: The axes on which the silhouette plot is drawn.
    """
    silhouette_df = pd.DataFrame({"silhouette": silhouette_coeffs,
                                  "label": labels})
    silhouette_df_sorted = silhouette_df.sort_values(by=["label", "silhouette"])

    average_score = silhouette_df_sorted.silhouette.mean()

    y_lower = 10
    n_clusters = len(silhouette_df_sorted.label.unique())
    for i in range(n_clusters):
        ith_cluster_silhouette = silhouette_df_sorted[silhouette_df_sorted.label == i].silhouette

        size_cluster_i = ith_cluster_silhouette.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette,
                        facecolor=color, edgecolor=color, alpha=0.7)

        y_lower = y_upper + 10

    # The vertical line for average silhouette score of all the values
    ax.axvline(x=average_score, color="red", linestyle="--")
    ax.set_yticks([])
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xlim([0, 1])

    return ax