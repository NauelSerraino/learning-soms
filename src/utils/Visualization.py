import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from minisom import MiniSom

class Visualization(MiniSom):
    def __init__(
            self,
            data:pd.DataFrame,
            target:np.ndarray,
            label_names:dict,
            best_params:dict,
    ):
        super().__init__(
            best_params['n_neurons'],
            best_params['n_neurons'],
            data.shape[1],
            sigma=best_params['sigma'],
            learning_rate=best_params['learning_rate'],
            neighborhood_function=best_params['neighborhood_function'],
            random_seed=best_params['random_seed'],
            topology=best_params['topology']
        )

        self.data = data
        self.target = target
        self.label_names = label_names

    def plot_pca_initial_weights(self):
        self.pca_weights_init(self.data)
        weights = self.get_weights()
        weights_flattened = weights.reshape(-1, self.data.shape[1])
        pca = PCA(n_components=2)
        weights_pca = pca.fit_transform(weights_flattened)
        data_pca = pca.fit_transform(self.data)

        plt.figure(figsize=(9, 9))
        plt.scatter(weights_pca[:, 0], weights_pca[:, 1], c='r')
        plt.scatter(data_pca[:, 0], data_pca[:, 1], c='b')
        plt.legend(['Initial Weights', 'Data'])
        plt.show()

    def plot_pca_final_weights(self):
        self.pca_weights_init(self.data)
        self.train(self.data, 1000, verbose=True)
        weights = self.get_weights()
        weights_flattened = weights.reshape(-1, self.data.shape[1])
        pca = PCA(n_components=2)
        weights_pca = pca.fit_transform(weights_flattened)
        data_pca = pca.fit_transform(self.data)

        plt.figure(figsize=(9, 9))
        plt.scatter(weights_pca[:, 0], weights_pca[:, 1], c='r')
        plt.scatter(data_pca[:, 0], data_pca[:, 1], c='b')
        plt.legend(['Final Weights', 'Data'])
        plt.show()

    def plot_som(self, som:object, title:str):
        plt.figure(figsize=(8, 8))
        plt.pcolor(som.distance_map(scaling="mean").T, cmap='bone_r')
        plt.colorbar()
        plt.title(title)
        plt.show()

    def plot_som_with_target(self, som:object, data:np.array, target:np.ndarray):
        plt.figure(figsize=(12, 9))
        plt.pcolor(som.distance_map(scaling="mean").T, cmap='bone_r')
        plt.colorbar()
        plt.title("Distance Map with Target")

        markers = ['o', 's', 'D']
        colors = ['C0', 'C1', 'C2']

        coord_count = {}

        for cnt, xx in enumerate(data):
            w = som.winner(xx)
            coord = tuple(w)
            if coord not in coord_count:
                coord_count[coord] = [0] * len(markers)
            coord_count[coord][
                target[cnt] - 1] += 1

        for cnt, xx in enumerate(data):
            w = som.winner(xx)
            coord = tuple(w)
            marker_size = 6 + coord_count[coord][
                target[cnt] - 1] * 2
            plt.plot(
                w[0] + 0.5,
                w[1] + 0.5,
                markers[target[cnt] - 1],
                markerfacecolor='None',
                markeredgecolor=colors[target[cnt] - 1],
                markersize=marker_size,
                markeredgewidth=2,
                alpha=1
            )

        plt.show()

    def plot_kmeans_on_som_distance_map(self, som, clusters):
        weights = som.get_weights()
        weights_reshaped = weights.reshape(-1, weights.shape[2])

        kmeans = KMeans(n_clusters=clusters, random_state=0)
        kmeans.fit(weights_reshaped)
        labels = kmeans.labels_

        plt.figure(figsize=(10, 10))
        plt.pcolor(som.distance_map().T, cmap='bone_r', alpha=0.5)
        plt.colorbar()

        label_colors = plt.cm.nipy_spectral(labels.astype(float) / clusters)
        label_colors = label_colors.reshape(som.get_weights().shape[:2] + (4,))

        for x in range(som.get_weights().shape[0]):
            for y in range(som.get_weights().shape[1]):
                plt.gca().add_patch(plt.Rectangle(
                    (x, y),
                    1,
                    1,
                    facecolor=label_colors[x, y],
                    edgecolor='none',
                    alpha=0.3
                ))

        plt.title("SOM Distance Map with K-Means Clusters")
        plt.show()
        plt.show()


